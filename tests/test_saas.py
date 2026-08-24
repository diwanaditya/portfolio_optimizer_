import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from portfolio_optimizer.saas.plans import PlanTier, get_plan, plan_from_string, PLANS
from portfolio_optimizer.saas.tenancy import TenancyStore
from portfolio_optimizer.saas.billing import handle_webhook_event


class TestPlans:
    def test_all_three_tiers_defined(self):
        assert set(PLANS.keys()) == {PlanTier.FREE, PlanTier.PRO, PlanTier.INSTITUTIONAL}

    def test_higher_tiers_have_higher_limits(self):
        free = get_plan(PlanTier.FREE)
        pro = get_plan(PlanTier.PRO)
        inst = get_plan(PlanTier.INSTITUTIONAL)
        assert free.requests_per_minute < pro.requests_per_minute < inst.requests_per_minute
        assert free.max_assets_per_request < pro.max_assets_per_request < inst.max_assets_per_request
        assert free.monthly_price_usd < pro.monthly_price_usd < inst.monthly_price_usd

    def test_free_plan_disallows_live_trading(self):
        assert get_plan(PlanTier.FREE).live_trading_allowed is False

    def test_paid_plans_allow_live_trading(self):
        assert get_plan(PlanTier.PRO).live_trading_allowed is True
        assert get_plan(PlanTier.INSTITUTIONAL).live_trading_allowed is True

    def test_plan_from_string_valid(self):
        assert plan_from_string("pro").tier == PlanTier.PRO

    def test_plan_from_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown plan tier"):
            plan_from_string("not_a_real_plan")


class TestTenancyStore:
    def _store(self, tmp_path):
        return TenancyStore(str(tmp_path / "test.db"))

    def test_create_and_retrieve_customer(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        fetched = store.get_customer(c.customer_id)
        assert fetched.email == "a@b.com"
        assert fetched.plan_tier == PlanTier.FREE

    def test_duplicate_email_rejected(self, tmp_path):
        store = self._store(tmp_path)
        store.create_customer("dup@b.com", PlanTier.FREE)
        with pytest.raises(Exception):  # sqlite3.IntegrityError via UNIQUE constraint
            store.create_customer("dup@b.com", PlanTier.PRO)

    def test_issued_key_is_never_stored_in_plaintext(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        raw_key = store.issue_api_key(c.customer_id)
        with store._connect() as conn:
            row = conn.execute("SELECT key_hash FROM api_keys").fetchone()
        assert raw_key != row["key_hash"]
        assert raw_key not in row["key_hash"]

    def test_validate_unknown_key_rejected(self, tmp_path):
        store = self._store(tmp_path)
        result = store.validate_api_key("not_a_real_key")
        assert result.valid is False
        assert "Unknown" in result.reason

    def test_validate_issued_key_succeeds(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.PRO)
        raw_key = store.issue_api_key(c.customer_id)
        result = store.validate_api_key(raw_key)
        assert result.valid is True
        assert result.plan.tier == PlanTier.PRO

    def test_revoked_key_rejected(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        raw_key = store.issue_api_key(c.customer_id)
        store.revoke_api_key(raw_key)
        result = store.validate_api_key(raw_key)
        assert result.valid is False
        assert "revoked" in result.reason.lower()

    def test_endpoint_gating_by_plan(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        raw_key = store.issue_api_key(c.customer_id)

        allowed = store.validate_api_key(raw_key, endpoint="markowitz")
        blocked = store.validate_api_key(raw_key, endpoint="black-litterman")
        assert allowed.valid is True
        assert blocked.valid is False
        assert "not available" in blocked.reason

    def test_plan_upgrade_unlocks_endpoint(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        raw_key = store.issue_api_key(c.customer_id)
        store.update_customer_plan(c.customer_id, PlanTier.PRO)
        result = store.validate_api_key(raw_key, endpoint="black-litterman")
        assert result.valid is True

    def test_inactive_subscription_rejected_but_plan_still_reported(self, tmp_path):
        """Regression test for a real bug caught during development: an
        inactive subscription used to return plan=None, losing the
        context needed to tell the customer what plan they're actually
        on (e.g. 'downgraded to Free, upgrade to restore access').
        """
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.PRO)
        raw_key = store.issue_api_key(c.customer_id)
        store.update_customer_plan(c.customer_id, PlanTier.FREE, subscription_status="canceled")
        result = store.validate_api_key(raw_key)
        assert result.valid is False
        assert result.plan is not None
        assert result.plan.tier == PlanTier.FREE

    def test_usage_metering_counts_events(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        for _ in range(7):
            store.record_usage(c.customer_id, "markowitz")
        assert store.usage_this_month(c.customer_id) == 7

    def test_quota_check_respects_plan_limit(self, tmp_path):
        store = self._store(tmp_path)
        c = store.create_customer("a@b.com", PlanTier.FREE)
        plan = get_plan(PlanTier.FREE)
        for _ in range(plan.requests_per_month + 5):
            store.record_usage(c.customer_id, "markowitz")
        within_quota, used = store.check_monthly_quota(c.customer_id, plan)
        assert within_quota is False
        assert used > plan.requests_per_month


class TestBillingWebhookHandling:
    def _store(self, tmp_path):
        return TenancyStore(str(tmp_path / "test.db"))

    def test_new_customer_checkout_creates_customer_and_key(self, tmp_path):
        store = self._store(tmp_path)
        event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_abc", "customer_details": {"email": "new@x.com"},
                "metadata": {"plan_tier": "pro"}, "subscription": "sub_xyz",
            }},
        }
        result = handle_webhook_event(event, store)
        assert result["action"] == "customer_created"
        assert "api_key" in result

        customer = store.get_customer_by_stripe_id("cus_abc")
        assert customer.plan_tier == PlanTier.PRO
        assert customer.subscription_status == "active"

    def test_existing_customer_checkout_updates_plan_not_duplicate(self, tmp_path):
        store = self._store(tmp_path)
        event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_abc", "customer_details": {"email": "new@x.com"},
                "metadata": {"plan_tier": "pro"}, "subscription": "sub_xyz",
            }},
        }
        handle_webhook_event(event, store)
        # simulate them later upgrading to institutional
        event2 = dict(event)
        event2["data"] = {"object": {**event["data"]["object"], "metadata": {"plan_tier": "institutional"}}}
        result2 = handle_webhook_event(event2, store)
        assert result2["action"] == "plan_updated"

        customer = store.get_customer_by_stripe_id("cus_abc")
        assert customer.plan_tier == PlanTier.INSTITUTIONAL

    def test_subscription_cancelled_downgrades_to_free(self, tmp_path):
        store = self._store(tmp_path)
        create_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_abc", "customer_details": {"email": "new@x.com"},
                "metadata": {"plan_tier": "pro"}, "subscription": "sub_xyz",
            }},
        }
        handle_webhook_event(create_event, store)

        cancel_event = {"type": "customer.subscription.deleted",
                         "data": {"object": {"customer": "cus_abc"}}}
        result = handle_webhook_event(cancel_event, store)
        assert result["action"] == "downgraded_to_free"

        customer = store.get_customer_by_stripe_id("cus_abc")
        assert customer.plan_tier == PlanTier.FREE
        assert customer.subscription_status == "canceled"

    def test_subscription_past_due_updates_status_not_plan(self, tmp_path):
        store = self._store(tmp_path)
        create_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_abc", "customer_details": {"email": "new@x.com"},
                "metadata": {"plan_tier": "pro"}, "subscription": "sub_xyz",
            }},
        }
        handle_webhook_event(create_event, store)

        past_due_event = {"type": "customer.subscription.updated",
                           "data": {"object": {"customer": "cus_abc", "status": "past_due"}}}
        result = handle_webhook_event(past_due_event, store)
        assert result["status"] == "past_due"

        customer = store.get_customer_by_stripe_id("cus_abc")
        assert customer.plan_tier == PlanTier.PRO  # plan unchanged
        assert customer.subscription_status == "past_due"  # status reflects the problem

    def test_unrecognized_event_type_ignored_gracefully(self, tmp_path):
        store = self._store(tmp_path)
        weird_event = {"type": "some.event.we.dont.handle", "data": {"object": {}}}
        result = handle_webhook_event(weird_event, store)
        assert result["action"] == "ignored"

    def test_resubscription_with_new_stripe_id_same_email_relinks_not_crashes(self, tmp_path):
        """Regression test for a real bug caught during development: if
        Stripe issues a NEW customer object for someone who already has an
        account here under a different (e.g. earlier, cancelled)
        stripe_customer_id, the original implementation crashed with a raw
        sqlite3.IntegrityError on the email UNIQUE constraint instead of
        recognizing this is the same person and re-linking their account.
        """
        store = self._store(tmp_path)
        first_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_original", "customer_details": {"email": "person@example.com"},
                "metadata": {"plan_tier": "pro"}, "subscription": "sub_original",
            }},
        }
        result1 = handle_webhook_event(first_event, store)
        assert result1["action"] == "customer_created"
        original_customer_id = result1["customer_id"]

        # subscription cancelled, then they resubscribe later -- Stripe
        # gives them a DIFFERENT customer id this time, same email
        cancel_event = {"type": "customer.subscription.deleted",
                         "data": {"object": {"customer": "cus_original"}}}
        handle_webhook_event(cancel_event, store)

        resubscribe_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_brand_new_stripe_id", "customer_details": {"email": "person@example.com"},
                "metadata": {"plan_tier": "institutional"}, "subscription": "sub_new",
            }},
        }
        result2 = handle_webhook_event(resubscribe_event, store)

        assert result2["action"] == "relinked_by_email"
        assert result2["customer_id"] == original_customer_id  # same account, not a duplicate

        customer = store.get_customer(original_customer_id)
        assert customer.plan_tier == PlanTier.INSTITUTIONAL
        assert customer.subscription_status == "active"
        assert customer.stripe_customer_id == "cus_brand_new_stripe_id"  # re-linked to the new Stripe ID


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
