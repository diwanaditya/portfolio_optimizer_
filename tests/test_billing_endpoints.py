import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NOTE: deliberately no module-level os.environ mutation here. An earlier
# version set PORTFOLIO_OPTIMIZER_API_KEYS at import time, which -- because
# pytest imports every test module during collection, BEFORE any test in
# ANY file actually runs -- silently overwrote the value test_api_security.py
# depends on for its own tests, causing spurious failures there with no
# obvious connection to this file. None of the /billing/* endpoints tested
# below actually require that env var (they're either public or gated by
# Stripe/SaaS config instead), so the line was both unnecessary and harmful.
# If a test here ever needs a specific env var, set it via monkeypatch
# inside that test function, never at module level.

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from portfolio_optimizer.api.service import app

client = TestClient(app)

_CHECKOUT_BASE = {
    "email": "test@example.com",
    "success_url": "https://example.com/success",
    "cancel_url": "https://example.com/cancel",
}


class TestCheckoutEndpoint:
    def test_missing_stripe_config_returns_503_not_crash(self, monkeypatch):
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        resp = client.post("/billing/checkout", json={**_CHECKOUT_BASE, "plan": "pro"})
        assert resp.status_code == 503
        assert "STRIPE_SECRET_KEY" in resp.json()["detail"]

    def test_invalid_plan_name_returns_400(self):
        resp = client.post("/billing/checkout", json={**_CHECKOUT_BASE, "plan": "not_real"})
        assert resp.status_code == 400

    def test_free_plan_checkout_rejected(self):
        resp = client.post("/billing/checkout", json={**_CHECKOUT_BASE, "plan": "free"})
        assert resp.status_code == 400
        assert "doesn't need checkout" in resp.json()["detail"]

    def test_successful_checkout_with_mocked_stripe(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
        monkeypatch.setenv("STRIPE_PRICE_ID_PRO", "price_fake123")

        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/fake-session"
        mock_session.id = "cs_fake_123"

        with patch("stripe.checkout.Session.create", return_value=mock_session):
            resp = client.post("/billing/checkout", json={**_CHECKOUT_BASE, "plan": "pro"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["checkout_url"] == "https://checkout.stripe.com/fake-session"
        assert body["session_id"] == "cs_fake_123"

    def test_missing_price_id_returns_clean_error(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
        monkeypatch.delenv("STRIPE_PRICE_ID_PRO", raising=False)
        resp = client.post("/billing/checkout", json={**_CHECKOUT_BASE, "plan": "pro"})
        assert resp.status_code == 503
        assert "STRIPE_PRICE_ID_PRO" in resp.json()["detail"]


class TestWebhookEndpoint:
    def test_unverified_webhook_rejected(self):
        resp = client.post("/billing/webhook", content=b'{"fake": "event"}',
                            headers={"stripe-signature": "invalid"})
        assert resp.status_code == 400

    def test_missing_signature_header_rejected(self):
        resp = client.post("/billing/webhook", content=b'{"fake": "event"}')
        assert resp.status_code == 400

    def test_verified_webhook_processed_and_key_not_echoed(self, monkeypatch):
        """Confirms the one-time API key returned by handle_webhook_event
        for a new customer is NOT echoed back in the webhook response --
        it should only ever reach the customer via a side channel (email),
        never sit in Stripe's webhook delivery logs.
        """
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_fake")

        # Use a unique customer ID per test run rather than a fixed
        # literal: the webhook endpoint uses the API's default (on-disk,
        # working-directory-relative) tenancy DB, which persists across
        # separate pytest invocations in the same directory -- a fixed
        # "cus_test_webhook" ID collided with a customer already created
        # by an earlier run of this same test, causing this test to see
        # "plan_updated" (existing customer) instead of the expected
        # "customer_created" (new customer) outcome it's actually testing.
        unique_customer_id = f"cus_test_webhook_{os.urandom(6).hex()}"
        unique_email = f"webhook_{os.urandom(6).hex()}@test.com"
        fake_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": unique_customer_id, "customer_details": {"email": unique_email},
                "metadata": {"plan_tier": "pro"}, "subscription": "sub_test",
            }},
        }
        with patch("stripe.Webhook.construct_event", return_value=fake_event):
            resp = client.post("/billing/webhook", content=b'{}',
                                headers={"stripe-signature": "valid_enough_for_mock"})
        assert resp.status_code == 200
        body = resp.json()
        assert "api_key" not in body
        assert body["action"] == "customer_created"


class TestPlansEndpoint:
    def test_list_plans_no_auth_required(self):
        resp = client.get("/billing/plans")
        assert resp.status_code == 200

    def test_plans_have_expected_structure(self):
        resp = client.get("/billing/plans")
        plans = resp.json()
        assert len(plans) == 3
        for p in plans:
            assert "tier" in p and "monthly_price_usd" in p and "allowed_endpoints" in p

    def test_plan_prices_increase_with_tier(self):
        resp = client.get("/billing/plans")
        plans = {p["tier"]: p["monthly_price_usd"] for p in resp.json()}
        assert plans["free"] < plans["pro"] < plans["institutional"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
