import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _sample_payload(n_assets=3, n_periods=100, seed=0):
    rng = np.random.default_rng(seed)
    assets = [f"A{i}" for i in range(n_assets)]
    dates = [f"2024-01-{(i % 28) + 1:02d}" for i in range(n_periods)]
    returns = (rng.standard_normal((n_periods, n_assets)) * 0.01).tolist()
    return {"dates": dates, "assets": assets, "returns": returns}


@pytest.fixture(scope="module")
def saas_client():
    """Sets up SaaS-mode environment ONLY for the duration of this
    module's tests, and restores the original environment afterward.

    IMPORTANT: this must be a fixture, not module-level code -- pytest
    imports every test module during its collection phase, BEFORE any
    test actually runs, so module-level `os.environ[...] = ...` mutates
    the environment for the entire test session the moment this file is
    collected, regardless of which file's tests execute "first". That
    was a real bug caught during development: it silently broke
    test_api_security.py's static-API-key-mode tests by clearing
    PORTFOLIO_OPTIMIZER_API_KEYS out from under them. Scoping the
    mutation to a fixture with explicit teardown is the fix.
    """
    tmp_dir = tempfile.mkdtemp()
    original_saas_mode = os.environ.get("PORTFOLIO_OPTIMIZER_SAAS_MODE")
    original_tenancy_db = os.environ.get("PORTFOLIO_OPTIMIZER_TENANCY_DB")
    original_api_keys = os.environ.get("PORTFOLIO_OPTIMIZER_API_KEYS")

    os.environ["PORTFOLIO_OPTIMIZER_SAAS_MODE"] = "1"
    os.environ["PORTFOLIO_OPTIMIZER_TENANCY_DB"] = os.path.join(tmp_dir, "tenancy.db")
    os.environ.pop("PORTFOLIO_OPTIMIZER_API_KEYS", None)

    # Force re-import so the API module's module-level `_tenancy_store = None`
    # singleton and any cached env reads pick up this test's fresh config.
    for mod_name in list(sys.modules):
        if mod_name.startswith("portfolio_optimizer.api"):
            del sys.modules[mod_name]

    from portfolio_optimizer.api.service import app, _get_tenancy_store
    test_client = TestClient(app)

    yield test_client, _get_tenancy_store

    # Restore the environment exactly as it was before this module ran,
    # so later-collected/later-run test modules aren't affected either.
    if original_saas_mode is None:
        os.environ.pop("PORTFOLIO_OPTIMIZER_SAAS_MODE", None)
    else:
        os.environ["PORTFOLIO_OPTIMIZER_SAAS_MODE"] = original_saas_mode
    if original_tenancy_db is None:
        os.environ.pop("PORTFOLIO_OPTIMIZER_TENANCY_DB", None)
    else:
        os.environ["PORTFOLIO_OPTIMIZER_TENANCY_DB"] = original_tenancy_db
    if original_api_keys is None:
        os.environ.pop("PORTFOLIO_OPTIMIZER_API_KEYS", None)
    else:
        os.environ["PORTFOLIO_OPTIMIZER_API_KEYS"] = original_api_keys

    for mod_name in list(sys.modules):
        if mod_name.startswith("portfolio_optimizer.api"):
            del sys.modules[mod_name]


from portfolio_optimizer.saas.plans import PlanTier


@pytest.fixture
def free_customer_key(saas_client):
    client, get_store = saas_client
    store = get_store()
    customer = store.create_customer(f"free_{os.urandom(4).hex()}@test.com", PlanTier.FREE)
    return store.issue_api_key(customer.customer_id), customer


@pytest.fixture
def pro_customer_key(saas_client):
    client, get_store = saas_client
    store = get_store()
    customer = store.create_customer(f"pro_{os.urandom(4).hex()}@test.com", PlanTier.PRO)
    return store.issue_api_key(customer.customer_id), customer


class TestSaaSModeEndToEnd:
    def test_free_plan_allows_markowitz(self, saas_client, free_customer_key):
        client, _ = saas_client
        raw_key, _customer = free_customer_key
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                            headers={"X-API-Key": raw_key})
        assert resp.status_code == 200

    def test_free_plan_blocks_black_litterman_with_403(self, saas_client, free_customer_key):
        client, _ = saas_client
        raw_key, _customer = free_customer_key
        payload = _sample_payload()
        caps = {a: 1.0 for a in payload["assets"]}
        resp = client.post("/optimize/black-litterman",
                            json={"payload": payload, "market_caps": caps},
                            headers={"X-API-Key": raw_key})
        assert resp.status_code == 403
        assert "not available" in resp.json()["detail"]

    def test_pro_plan_allows_black_litterman(self, saas_client, pro_customer_key):
        client, _ = saas_client
        raw_key, _customer = pro_customer_key
        payload = _sample_payload()
        caps = {a: 1.0 for a in payload["assets"]}
        resp = client.post("/optimize/black-litterman",
                            json={"payload": payload, "market_caps": caps},
                            headers={"X-API-Key": raw_key})
        assert resp.status_code == 200

    def test_unknown_key_rejected_with_401(self, saas_client):
        client, _ = saas_client
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                            headers={"X-API-Key": "totally_made_up_key"})
        assert resp.status_code == 401

    def test_missing_key_rejected_with_401(self, saas_client):
        client, _ = saas_client
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()})
        assert resp.status_code == 401

    def test_revoked_key_rejected(self, saas_client, free_customer_key):
        client, get_store = saas_client
        raw_key, customer = free_customer_key
        get_store().revoke_api_key(raw_key)
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                            headers={"X-API-Key": raw_key})
        assert resp.status_code == 401

    def test_usage_is_actually_recorded_per_request(self, saas_client, free_customer_key):
        client, get_store = saas_client
        raw_key, customer = free_customer_key
        store = get_store()
        before = store.usage_this_month(customer.customer_id)
        client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                    headers={"X-API-Key": raw_key})
        after = store.usage_this_month(customer.customer_id)
        assert after == before + 1

    def test_failed_endpoint_gate_does_not_record_usage(self, saas_client, free_customer_key):
        """A request blocked by plan gating (403) shouldn't count against
        quota -- you shouldn't be charged usage for a request that never
        actually ran the optimizer.
        """
        client, get_store = saas_client
        raw_key, customer = free_customer_key
        store = get_store()
        before = store.usage_this_month(customer.customer_id)
        payload = _sample_payload()
        caps = {a: 1.0 for a in payload["assets"]}
        client.post("/optimize/black-litterman", json={"payload": payload, "market_caps": caps},
                    headers={"X-API-Key": raw_key})  # blocked, free plan
        after = store.usage_this_month(customer.customer_id)
        assert after == before  # no usage recorded for a blocked request

    def test_quota_exceeded_returns_429(self, saas_client, free_customer_key):
        from portfolio_optimizer.saas.plans import get_plan, PlanTier as PT
        client, get_store = saas_client
        raw_key, customer = free_customer_key
        store = get_store()
        plan = get_plan(PT.FREE)
        for _ in range(plan.requests_per_month):
            store.record_usage(customer.customer_id, "markowitz")

        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                            headers={"X-API-Key": raw_key})
        assert resp.status_code == 429
        assert "quota" in resp.json()["detail"].lower()

    def test_plan_downgrade_immediately_blocks_previously_allowed_endpoint(self, saas_client, pro_customer_key):
        client, get_store = saas_client
        raw_key, customer = pro_customer_key
        store = get_store()
        payload = _sample_payload()
        caps = {a: 1.0 for a in payload["assets"]}

        resp_before = client.post("/optimize/black-litterman",
                                    json={"payload": payload, "market_caps": caps},
                                    headers={"X-API-Key": raw_key})
        assert resp_before.status_code == 200

        store.update_customer_plan(customer.customer_id, PlanTier.FREE)

        resp_after = client.post("/optimize/black-litterman",
                                   json={"payload": payload, "market_caps": caps},
                                   headers={"X-API-Key": raw_key})
        assert resp_after.status_code == 403


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
