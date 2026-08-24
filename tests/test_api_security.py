import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# WARNING FOR FUTURE EDITS: pytest imports every test module during its
# collection phase, before ANY test in ANY file runs -- so this line
# executes (and this env var stays set) for the whole test session, not
# just while this file's tests run. This exact pattern once caused a real,
# hard-to-diagnose bug: another test file also set this same env var at
# module level with a DIFFERENT value, and whichever file happened to be
# imported last during collection silently won, breaking this file's
# tests with no obvious connection to the actual cause. If you add a new
# test file that also needs PORTFOLIO_OPTIMIZER_API_KEYS, use monkeypatch
# scoped to a fixture/test function instead of a bare module-level line.
os.environ["PORTFOLIO_OPTIMIZER_API_KEYS"] = "test-key-abc,test-key-def"

import numpy as np
import pytest
from fastapi.testclient import TestClient

from portfolio_optimizer.api.service import app, MAX_ASSETS, MAX_PERIODS, _rate_limit_identity

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key-abc"}


def _sample_payload(n_assets=4, n_periods=100, seed=0):
    rng = np.random.default_rng(seed)
    assets = [f"A{i}" for i in range(n_assets)]
    dates = [f"2024-01-{(i % 28) + 1:02d}" for i in range(n_periods)]
    returns = (rng.standard_normal((n_periods, n_assets)) * 0.01).tolist()
    return {"dates": dates, "assets": assets, "returns": returns}


class TestAuthentication:
    def test_missing_api_key_rejected(self):
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()})
        assert resp.status_code == 401

    def test_invalid_api_key_rejected(self):
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                            headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_valid_api_key_accepted(self):
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()}, headers=HEADERS)
        assert resp.status_code == 200

    def test_health_endpoint_does_not_require_auth(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_audit_endpoint_requires_auth(self):
        resp = client.get("/audit/verify")
        assert resp.status_code == 401


class TestHardenedAuthentication:
    def test_hashed_api_key_accepted_with_constant_time_path(self, monkeypatch):
        import hashlib
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_API_KEY_HASHES", hashlib.sha256(b"hashed-key").hexdigest())
        monkeypatch.delenv("PORTFOLIO_OPTIMIZER_API_KEYS", raising=False)
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                           headers={"X-API-Key": "hashed-key"})
        assert resp.status_code == 200

    def test_plaintext_key_rejected_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_API_KEYS", "plain-key")
        monkeypatch.delenv("PORTFOLIO_OPTIMIZER_API_KEY_HASHES", raising=False)
        from portfolio_optimizer.api.settings import reset_settings_cache
        reset_settings_cache()
        resp = client.post("/optimize/markowitz", json={"payload": _sample_payload()},
                           headers={"X-API-Key": "plain-key"})
        assert resp.status_code == 503
        monkeypatch.setenv("ENVIRONMENT", "development")
        reset_settings_cache()

    def test_rate_limit_identity_is_not_raw_api_key(self):
        from starlette.requests import Request
        scope = {"type": "http", "method": "GET", "path": "/",
                 "headers": [(b"x-api-key", b"secret-key")], "client": ("127.0.0.1", 1234),
                 "query_string": b"", "scheme": "http", "server": ("test", 80)}
        request = Request(scope)
        identity = _rate_limit_identity(request)
        assert "secret-key" not in identity
        assert identity.startswith("api-key:")


class TestInputValidation:
    def test_too_many_assets_rejected(self):
        payload = _sample_payload(n_assets=MAX_ASSETS + 1, n_periods=5)
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 422  # pydantic validation error

    def test_mismatched_returns_and_assets_rejected(self):
        payload = _sample_payload(n_assets=4, n_periods=10)
        payload["returns"][0] = payload["returns"][0][:2]  # corrupt one row
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 400

    def test_mismatched_dates_and_returns_length_rejected(self):
        payload = _sample_payload(n_assets=4, n_periods=10)
        payload["dates"] = payload["dates"][:5]  # fewer dates than return rows
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 400

    def test_nan_values_rejected(self):
        import json as _json
        payload = _sample_payload(n_assets=3, n_periods=10)
        payload["returns"][0][0] = float("nan")
        # httpx's default JSON encoder refuses to serialize NaN client-side
        # (which is itself a reasonable strictness), so to actually
        # exercise the SERVER's NaN validation we build the raw JSON body
        # ourselves (Python's stdlib json module permits NaN by default)
        # and send it with an explicit content-type.
        raw_body = _json.dumps({"payload": payload})
        resp = client.post("/optimize/markowitz", content=raw_body,
                            headers={**HEADERS, "Content-Type": "application/json"})
        assert resp.status_code in (400, 422)

    def test_empty_assets_rejected(self):
        payload = {"dates": ["2024-01-01"], "assets": [], "returns": [[]]}
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 422

    def test_duplicate_asset_names_rejected(self):
        payload = _sample_payload(n_assets=3, n_periods=10)
        payload["assets"] = ["AAPL", "AAPL", "MSFT"]
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 422

    def test_unparseable_dates_rejected(self):
        payload = _sample_payload(n_assets=3, n_periods=5)
        payload["dates"] = ["not-a-date"] * 5
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 400

    def test_bl_missing_market_caps_rejected(self):
        payload = _sample_payload(n_assets=4, n_periods=50)
        req = {"payload": payload, "market_caps": {"A0": 100.0}}  # missing A1, A2, A3
        resp = client.post("/optimize/black-litterman", json=req, headers=HEADERS)
        assert resp.status_code == 400

    def test_bl_view_on_unknown_asset_rejected(self):
        payload = _sample_payload(n_assets=3, n_periods=50)
        market_caps = {a: 100.0 for a in payload["assets"]}
        req = {"payload": payload, "market_caps": market_caps,
               "views": [{"assets": ["NOT_REAL"], "weights": [1.0], "value": 0.1}]}
        resp = client.post("/optimize/black-litterman", json=req, headers=HEADERS)
        assert resp.status_code == 400

    def test_invalid_objective_rejected(self):
        payload = _sample_payload(n_assets=3, n_periods=50)
        req = {"payload": payload, "objective": "not_a_real_objective"}
        resp = client.post("/optimize/markowitz", json=req, headers=HEADERS)
        assert resp.status_code == 400

    def test_cvar_invalid_alpha_rejected(self):
        payload = _sample_payload(n_assets=3, n_periods=50)
        req = {"payload": payload, "alpha": 1.5}
        resp = client.post("/optimize/cvar", json=req, headers=HEADERS)
        assert resp.status_code == 400

    def test_risk_parity_invalid_method_rejected(self):
        payload = _sample_payload(n_assets=3, n_periods=50)
        req = {"payload": payload, "method": "not_real"}
        resp = client.post("/optimize/risk-parity", json=req, headers=HEADERS)
        assert resp.status_code == 400


class TestRiskControlsWiredIntoResponse:
    def test_response_includes_risk_checks(self):
        payload = _sample_payload(n_assets=4, n_periods=100)
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert "risk_checks" in body
        assert "risk_checks_passed" in body
        assert isinstance(body["risk_checks"], list)
        assert len(body["risk_checks"]) > 0

    def test_tight_weight_bounds_still_passes_concentration_check(self):
        payload = _sample_payload(n_assets=5, n_periods=100)
        req = {"payload": payload, "weight_bounds": [0.0, 0.3]}
        resp = client.post("/optimize/markowitz", json=req, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        concentration_check = next(c for c in body["risk_checks"] if c["check"] == "max_position_weight")
        assert concentration_check["passed"] is True

    def test_hrp_response_also_includes_risk_checks(self):
        payload = _sample_payload(n_assets=4, n_periods=100)
        req = {"payload": payload, "method": "hrp"}
        resp = client.post("/optimize/risk-parity", json=req, headers=HEADERS)
        assert resp.status_code == 200
        assert "risk_checks" in resp.json()

    def test_mandate_violating_portfolio_returns_200_with_explicit_false_flag(self):
        """Documents the deliberate design contract: a mathematically valid
        but mandate-violating result is NOT silently blocked or turned
        into an error -- it's served with risk_checks_passed=False and the
        specific breach listed, so the caller can see exactly what the
        optimizer wanted and why it's flagged. Integrators MUST check this
        flag before acting on the weights; this test exists so that
        contract can never silently regress into "just returns 200 and
        looks fine" without anyone noticing.
        """
        rng = np.random.default_rng(1)
        n_periods = 200
        dates = [f"2024-{(i // 28) % 12 + 1:02d}-{(i % 28) + 1:02d}" for i in range(n_periods)]
        returns = rng.standard_normal((n_periods, 3)) * 0.005
        returns[:, 0] += 0.004  # dominant asset -> optimizer wants to concentrate heavily
        payload = {"dates": dates, "assets": ["STRONG", "B", "C"], "returns": returns.tolist()}

        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 200  # NOT blocked -- served with a flag, per the documented contract
        body = resp.json()
        assert body["risk_checks_passed"] is False
        breaches = [c for c in body["risk_checks"] if not c["passed"]]
        assert any(c["check"] == "max_position_weight" for c in breaches)
        assert body["weights"]["STRONG"] > 0.4  # the actual (flagged) optimizer output is still returned


class TestAuditLogWiring:
    def test_audit_chain_valid_after_requests(self):
        payload = _sample_payload(n_assets=3, n_periods=50)
        client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        resp = client.get("/audit/verify", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["chain_valid"] is True
        assert body["total_entries"] >= 2  # at least one request + one response entry

    def test_audit_entries_accumulate_across_requests(self):
        resp_before = client.get("/audit/verify", headers=HEADERS)
        n_before = resp_before.json()["total_entries"]
        payload = _sample_payload(n_assets=3, n_periods=50)
        client.post("/optimize/cvar", json={"payload": payload}, headers=HEADERS)
        resp_after = client.get("/audit/verify", headers=HEADERS)
        n_after = resp_after.json()["total_entries"]
        assert n_after > n_before


class TestErrorHandlingReturnsCleanResponses:
    def test_solver_failure_does_not_return_raw_stack_trace(self):
        # Nearly-zero-variance data across the board can make CVaR
        # optimization degenerate; verify we get a clean error, not a 500
        # with a leaked stack trace.
        payload = {"dates": [f"2024-01-{i+1:02d}" for i in range(10)],
                   "assets": ["A", "B"], "returns": [[0.0, 0.0]] * 10}
        resp = client.post("/optimize/cvar", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code in (200, 422)
        if resp.status_code != 200:
            assert "traceback" not in resp.text.lower()

    def test_all_optimize_endpoints_reachable_with_valid_data(self):
        payload = _sample_payload(n_assets=4, n_periods=100)
        market_caps = {a: 100.0 for a in payload["assets"]}

        r1 = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        r2 = client.post("/optimize/black-litterman",
                          json={"payload": payload, "market_caps": market_caps}, headers=HEADERS)
        r3 = client.post("/optimize/risk-parity", json={"payload": payload}, headers=HEADERS)
        r4 = client.post("/optimize/cvar", json={"payload": payload}, headers=HEADERS)

        for r in (r1, r2, r3, r4):
            assert r.status_code == 200
            assert "weights" in r.json()


class TestFailClosedWhenUnconfigured:
    def test_no_configured_keys_refuses_everything(self, monkeypatch):
        monkeypatch.delenv("PORTFOLIO_OPTIMIZER_API_KEYS", raising=False)
        payload = _sample_payload(n_assets=3, n_periods=50)
        resp = client.post("/optimize/markowitz", json={"payload": payload}, headers=HEADERS)
        assert resp.status_code == 503


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
