import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


class TestSettingsModule:
    def test_defaults_are_fail_closed(self):
        from portfolio_optimizer.api.settings import Settings
        s = Settings(_env_file=None)
        assert s.cors_origins_list == []
        assert s.environment == "development"

    def test_cors_origins_parsed_and_trimmed(self, monkeypatch):
        from portfolio_optimizer.api.settings import Settings
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", " https://a.com , https://b.com ")
        s = Settings(_env_file=None)
        assert s.cors_origins_list == ["https://a.com", "https://b.com"]

    def test_invalid_environment_rejected(self, monkeypatch):
        from portfolio_optimizer.api.settings import Settings
        monkeypatch.setenv("ENVIRONMENT", "not_a_real_environment")
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_is_production_property(self, monkeypatch):
        from portfolio_optimizer.api.settings import Settings
        monkeypatch.setenv("ENVIRONMENT", "production")
        s = Settings(_env_file=None)
        assert s.is_production is True

        monkeypatch.setenv("ENVIRONMENT", "development")
        s2 = Settings(_env_file=None)
        assert s2.is_production is False

    def test_get_settings_is_cached_singleton(self):
        from portfolio_optimizer.api.settings import get_settings, reset_settings_cache
        reset_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        reset_settings_cache()


class TestSecurityHeaders:
    def _client(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_API_KEYS", "test-key")
        from portfolio_optimizer.api.service import app
        return TestClient(app)

    def test_standard_security_headers_present(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "default-src" in resp.headers.get("content-security-policy", "")

    def test_hsts_only_sent_in_production(self, monkeypatch):
        """HSTS forces HTTPS for a domain going forward -- sending it in
        development (where the server is often plain HTTP on localhost)
        would be actively harmful, so it must be production-only.
        """
        client = self._client(monkeypatch)
        resp = client.get("/health")
        # default environment is "development" in this test process
        assert "strict-transport-security" not in {k.lower() for k in resp.headers.keys()}


class TestCORS:
    def test_disallowed_origin_gets_no_cors_headers(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_API_KEYS", "test-key")
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://allowed.example.com")
        from portfolio_optimizer.api.settings import reset_settings_cache
        reset_settings_cache()

        # Force a fresh app import so it picks up the env-configured origins
        for mod in list(sys.modules):
            if mod.startswith("portfolio_optimizer.api"):
                del sys.modules[mod]
        from portfolio_optimizer.api.service import app
        client = TestClient(app)

        resp = client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in {k.lower() for k in resp.headers.keys()}

    def test_allowed_origin_gets_cors_header(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_API_KEYS", "test-key")
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://allowed.example.com")
        from portfolio_optimizer.api.settings import reset_settings_cache
        reset_settings_cache()

        for mod in list(sys.modules):
            if mod.startswith("portfolio_optimizer.api"):
                del sys.modules[mod]
        from portfolio_optimizer.api.service import app
        client = TestClient(app)

        resp = client.get("/health", headers={"Origin": "https://allowed.example.com"})
        assert resp.headers.get("access-control-allow-origin") == "https://allowed.example.com"


class TestHealthEndpoints:
    def _client(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_API_KEYS", "test-key")
        from portfolio_optimizer.api.service import app
        return TestClient(app)

    def test_liveness_check_is_cheap_and_always_ok(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness_check_reports_dependency_status(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True
        assert body["checks"]["audit_log"] == "ok"

    def test_readiness_checks_tenancy_db_when_saas_mode_on(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_SAAS_MODE", "1")
        monkeypatch.setenv("PORTFOLIO_OPTIMIZER_TENANCY_DB", str(tmp_path / "test_tenancy.db"))
        for mod in list(sys.modules):
            if mod.startswith("portfolio_optimizer.api"):
                del sys.modules[mod]
        from portfolio_optimizer.api.service import app
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["checks"]["tenancy_db"] == "ok"

    def test_readiness_skips_tenancy_check_when_saas_mode_off(self, monkeypatch):
        monkeypatch.delenv("PORTFOLIO_OPTIMIZER_SAAS_MODE", raising=False)
        client = self._client(monkeypatch)
        resp = client.get("/health/ready")
        assert "skipped" in resp.json()["checks"]["tenancy_db"]


class TestStructuredLogging:
    def test_json_formatter_produces_valid_json(self):
        import logging, json
        from portfolio_optimizer.api.service import _JsonLogFormatter

        formatter = _JsonLogFormatter()
        record = logging.LogRecord(
            name="test.logger", level=logging.INFO, pathname="x.py", lineno=1,
            msg="a test message", args=(), exc_info=None,
        )
        formatted = formatter.format(record)
        parsed = json.loads(formatted)  # raises if not valid JSON
        assert parsed["message"] == "a test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"

    def test_json_formatter_includes_exception_info(self):
        import logging
        from portfolio_optimizer.api.service import _JsonLogFormatter
        import json, sys as _sys

        formatter = _JsonLogFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger", level=logging.ERROR, pathname="x.py", lineno=1,
                msg="something failed", args=(), exc_info=_sys.exc_info(),
            )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
