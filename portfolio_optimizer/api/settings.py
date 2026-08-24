"""
Production Settings -- environment-driven configuration for the API.

Every value here has a safe (restrictive, fail-closed) default, and every
value can be overridden by an environment variable of the same name.
This replaces scattered `os.environ.get(...)` calls throughout
`service.py` with one validated, typed, documented source of truth --
the difference between "works on my machine" and "safe to point a real
domain at."
"""
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -- Environment ------------------------------------------------------- #
    environment: str = "development"   # "development" | "staging" | "production"
    log_level: str = "INFO"
    log_format: str = "json"           # "json" | "text" -- json for production log aggregation

    # -- Server -------------------------------------------------------------- #
    host: str = "127.0.0.1"            # bind 0.0.0.0 only behind a reverse proxy (see nginx.conf)
    port: int = 8000
    workers: int = 4

    # -- CORS -------------------------------------------------------------- #
    # Comma-separated list of allowed origins, e.g.
    # "https://app.example.com,https://example.com". Empty by default
    # (fail-closed): a browser-based frontend gets NO cross-origin access
    # until this is explicitly configured, same fail-closed philosophy as
    # the API-key check.
    cors_allowed_origins: str = ""

    @property
    def cors_origins_list(self) -> list:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # -- Auth / SaaS --------------------------------------------------------- #
    portfolio_optimizer_api_keys: str = ""
    portfolio_optimizer_api_key_hashes: str = ""
    portfolio_optimizer_saas_mode: bool = False
    portfolio_optimizer_tenancy_db: str = "saas_tenancy.db"

    # -- Security headers ---------------------------------------------------- #
    enable_security_headers: bool = True
    hsts_max_age_seconds: int = 63072000   # 2 years, standard HSTS preload minimum

    # -- Rate limiting -------------------------------------------------------- #
    default_rate_limit: str = "60/minute"

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, v):
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got {v!r}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached singleton -- read once per process, not on every request."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache():
    """Test-only escape hatch: forces the next get_settings() call to
    re-read the environment, since the singleton above is deliberately
    cached for production performance but that caching is exactly what
    tests need to bypass to exercise different configurations."""
    global _settings
    _settings = None
