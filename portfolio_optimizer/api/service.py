"""
Production REST API -- FastAPI service wrapping every optimizer.

Run:
    export PORTFOLIO_OPTIMIZER_API_KEYS="key1,key2"   # comma-separated valid keys
    uvicorn portfolio_optimizer.api.service:app --host 0.0.0.0 --port 8000

Every /optimize/* endpoint:
    1. Requires a valid API key (X-API-Key header)
    2. Is rate-limited per API key
    3. Validates payload size before touching the solver
    4. Runs the solved weights through PreTradeRiskEngine checks and
       reports violations in the response (does not silently hide them)
    5. Logs the request and outcome to a hash-chained audit log
    6. Returns clean 4xx errors for bad input / solver failure, not raw 500s

DESIGN CHOICE ON RISK CHECKS (read before integrating): a mathematically
valid optimization result that breaches a pre-trade limit (e.g. 93%
concentrated in one asset) is still returned with HTTP 200 -- it is NOT
silently blocked or downgraded to an error. Instead `risk_checks_passed`
is set to `false` and every failing check is listed with its severity and
message. This is deliberate: the caller (a human reviewing a proposed
rebalance, or an automated system with its own veto logic) gets to see
what the optimizer actually wanted and exactly why it's flagged, rather
than the API silently overriding the math or hiding the breach behind a
generic error. INTEGRATORS MUST CHECK `risk_checks_passed` BEFORE ACTING
ON `weights` -- treating a 200 response as "safe to trade" without
checking that field defeats the entire point of this layer. See
tests/test_api_security.py::TestRiskControlsWiredIntoResponse for the
explicit contract test.
"""
from __future__ import annotations
import os
import logging
import hashlib
import hmac
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .settings import get_settings
from ..estimators.expected_returns import mean_historical_return
from ..estimators.covariance import ledoit_wolf_shrinkage
from ..optimizers.markowitz import MarkowitzOptimizer
from ..optimizers.black_litterman import BlackLitterman
from ..optimizers.risk_parity import RiskParity, HierarchicalRiskParity
from ..optimizers.cvar import CVaROptimizer
from ..infra.risk_controls import PreTradeRiskEngine, PreTradeLimits, CheckSeverity
from ..infra.audit_log import TamperEvidentAuditLog

logger = logging.getLogger("portfolio_optimizer.api")


class _JsonLogFormatter(logging.Formatter):
    """Structured JSON logs -- what a real log aggregator (CloudWatch,
    Datadog, an ELK stack) expects, as opposed to plain text lines that
    have to be regex-parsed back apart. Falls back to plain text when
    LOG_FORMAT=text (nicer for a human staring at a terminal in dev).
    """
    def format(self, record: logging.LogRecord) -> str:
        import json as _json
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname, "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return _json.dumps(payload)


def _configure_logging():
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    handler = logging.StreamHandler()
    if settings.log_format == "json":
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers = [handler]


_configure_logging()

# --------------------------------------------------------------------- #
# Auth: simple API-key check against an env-configured allowlist.
# SCOPE NOTE: this is a minimal, real API-key gate -- adequate for an
# internal/service-to-service deployment behind your own network
# boundary. For a client-facing product you'd want per-key issuance,
# rotation, and expiry (a JWT-based scheme, or a proper auth provider
# like Auth0/Cognito) rather than a static env-var allowlist.
# --------------------------------------------------------------------- #
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_api_key_hashes() -> set[str]:
    raw = os.environ.get("PORTFOLIO_OPTIMIZER_API_KEY_HASHES", "")
    return {h.strip().lower().removeprefix("sha256:") for h in raw.split(",") if h.strip()}


def _legacy_plain_api_keys() -> set[str]:
    raw = os.environ.get("PORTFOLIO_OPTIMIZER_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _api_key_is_valid(api_key: str) -> bool:
    presented_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    for configured_hash in _configured_api_key_hashes():
        if hmac.compare_digest(presented_hash, configured_hash):
            return True

    # Backward-compatible plaintext configuration is retained for local/dev
    # deployments. Production deployments must use the hashed setting below.
    if get_settings().is_production:
        return False
    for configured_key in _legacy_plain_api_keys():
        if hmac.compare_digest(api_key, configured_key):
            return True
    return False


def require_api_key(api_key: str = Depends(_api_key_header)) -> str:
    configured_hashes = _configured_api_key_hashes()
    legacy_keys = _legacy_plain_api_keys()
    if not configured_hashes and (get_settings().is_production or not legacy_keys):
        raise HTTPException(
            status_code=503,
            detail=(
                "API authentication is not configured safely. Set "
                "PORTFOLIO_OPTIMIZER_API_KEY_HASHES to SHA-256 API-key hashes."
            ),
        )
    if not api_key or not _api_key_is_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return api_key


def _rate_limit_identity(request: Request) -> str:
    """Use the authenticated API key as the primary rate-limit identity.

    The raw secret is never used as the limiter key; only a SHA-256 digest is
    exposed to slowapi. Requests without a key fall back to client IP so an
    unauthenticated attacker cannot share a single global bucket.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return f"api-key:{digest}"
    return f"ip:{get_remote_address(request)}"


# --------------------------------------------------------------------- #
# SaaS mode: real per-customer plan gating + usage metering via
# TenancyStore, instead of the flat env-var allowlist above. Controlled
# by PORTFOLIO_OPTIMIZER_SAAS_MODE=1 -- when unset, the API falls back to
# the simpler static-key mode (useful for internal/self-hosted
# deployments that don't need billing at all). This keeps the original
# simple mode fully working rather than forcing SaaS plumbing on every
# deployment.
# --------------------------------------------------------------------- #
def _saas_mode_enabled() -> bool:
    return os.environ.get("PORTFOLIO_OPTIMIZER_SAAS_MODE", "0") == "1"


_tenancy_store = None


def _get_tenancy_store():
    global _tenancy_store
    if _tenancy_store is None:
        from ..saas.tenancy import TenancyStore
        db_path = os.environ.get("PORTFOLIO_OPTIMIZER_TENANCY_DB", "saas_tenancy.db")
        _tenancy_store = TenancyStore(db_path)
    return _tenancy_store


def require_tenant_for_endpoint(endpoint_name: str):
    """Returns a FastAPI dependency that validates the API key against
    TenancyStore, checks the customer's plan actually allows this
    endpoint, checks their monthly quota isn't exhausted, and records the
    usage event -- the real, monetization-relevant version of
    `require_api_key` above. Only active when SaaS mode is enabled.
    """
    def _dependency(api_key: str = Depends(_api_key_header)):
        if not _saas_mode_enabled():
            # SaaS mode off -- fall back to the simple static-key check,
            # with no plan gating or usage metering (self-hosted mode).
            return require_api_key(api_key)

        if not api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

        store = _get_tenancy_store()
        result = store.validate_api_key(api_key, endpoint=endpoint_name)
        if not result.valid:
            status = 402 if result.plan and result.customer and \
                result.customer.subscription_status != "active" else \
                (403 if "not available" in (result.reason or "") else 401)
            raise HTTPException(status_code=status, detail=result.reason)

        within_quota, used = store.check_monthly_quota(result.customer.customer_id, result.plan)
        if not within_quota:
            raise HTTPException(
                status_code=429,
                detail=f"Monthly quota exceeded ({used}/{result.plan.requests_per_month} requests "
                       f"on the {result.plan.display_name} plan). Upgrade or wait for next billing cycle.",
            )

        store.record_usage(result.customer.customer_id, endpoint_name)
        return api_key

    return _dependency


# --------------------------------------------------------------------- #
# Rate limiting: per-API-key token bucket via slowapi.
# --------------------------------------------------------------------- #
limiter = Limiter(key_func=_rate_limit_identity, default_limits=["60/minute"])

app = FastAPI(
    title="Portfolio Optimizer API",
    description="Institutional-grade portfolio construction engine -- Markowitz, "
                "Black-Litterman, Risk Parity, HRP, and CVaR optimization.",
    version="3.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_prod_settings = get_settings()

# --------------------------------------------------------------------- #
# CORS: fail-closed by default (CORS_ALLOWED_ORIGINS unset -> empty list
# -> CORSMiddleware allows NO cross-origin browser requests), same
# fail-closed philosophy as the API-key check above. A real website's
# frontend JS literally cannot call this API from a browser until its
# origin is explicitly listed here -- this is not optional plumbing, it's
# the actual mechanism that lets or blocks pricing.html (or any other
# frontend) from talking to a deployed instance of this API.
# --------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=_prod_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "stripe-signature"],
)


# --------------------------------------------------------------------- #
# Security headers -- applied to every response when enabled (on by
# default). These are the standard baseline a real internet-facing API
# should send regardless of framework: stop MIME-sniffing, stop being
# framed (clickjacking), force HTTPS going forward once first seen over
# HTTPS, and a conservative default Content-Security-Policy for any HTML
# this process might ever serve directly (it doesn't today, but a
# misconfigured deployment shouldn't rely on that staying true).
# --------------------------------------------------------------------- #
@app.middleware("http")
async def _security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    if _prod_settings.enable_security_headers:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if _prod_settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={_prod_settings.hsts_max_age_seconds}; includeSubDomains"
            )
    return response

# Module-level singletons: a process-lifetime audit log and default
# pre-trade limits. In a real multi-process deployment the audit log
# should be backed by shared/durable storage (e.g. appended to the
# PortfolioStateStore's SQLite DB, or streamed to an external log
# aggregator) rather than living in one process's memory -- flagged here
# rather than silently pretending this in-memory version is durable.
_audit_log = TamperEvidentAuditLog()
_default_limits = PreTradeLimits(
    max_position_weight=0.40, max_gross_leverage=1.0, max_total_turnover=2.0,
)
_risk_engine = PreTradeRiskEngine(_default_limits)

MAX_ASSETS = 500
MAX_PERIODS = 10_000


class ReturnsPayload(BaseModel):
    """Wire format: {"dates": [...], "assets": [...], "returns": [[...], ...]}"""
    dates: list[str]
    assets: list[str]
    returns: list[list[float]]

    @field_validator("assets")
    @classmethod
    def _check_asset_count(cls, v):
        if len(v) == 0:
            raise ValueError("assets list cannot be empty")
        if len(v) > MAX_ASSETS:
            raise ValueError(f"Too many assets: {len(v)} > limit of {MAX_ASSETS}")
        if len(set(v)) != len(v):
            raise ValueError("assets list contains duplicate names")
        return v

    @field_validator("dates")
    @classmethod
    def _check_period_count(cls, v):
        if len(v) == 0:
            raise ValueError("dates list cannot be empty")
        if len(v) > MAX_PERIODS:
            raise ValueError(f"Too many periods: {len(v)} > limit of {MAX_PERIODS}")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        if len(self.returns) != len(self.dates):
            raise ValueError(
                f"returns has {len(self.returns)} rows but dates has {len(self.dates)} entries"
            )
        for i, row in enumerate(self.returns):
            if len(row) != len(self.assets):
                raise ValueError(
                    f"returns row {i} has {len(row)} values but assets has {len(self.assets)} entries"
                )
        try:
            idx = pd.to_datetime(self.dates)
        except Exception as e:
            raise ValueError(f"Could not parse dates: {e}")
        df = pd.DataFrame(self.returns, index=idx, columns=self.assets)
        if df.isna().any().any():
            bad_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(f"returns contains NaN values in columns: {bad_cols}")
        if not np.isfinite(df.values).all():
            raise ValueError("returns contains non-finite values (inf/-inf)")
        return df


class MarkowitzRequest(BaseModel):
    payload: ReturnsPayload
    objective: str = Field("max_sharpe", description="max_sharpe | min_volatility | target_return")
    target_return: Optional[float] = None
    risk_free_rate: float = 0.0
    weight_bounds: tuple[float, float] = (0.0, 1.0)
    use_shrinkage: bool = True


class ViewPayload(BaseModel):
    assets: list[str]
    weights: list[float]
    value: float
    confidence: float = 0.5


class BLRequest(BaseModel):
    payload: ReturnsPayload
    market_caps: dict[str, float]
    views: list[ViewPayload] = []
    risk_aversion: float = 2.5
    tau: float = 0.05


class RiskParityRequest(BaseModel):
    payload: ReturnsPayload
    method: str = Field("erc", description="erc | hrp")
    risk_budget: Optional[dict[str, float]] = None


class CVaRRequest(BaseModel):
    payload: ReturnsPayload
    alpha: float = 0.95
    target_return: Optional[float] = None


class RiskCheckSummary(BaseModel):
    check: str
    passed: bool
    severity: str
    message: str


class WeightsResponse(BaseModel):
    weights: dict[str, float]
    metrics: dict
    risk_checks: list[RiskCheckSummary]
    risk_checks_passed: bool


def _run_risk_checks(weights: pd.Series) -> tuple[list, bool]:
    results = _risk_engine.check_target_portfolio(weights)
    summaries = [RiskCheckSummary(check=r.check_name, passed=r.passed,
                                    severity=r.severity.value, message=r.message)
                 for r in results]
    approved = _risk_engine.is_approved(results)
    return summaries, approved


def _audit(event_type: str, payload: dict):
    try:
        _audit_log.record(event_type, payload)
    except Exception:
        logger.exception("Failed to write audit log entry (continuing -- audit failure "
                          "should not block a response, but IS logged for follow-up).")


@app.get("/health")
def health():
    """Liveness check: is the process up at all. Deliberately cheap (no
    I/O) -- a load balancer or orchestrator polls this frequently, and a
    slow liveness check is itself a production problem.
    """
    return {"status": "ok", "version": "3.1.0"}


@app.get("/health/ready")
def readiness():
    """Readiness check: can this instance actually serve real requests
    right now. Unlike /health, this DOES touch dependencies -- checks the
    audit log is writable and, if SaaS mode is on, that the tenancy DB is
    reachable. A load balancer should stop routing traffic to an instance
    that fails this, even if the bare process is technically still alive.
    """
    checks = {}
    overall_ok = True

    try:
        test_entry = _audit_log.record("readiness_probe", {})
        checks["audit_log"] = "ok"
    except Exception as e:
        checks["audit_log"] = f"failed: {e}"
        overall_ok = False

    if _saas_mode_enabled():
        try:
            store = _get_tenancy_store()
            store.usage_this_month("__readiness_probe__")
            checks["tenancy_db"] = "ok"
        except Exception as e:
            checks["tenancy_db"] = f"failed: {e}"
            overall_ok = False
    else:
        checks["tenancy_db"] = "skipped (SaaS mode off)"

    status_code = 200 if overall_ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status_code,
                         content={"ready": overall_ok, "checks": checks})


@app.get("/audit/verify")
def verify_audit_chain(api_key: str = Depends(require_api_key)):
    """Verify the in-process audit log hasn't been tampered with, and
    report how many entries have been recorded so far."""
    valid, broken_index = _audit_log.verify_chain()
    return {"chain_valid": valid, "broken_at_index": broken_index,
            "total_entries": len(_audit_log.entries)}


@app.post("/optimize/markowitz", response_model=WeightsResponse)
@limiter.limit("30/minute")
def optimize_markowitz(request: Request, req: MarkowitzRequest,
                        api_key: str = Depends(require_tenant_for_endpoint("markowitz"))):
    _audit("optimize_request", {"endpoint": "markowitz", "api_key_prefix": api_key[:4],
                                 "n_assets": len(req.payload.assets), "objective": req.objective})
    try:
        returns = req.payload.to_dataframe()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid returns payload: {e}")

    try:
        mu = mean_historical_return(returns)
        if req.use_shrinkage:
            cov, shrink = ledoit_wolf_shrinkage(returns)
        else:
            cov = returns.cov() * 252
            shrink = 0.0
        opt = MarkowitzOptimizer(mu, cov, risk_free_rate=req.risk_free_rate,
                                  weight_bounds=req.weight_bounds)
        if req.objective == "max_sharpe":
            res = opt.max_sharpe()
        elif req.objective == "min_volatility":
            res = opt.min_volatility()
        elif req.objective == "target_return":
            if req.target_return is None:
                raise HTTPException(status_code=400,
                                     detail="target_return is required for objective=target_return")
            res = opt.target_return(req.target_return)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown objective: {req.objective}")
    except HTTPException:
        raise
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=422, detail=f"Covariance matrix is not invertible: {e}")
    except Exception as e:
        logger.exception("Unexpected error in markowitz optimization")
        raise HTTPException(status_code=500, detail="Internal optimization error; see server logs.")

    if not res.success:
        raise HTTPException(status_code=422, detail=f"Optimizer did not converge: {res.message}")

    risk_checks, approved = _run_risk_checks(res.weights)
    _audit("optimize_response", {"endpoint": "markowitz", "weights": res.weights.round(6).to_dict(),
                                  "risk_checks_passed": approved})

    return WeightsResponse(
        weights=res.weights.round(6).to_dict(),
        metrics={"expected_return": res.expected_return, "volatility": res.volatility,
                 "sharpe_ratio": res.sharpe_ratio, "shrinkage_intensity": shrink},
        risk_checks=risk_checks, risk_checks_passed=approved,
    )


@app.post("/optimize/black-litterman", response_model=WeightsResponse)
@limiter.limit("30/minute")
def optimize_black_litterman(request: Request, req: BLRequest,
                              api_key: str = Depends(require_tenant_for_endpoint("black-litterman"))):
    _audit("optimize_request", {"endpoint": "black_litterman", "api_key_prefix": api_key[:4],
                                 "n_assets": len(req.payload.assets), "n_views": len(req.views)})
    try:
        returns = req.payload.to_dataframe()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid returns payload: {e}")

    missing_caps = set(req.payload.assets) - set(req.market_caps)
    if missing_caps:
        raise HTTPException(status_code=400,
                             detail=f"market_caps missing entries for: {sorted(missing_caps)}")

    try:
        cov, _ = ledoit_wolf_shrinkage(returns)
        caps = pd.Series(req.market_caps)
        bl = BlackLitterman(cov, market_caps=caps, risk_aversion=req.risk_aversion, tau=req.tau)
        from ..optimizers.black_litterman import View
        for v in req.views:
            unknown = set(v.assets) - set(req.payload.assets)
            if unknown:
                raise HTTPException(status_code=400,
                                     detail=f"View references unknown assets: {sorted(unknown)}")
            bl.add_view(View(assets=v.assets, weights=v.weights, value=v.value, confidence=v.confidence))
        post_mu, post_cov = bl.posterior()
        opt = MarkowitzOptimizer(post_mu, post_cov, risk_free_rate=0.0)
        res = opt.max_sharpe()
    except HTTPException:
        raise
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=422, detail=f"Covariance matrix is not invertible: {e}")
    except Exception as e:
        logger.exception("Unexpected error in black-litterman optimization")
        raise HTTPException(status_code=500, detail="Internal optimization error; see server logs.")

    if not res.success:
        raise HTTPException(status_code=422, detail=f"Optimizer did not converge: {res.message}")

    risk_checks, approved = _run_risk_checks(res.weights)
    _audit("optimize_response", {"endpoint": "black_litterman", "weights": res.weights.round(6).to_dict(),
                                  "risk_checks_passed": approved})

    return WeightsResponse(
        weights=res.weights.round(6).to_dict(),
        metrics={"expected_return": res.expected_return, "volatility": res.volatility,
                 "sharpe_ratio": res.sharpe_ratio},
        risk_checks=risk_checks, risk_checks_passed=approved,
    )


@app.post("/optimize/risk-parity", response_model=WeightsResponse)
@limiter.limit("30/minute")
def optimize_risk_parity(request: Request, req: RiskParityRequest,
                          api_key: str = Depends(require_tenant_for_endpoint("risk-parity"))):
    _audit("optimize_request", {"endpoint": "risk_parity", "api_key_prefix": api_key[:4],
                                 "n_assets": len(req.payload.assets), "method": req.method})
    try:
        returns = req.payload.to_dataframe()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid returns payload: {e}")

    try:
        if req.method == "hrp":
            w = HierarchicalRiskParity(returns).solve()
            metrics = {"method": "HRP"}
        elif req.method == "erc":
            cov, _ = ledoit_wolf_shrinkage(returns)
            budget = pd.Series(req.risk_budget) if req.risk_budget else None
            rp = RiskParity(cov, risk_budget=budget)
            w = rp.solve()
            report = rp.risk_contribution_report(w)
            metrics = {"method": "ERC",
                       "risk_contribution_pct": report["risk_contribution_pct"].round(4).to_dict()}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {req.method} (use 'erc' or 'hrp')")
    except HTTPException:
        raise
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=422, detail=f"Covariance matrix is not invertible: {e}")
    except Exception as e:
        logger.exception("Unexpected error in risk parity optimization")
        raise HTTPException(status_code=500, detail="Internal optimization error; see server logs.")

    risk_checks, approved = _run_risk_checks(w)
    _audit("optimize_response", {"endpoint": "risk_parity", "weights": w.round(6).to_dict(),
                                  "risk_checks_passed": approved})

    return WeightsResponse(weights=w.round(6).to_dict(), metrics=metrics,
                            risk_checks=risk_checks, risk_checks_passed=approved)


@app.post("/optimize/cvar", response_model=WeightsResponse)
@limiter.limit("30/minute")
def optimize_cvar(request: Request, req: CVaRRequest, api_key: str = Depends(require_tenant_for_endpoint("cvar"))):
    _audit("optimize_request", {"endpoint": "cvar", "api_key_prefix": api_key[:4],
                                 "n_assets": len(req.payload.assets), "alpha": req.alpha})
    try:
        returns = req.payload.to_dataframe()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid returns payload: {e}")

    if not (0.5 <= req.alpha < 1.0):
        raise HTTPException(status_code=400, detail="alpha must be in [0.5, 1.0)")

    try:
        cvar_opt = CVaROptimizer(returns, alpha=req.alpha)
        res = cvar_opt.optimize(target_return=req.target_return)
    except Exception as e:
        logger.exception("Unexpected error in CVaR optimization")
        raise HTTPException(status_code=500, detail="Internal optimization error; see server logs.")

    if not res.success:
        raise HTTPException(status_code=422, detail="CVaR optimization failed to converge "
                                                       "(check for infeasible target_return).")

    risk_checks, approved = _run_risk_checks(res.weights)
    _audit("optimize_response", {"endpoint": "cvar", "weights": res.weights.round(6).to_dict(),
                                  "risk_checks_passed": approved})

    return WeightsResponse(
        weights=res.weights.round(6).to_dict(),
        metrics={"expected_return": res.expected_return, "cvar": res.cvar, "var": res.var},
        risk_checks=risk_checks, risk_checks_passed=approved,
    )


# ======================================================================= #
# SaaS billing endpoints -- only meaningful when PORTFOLIO_OPTIMIZER_SAAS_MODE=1
# and Stripe environment variables are configured (see saas/billing.py's
# module docstring for exactly which ones and where to get them).
# ======================================================================= #

class CheckoutRequest(BaseModel):
    email: str
    plan: str = Field(..., description="pro | institutional")
    success_url: str
    cancel_url: str


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


@app.post("/billing/checkout", response_model=CheckoutResponse)
def create_checkout(req: CheckoutRequest):
    """Creates a real Stripe Checkout session and returns the URL to
    redirect the customer to -- Stripe hosts the actual payment page.
    """
    from ..saas.billing import create_checkout_session
    from ..saas.plans import plan_from_string

    try:
        plan = plan_from_string(req.plan)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if plan.monthly_price_usd == 0:
        raise HTTPException(status_code=400, detail="The Free plan doesn't need checkout -- "
                                                       "sign up directly to get a Free-tier API key.")
    try:
        result = create_checkout_session(req.email, plan.tier, req.success_url, req.cancel_url)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CheckoutResponse(checkout_url=result.checkout_url, session_id=result.session_id)


@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """Stripe calls this endpoint directly (not the customer's browser) to
    notify about subscription lifecycle events. Signature verification
    (via STRIPE_WEBHOOK_SECRET) is what stops anyone else from POSTing a
    forged event here to grant themselves a free subscription.
    """
    from ..saas.billing import verify_and_parse_webhook, handle_webhook_event

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = verify_and_parse_webhook(payload, signature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook verification failed: {e}")

    store = _get_tenancy_store()
    result = handle_webhook_event(event, store)
    _audit("billing_webhook", {"event_type": event.get("type"), "result_action": result.get("action")})

    # Never echo the one-time API key back in a webhook response (it's
    # already returned once, correctly, in `result` for the caller who
    # owns this webhook handler to email to the customer out-of-band --
    # logging/returning it here would put it in Stripe's dashboard logs).
    safe_result = {k: v for k, v in result.items() if k != "api_key"}
    return safe_result


@app.get("/billing/plans")
def list_plans():
    """Public endpoint -- what a pricing page would call to render plan
    cards without hardcoding limits in two places.
    """
    from ..saas.plans import PLANS
    return [
        {
            "tier": p.tier.value, "display_name": p.display_name,
            "monthly_price_usd": p.monthly_price_usd,
            "requests_per_month": p.requests_per_month,
            "requests_per_minute": p.requests_per_minute,
            "max_assets_per_request": p.max_assets_per_request,
            "allowed_endpoints": sorted(p.allowed_endpoints),
            "live_trading_allowed": p.live_trading_allowed,
        }
        for p in PLANS.values()
    ]
