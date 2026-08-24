"""
SaaS Pricing Plans.

Defines the tiers this could actually be sold under, each with concrete,
enforceable limits (not just a name) -- rate limit, max assets per
request, which optimizer endpoints are unlocked, and whether the live
trading loop is allowed. These limits are read by `tenancy.py` (to gate
requests) and `billing.py` (to map a Stripe subscription back to a plan).

HONEST FRAMING: these numbers and prices are a reasonable, defensible
starting point for a research-tool SaaS, not market-tested pricing. Real
pricing needs actual customer conversations -- this is the plumbing that
lets you change the numbers in one place once you have that data, not a
claim that these specific numbers are correct.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class PlanTier(Enum):
    FREE = "free"
    PRO = "pro"
    INSTITUTIONAL = "institutional"


@dataclass(frozen=True)
class PlanLimits:
    tier: PlanTier
    display_name: str
    monthly_price_usd: float
    requests_per_minute: int
    requests_per_month: int
    max_assets_per_request: int
    max_periods_per_request: int
    allowed_endpoints: frozenset       # which /optimize/* endpoints are unlocked
    live_trading_allowed: bool
    concurrent_live_portfolios: int
    stripe_price_id_env_var: str        # which env var holds the real Stripe Price ID for this plan


PLANS: dict = {
    PlanTier.FREE: PlanLimits(
        tier=PlanTier.FREE, display_name="Free", monthly_price_usd=0.0,
        requests_per_minute=10, requests_per_month=500,
        max_assets_per_request=10, max_periods_per_request=500,
        allowed_endpoints=frozenset({"markowitz", "risk-parity"}),
        live_trading_allowed=False, concurrent_live_portfolios=0,
        stripe_price_id_env_var="STRIPE_PRICE_ID_FREE",
    ),
    PlanTier.PRO: PlanLimits(
        tier=PlanTier.PRO, display_name="Pro", monthly_price_usd=99.0,
        requests_per_minute=60, requests_per_month=50_000,
        max_assets_per_request=100, max_periods_per_request=5_000,
        allowed_endpoints=frozenset({"markowitz", "risk-parity", "black-litterman", "cvar"}),
        live_trading_allowed=True, concurrent_live_portfolios=3,
        stripe_price_id_env_var="STRIPE_PRICE_ID_PRO",
    ),
    PlanTier.INSTITUTIONAL: PlanLimits(
        tier=PlanTier.INSTITUTIONAL, display_name="Institutional", monthly_price_usd=999.0,
        requests_per_minute=600, requests_per_month=2_000_000,
        max_assets_per_request=500, max_periods_per_request=10_000,
        allowed_endpoints=frozenset({"markowitz", "risk-parity", "black-litterman", "cvar"}),
        live_trading_allowed=True, concurrent_live_portfolios=50,
        stripe_price_id_env_var="STRIPE_PRICE_ID_INSTITUTIONAL",
    ),
}


def get_plan(tier: PlanTier) -> PlanLimits:
    return PLANS[tier]


def plan_from_string(tier_str: str) -> PlanLimits:
    try:
        return PLANS[PlanTier(tier_str.lower())]
    except (ValueError, KeyError):
        raise ValueError(f"Unknown plan tier: {tier_str!r}. Valid: {[t.value for t in PlanTier]}")
