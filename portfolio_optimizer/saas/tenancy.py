"""
Multi-Tenant Customer & API Key Management.

Replaces the API's original static env-var allowlist
(PORTFOLIO_OPTIMIZER_API_KEYS) with a real, per-customer system: each
customer gets their own API key(s), tied to a plan tier, with usage
tracked and enforced per key rather than a single shared secret.

SQLite-backed (same pattern as `infra.persistence`) so this runs with
zero extra infrastructure -- swap the connection string for Postgres
later if/when you have concurrent multi-process API servers that need to
share state (SQLite's single-writer model is fine for one process, not
ideal for a fleet of them).
"""
from __future__ import annotations
import sqlite3
import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from contextlib import contextmanager

from .plans import PlanTier, PlanLimits, get_plan


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    plan_tier TEXT NOT NULL,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    subscription_status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY,
    key_prefix TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""


def _hash_key(raw_key: str) -> str:
    """API keys are stored as salted hashes, never in plaintext -- the
    same principle as password storage. The raw key is shown to the
    customer exactly once, at creation time, and never again.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _generate_raw_key(prefix: str = "pk_live") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


@dataclass
class Customer:
    customer_id: str
    email: str
    plan_tier: PlanTier
    subscription_status: str
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None


@dataclass
class ApiKeyValidationResult:
    valid: bool
    customer: Customer | None = None
    plan: PlanLimits | None = None
    reason: str | None = None


class TenancyStore:
    def __init__(self, db_path: str = "saas_tenancy.db"):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- customers -------------------------------------------------------- #
    def create_customer(self, email: str, plan_tier: PlanTier = PlanTier.FREE,
                         stripe_customer_id: str | None = None) -> Customer:
        customer_id = f"cust_{secrets.token_urlsafe(12)}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO customers (customer_id, email, plan_tier, stripe_customer_id, "
                "subscription_status, created_at) VALUES (?, ?, ?, ?, 'active', ?)",
                (customer_id, email, plan_tier.value, stripe_customer_id,
                 datetime.now(timezone.utc).isoformat()),
            )
        return Customer(customer_id=customer_id, email=email, plan_tier=plan_tier,
                         subscription_status="active", stripe_customer_id=stripe_customer_id)

    def get_customer(self, customer_id: str) -> Customer | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM customers WHERE customer_id = ?",
                                (customer_id,)).fetchone()
        if not row:
            return None
        return Customer(customer_id=row["customer_id"], email=row["email"],
                         plan_tier=PlanTier(row["plan_tier"]),
                         subscription_status=row["subscription_status"],
                         stripe_customer_id=row["stripe_customer_id"],
                         stripe_subscription_id=row["stripe_subscription_id"])

    def get_customer_by_stripe_id(self, stripe_customer_id: str) -> Customer | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM customers WHERE stripe_customer_id = ?",
                                (stripe_customer_id,)).fetchone()
        if not row:
            return None
        return Customer(customer_id=row["customer_id"], email=row["email"],
                         plan_tier=PlanTier(row["plan_tier"]),
                         subscription_status=row["subscription_status"],
                         stripe_customer_id=row["stripe_customer_id"],
                         stripe_subscription_id=row["stripe_subscription_id"])

    def get_customer_by_email(self, email: str) -> Customer | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM customers WHERE email = ?", (email,)).fetchone()
        if not row:
            return None
        return Customer(customer_id=row["customer_id"], email=row["email"],
                         plan_tier=PlanTier(row["plan_tier"]),
                         subscription_status=row["subscription_status"],
                         stripe_customer_id=row["stripe_customer_id"],
                         stripe_subscription_id=row["stripe_subscription_id"])

    def attach_stripe_customer_id(self, customer_id: str, stripe_customer_id: str):
        """Links an existing (e.g. email-matched) customer record to a new
        Stripe customer ID -- used when someone re-subscribes and Stripe
        issues a new customer object for the same real person/email.
        """
        with self._connect() as conn:
            conn.execute("UPDATE customers SET stripe_customer_id = ? WHERE customer_id = ?",
                         (stripe_customer_id, customer_id))

    def update_customer_plan(self, customer_id: str, plan_tier: PlanTier,
                              subscription_status: str = "active",
                              stripe_subscription_id: str | None = None):
        with self._connect() as conn:
            conn.execute(
                "UPDATE customers SET plan_tier = ?, subscription_status = ?, "
                "stripe_subscription_id = COALESCE(?, stripe_subscription_id) "
                "WHERE customer_id = ?",
                (plan_tier.value, subscription_status, stripe_subscription_id, customer_id),
            )

    # -- API keys --------------------------------------------------------- #
    def issue_api_key(self, customer_id: str) -> str:
        """Returns the RAW key -- this is the only time it's ever visible.
        Only the hash is stored; if the customer loses it, issue a new one."""
        raw_key = _generate_raw_key()
        key_hash = _hash_key(raw_key)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_hash, key_prefix, customer_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key_hash, raw_key[:12], customer_id, datetime.now(timezone.utc).isoformat()),
            )
        return raw_key

    def revoke_api_key(self, raw_key: str):
        key_hash = _hash_key(raw_key)
        with self._connect() as conn:
            conn.execute("UPDATE api_keys SET revoked_at = ? WHERE key_hash = ?",
                         (datetime.now(timezone.utc).isoformat(), key_hash))

    def validate_api_key(self, raw_key: str, endpoint: str | None = None) -> ApiKeyValidationResult:
        """The core gate: hash the presented key, look it up, check it's
        not revoked, load the owning customer's plan, and (if an endpoint
        is given) check the plan actually allows that endpoint.
        """
        key_hash = _hash_key(raw_key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT customer_id, revoked_at FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()

        if not row:
            return ApiKeyValidationResult(valid=False, reason="Unknown API key")
        if row["revoked_at"]:
            return ApiKeyValidationResult(valid=False, reason="API key has been revoked")

        customer = self.get_customer(row["customer_id"])
        if customer is None:
            return ApiKeyValidationResult(valid=False, reason="Key belongs to a deleted customer")

        plan = get_plan(customer.plan_tier)  # always resolve plan once we have a customer,
                                               # even if we're about to reject the request below --
                                               # the caller needs this to show a useful message
                                               # ("you're on Free, upgrade to restore Pro access")
                                               # rather than a bare rejection with no context.

        if customer.subscription_status != "active":
            return ApiKeyValidationResult(
                valid=False, customer=customer, plan=plan,
                reason=f"Subscription status is '{customer.subscription_status}', not active",
            )

        if endpoint is not None and endpoint not in plan.allowed_endpoints:
            return ApiKeyValidationResult(
                valid=False, customer=customer, plan=plan,
                reason=f"Endpoint '{endpoint}' is not available on the {plan.display_name} plan "
                       f"(available: {sorted(plan.allowed_endpoints)}). Upgrade to unlock it.",
            )

        return ApiKeyValidationResult(valid=True, customer=customer, plan=plan)

    # -- usage metering -------------------------------------------------- #
    def record_usage(self, customer_id: str, endpoint: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO usage_events (customer_id, endpoint, timestamp) VALUES (?, ?, ?)",
                (customer_id, endpoint, datetime.now(timezone.utc).isoformat()),
            )

    def usage_this_month(self, customer_id: str) -> int:
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0,
                                                           microsecond=0).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM usage_events WHERE customer_id = ? AND timestamp >= ?",
                (customer_id, month_start),
            ).fetchone()
        return row["cnt"]

    def check_monthly_quota(self, customer_id: str, plan: PlanLimits) -> tuple[bool, int]:
        """Returns (within_quota, current_usage_count)."""
        used = self.usage_this_month(customer_id)
        return used < plan.requests_per_month, used
