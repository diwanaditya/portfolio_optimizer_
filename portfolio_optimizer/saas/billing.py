"""
Stripe Billing Integration.

Handles the two things a real SaaS needs from a payment provider:
  1. Creating a customer + checkout session so someone can actually pay you
  2. Listening for webhook events (subscription created/updated/cancelled,
     payment failed) and keeping `TenancyStore` in sync with what Stripe
     says is true -- Stripe's records are the source of truth for
     billing state, this module's job is to mirror that into the plan
     gating that `tenancy.py` enforces.

SCOPE HONESTY: this is real, correct Stripe API usage (checkout sessions,
webhook signature verification, subscription lifecycle events) -- it is
NOT a hosted payment page, a dunning/retry email system, tax calculation,
or invoicing UI. Stripe Checkout (linked below) already provides the
actual payment page; you don't need to build one. What you DO need before
this takes real payments: a real Stripe account (not test mode), your own
domain for success/cancel redirect URLs, and Stripe's webhook endpoint
configured to point at wherever `handle_webhook_event` is served from.

Requires environment variables:
    STRIPE_SECRET_KEY              -- from the Stripe dashboard
    STRIPE_WEBHOOK_SECRET          -- from the webhook endpoint's settings
    STRIPE_PRICE_ID_PRO            -- the Price object ID for the Pro plan
    STRIPE_PRICE_ID_INSTITUTIONAL  -- the Price object ID for the Institutional plan
"""
from __future__ import annotations
import os
import logging
from dataclasses import dataclass

from .plans import PlanTier, get_plan
from .tenancy import TenancyStore

logger = logging.getLogger("portfolio_optimizer.saas.billing")


def _require_stripe():
    try:
        import stripe
        return stripe
    except ImportError as e:
        raise ImportError("pip install stripe") from e


def _configure_stripe():
    stripe = _require_stripe()
    api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not api_key:
        raise RuntimeError("STRIPE_SECRET_KEY is not set -- billing cannot run without it.")
    stripe.api_key = api_key
    return stripe


@dataclass
class CheckoutSessionResult:
    checkout_url: str
    session_id: str


def create_checkout_session(customer_email: str, plan_tier: PlanTier,
                             success_url: str, cancel_url: str) -> CheckoutSessionResult:
    """Creates a real Stripe Checkout session -- the returned `checkout_url`
    is where you redirect the customer to actually enter payment details
    and subscribe. Stripe hosts that page; nothing to build for it.
    """
    stripe = _configure_stripe()
    plan = get_plan(plan_tier)
    price_id = os.environ.get(plan.stripe_price_id_env_var)
    if not price_id:
        raise RuntimeError(
            f"{plan.stripe_price_id_env_var} is not set -- create a recurring Price for the "
            f"{plan.display_name} plan (${plan.monthly_price_usd}/mo) in the Stripe dashboard "
            f"and set its Price ID as this environment variable."
        )

    session = stripe.checkout.Session.create(
        customer_email=customer_email,
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"plan_tier": plan_tier.value},
    )
    return CheckoutSessionResult(checkout_url=session.url, session_id=session.id)


def create_billing_portal_session(stripe_customer_id: str, return_url: str) -> str:
    """Returns a URL to Stripe's hosted billing portal, where an existing
    customer can update payment method, change plan, or cancel -- again,
    Stripe hosts this page; nothing to build.
    """
    stripe = _configure_stripe()
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id, return_url=return_url,
    )
    return session.url


def verify_and_parse_webhook(payload: bytes, signature_header: str):
    """Verifies the webhook actually came from Stripe (not a forged
    request hitting your endpoint) using the signing secret, and returns
    the parsed event. ALWAYS verify -- an unverified webhook endpoint lets
    anyone grant themselves a free subscription by POSTing a fake event.
    """
    stripe = _configure_stripe()
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET is not set -- refusing to process webhooks "
                            "without signature verification.")
    return stripe.Webhook.construct_event(payload, signature_header, webhook_secret)


def handle_webhook_event(event, tenancy_store: TenancyStore) -> dict:
    """Processes a verified Stripe event and updates TenancyStore
    accordingly -- this is the sync step that keeps plan gating correct
    as subscriptions are created, upgraded, downgraded, or cancelled.
    """
    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        stripe_customer_id = data["customer"]
        customer_email = data.get("customer_details", {}).get("email") or data.get("customer_email")
        plan_tier_str = data.get("metadata", {}).get("plan_tier", "pro")
        plan_tier = PlanTier(plan_tier_str)

        existing = tenancy_store.get_customer_by_stripe_id(stripe_customer_id)
        if existing:
            tenancy_store.update_customer_plan(existing.customer_id, plan_tier, "active",
                                                 data.get("subscription"))
            logger.info(f"Updated existing customer {existing.customer_id} to {plan_tier.value}")
            return {"action": "plan_updated", "customer_id": existing.customer_id}

        # Fall back to an email match before creating: Stripe issues a new
        # customer object on some re-subscription flows even for someone
        # who already has an account here under a different (e.g. earlier,
        # now-cancelled) stripe_customer_id. Without this check,
        # create_customer's UNIQUE(email) constraint would raise a raw,
        # unhandled sqlite3.IntegrityError here instead of doing the
        # obviously-correct thing (re-link the existing account).
        existing_by_email = tenancy_store.get_customer_by_email(customer_email) if customer_email else None
        if existing_by_email:
            tenancy_store.attach_stripe_customer_id(existing_by_email.customer_id, stripe_customer_id)
            tenancy_store.update_customer_plan(existing_by_email.customer_id, plan_tier, "active",
                                                 data.get("subscription"))
            logger.info(f"Re-linked existing customer {existing_by_email.customer_id} "
                        f"(matched by email) to new Stripe customer {stripe_customer_id}")
            return {"action": "relinked_by_email", "customer_id": existing_by_email.customer_id}

        customer = tenancy_store.create_customer(customer_email, plan_tier, stripe_customer_id)
        tenancy_store.update_customer_plan(customer.customer_id, plan_tier, "active",
                                             data.get("subscription"))
        raw_key = tenancy_store.issue_api_key(customer.customer_id)
        logger.info(f"Created new customer {customer.customer_id} on {plan_tier.value}")
        return {"action": "customer_created", "customer_id": customer.customer_id,
                 "api_key": raw_key}  # only place this ever appears -- must be emailed/shown once

    elif event_type == "customer.subscription.updated":
        stripe_customer_id = data["customer"]
        status = data["status"]  # active, past_due, canceled, unpaid, etc.
        customer = tenancy_store.get_customer_by_stripe_id(stripe_customer_id)
        if customer:
            tenancy_store.update_customer_plan(customer.customer_id, customer.plan_tier, status)
            logger.info(f"Customer {customer.customer_id} subscription status -> {status}")
            return {"action": "status_updated", "customer_id": customer.customer_id, "status": status}

    elif event_type == "customer.subscription.deleted":
        stripe_customer_id = data["customer"]
        customer = tenancy_store.get_customer_by_stripe_id(stripe_customer_id)
        if customer:
            tenancy_store.update_customer_plan(customer.customer_id, PlanTier.FREE, "canceled")
            logger.info(f"Customer {customer.customer_id} downgraded to Free (subscription cancelled)")
            return {"action": "downgraded_to_free", "customer_id": customer.customer_id}

    return {"action": "ignored", "event_type": event_type}
