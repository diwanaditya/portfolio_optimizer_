#!/usr/bin/env python3
"""
START HERE -- the one file to run to go live (paper trading).

WHAT TO DO:
    1. cp .env.example .env
    2. Open .env and paste your Alpaca paper-trading keys into it
       (get free ones at https://app.alpaca.markets/signup -> Paper Trading -> API Keys)
    3. Run:  python start_live_trading.py --once      (single test cycle)
       or:   python start_live_trading.py             (repeats on a schedule)

That's it. Nothing else needs editing unless you want to change which
symbols it trades or the risk limits -- those are also in .env.

WHAT THIS ACTUALLY DOES:
    Reads your .env file -> pulls real live prices for your symbols from
    Alpaca -> runs the Markowitz optimizer -> checks the result against
    your risk limits -> if it passes, submits real (PAPER, not real-money)
    orders through Alpaca -> records everything to a local database and a
    tamper-evident audit log -> regenerates dashboard.html so you can see
    what happened.

This trades with Alpaca's paper account (simulated money, real prices).
No real capital is at risk running this. See README.md Section 7 for why
that's the right place to start, not a limitation to work around.
"""
import os
import sys
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in this directory and loads it into os.environ
except ImportError:
    print("NOTE: python-dotenv not installed (pip install python-dotenv). "
          "Falling back to whatever's already in your shell's environment variables.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check_env_configured():
    missing = []
    if not os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY") == "PASTE_YOUR_ALPACA_KEY_HERE":
        missing.append("ALPACA_API_KEY")
    if not os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY") == "PASTE_YOUR_ALPACA_SECRET_HERE":
        missing.append("ALPACA_SECRET_KEY")
    if missing:
        print(f"\nMissing or unfilled: {', '.join(missing)}\n")
        print("Fix this by:")
        print("  1. cp .env.example .env      (if you haven't already)")
        print("  2. Open .env in any text editor")
        print("  3. Replace the placeholder text with your real Alpaca paper-trading keys")
        print("     (get them free at https://app.alpaca.markets/signup -> Paper Trading -> API Keys)")
        print("  4. Run this script again\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start ADC's live (paper) trading loop")
    parser.add_argument("--once", action="store_true", help="Run a single rebalance cycle and exit")
    parser.add_argument("--interval-hours", type=float, default=24.0,
                         help="Hours between rebalances when not using --once")
    parser.add_argument("--dashboard-only", action="store_true",
                         help="Just regenerate dashboard.html from existing data, don't trade")
    args = parser.parse_args()

    _check_env_configured()

    symbols = [s.strip() for s in os.environ.get("LIVE_TRADING_SYMBOLS", "AAPL,MSFT,GOOGL").split(",")]
    max_position_weight = float(os.environ.get("LIVE_MAX_POSITION_WEIGHT", "0.35"))
    lookback_days = int(os.environ.get("LIVE_LOOKBACK_DAYS", "252"))
    db_path = "live_portfolio.db"
    portfolio_id = "adc_live_v1"

    from portfolio_optimizer.live.paper_trading_loop import AlpacaPaperTradingLoop
    from portfolio_optimizer.dashboard.generator import generate_dashboard

    if args.dashboard_only:
        try:
            path = generate_dashboard(db_path, portfolio_id, "dashboard.html")
            print(f"Dashboard regenerated: {path}")
        except ValueError as e:
            print(f"Can't generate dashboard yet: {e}")
            print("Run a trading cycle first (python start_live_trading.py --once)")
        return

    print(f"Starting ADC live paper-trading loop")
    print(f"  Symbols: {symbols}")
    print(f"  Max position weight: {max_position_weight:.0%}")
    print(f"  Lookback: {lookback_days} days")
    print(f"  Mode: {'single cycle' if args.once else f'scheduled every {args.interval_hours}h'}")
    print()

    loop = AlpacaPaperTradingLoop(
        symbols=symbols, lookback_days=lookback_days,
        max_position_weight=max_position_weight, db_path=db_path, portfolio_id=portfolio_id,
    )

    if args.once:
        try:
            result = loop.run_once()
        except Exception as e:
            _print_friendly_error(e)
            sys.exit(1)
        print("\nCycle result:")
        print(result)
        try:
            path = generate_dashboard(db_path, portfolio_id, "dashboard.html")
            print(f"\nDashboard updated: {path} -- open it in a browser to see current state.")
        except ValueError:
            pass
    else:
        # regenerate the dashboard after every cycle, not just at the end
        original_run_once = loop.run_once

        def run_once_and_refresh_dashboard():
            result = original_run_once()
            try:
                generate_dashboard(db_path, portfolio_id, "dashboard.html")
            except ValueError:
                pass
            return result

        loop.run_once = run_once_and_refresh_dashboard
        loop.run_scheduled(args.interval_hours)


def _print_friendly_error(e: Exception):
    msg = str(e)
    print("\n" + "=" * 70)
    if "403" in msg or "401" in msg or "Forbidden" in msg or "Unauthorized" in msg:
        print("AUTHENTICATION FAILED talking to Alpaca.")
        print()
        print("This almost always means the keys in your .env file are wrong,")
        print("expired, or copy-pasted with extra whitespace. Fix by:")
        print("  1. Go to https://app.alpaca.markets -> Paper Trading -> API Keys")
        print("  2. Generate a fresh key pair")
        print("  3. Open .env and replace ALPACA_API_KEY / ALPACA_SECRET_KEY exactly")
        print("     (no quotes, no extra spaces)")
        print("  4. Run this script again")
    elif "ConnectionError" in type(e).__name__ or "Timeout" in type(e).__name__:
        print("COULDN'T REACH ALPACA -- looks like a network issue, not a code problem.")
        print("Check your internet connection and try again.")
    elif "Non-positive prices" in msg or "insufficient data" in msg:
        print("DATA VALIDATION FAILED -- this is the system correctly refusing to")
        print("trade on bad data rather than a bug. Details:")
        print(f"  {msg}")
    elif "risk_checks_failed" in msg:
        print("REBALANCE BLOCKED by pre-trade risk limits -- this is expected")
        print("behavior, not an error. Check dashboard.html / the audit log for details.")
    else:
        print("Something went wrong that isn't one of the common cases above.")
        print(f"Error: {type(e).__name__}: {msg}")
        print()
        print("Full details for debugging:")
        import traceback
        traceback.print_exc()
    print("=" * 70)


if __name__ == "__main__":
    main()
