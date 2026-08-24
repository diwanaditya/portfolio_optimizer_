"""
Live Paper-Trading Loop -- Alpaca (paper account).

This is the piece that was missing: everything else in this repo (data
adapters, optimizers, risk controls, OMS, audit log) was real but never
connected end-to-end into something that actually runs. This module does
that, against Alpaca's PAPER trading API specifically -- real live market
data, a real broker connection, real order execution mechanics, and
*zero actual capital at risk*, since Alpaca paper accounts trade with
simulated money against real market prices.

WHY PAPER, NOT LIVE, AND WHY THAT'S THE RIGHT CALL RIGHT NOW:
  - The novel method in this repo (SA-CVaR-RP) has a disclosed NULL
    result on its own validation test -- it doesn't outperform anything.
  - The RL agents were trained on demo-scale budgets, not production ones.
  - Every backtest in this repo so far has run on synthetic data, not
    real market history.
  - SEBI/compliance review for Indian equities is explicitly unresolved.
  This loop is how you'd build the track record and confidence needed
  before any of that changes -- running it here doesn't skip that step,
  it's the honest way to take it.

REQUIRES: an Alpaca paper-trading account (free at alpaca.markets) and
its API key/secret set as environment variables:
    ALPACA_API_KEY, ALPACA_SECRET_KEY

USAGE:
    python -m portfolio_optimizer.live.paper_trading_loop --symbols AAPL MSFT GOOGL --once

Run with --once for a single rebalance cycle (safe to test), or without
it for a scheduled loop (rebalances every `rebalance_interval_hours`).
"""
from __future__ import annotations
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
import math
import hashlib
import json

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_optimizer.live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class AlpacaPaperTradingLoop:
    def __init__(self, symbols: list, lookback_days: int = 252,
                 max_position_weight: float = 0.35, db_path: str = "live_portfolio.db",
                 portfolio_id: str = "adc_paper_v1"):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as e:
            raise ImportError("pip install alpaca-py") from e

        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. Get free paper-trading "
                "credentials at https://alpaca.markets -- this loop refuses to run without "
                "them rather than silently doing nothing."
            )

        self.symbols = symbols
        self.lookback_days = lookback_days
        self.portfolio_id = portfolio_id
        self.max_daily_loss_pct = float(os.environ.get("LIVE_MAX_DAILY_LOSS_PCT", "0.05"))
        self.max_drawdown_pct = float(os.environ.get("LIVE_MAX_DRAWDOWN_PCT", "0.15"))
        self.max_order_notional_pct = float(os.environ.get("LIVE_MAX_ORDER_NOTIONAL_PCT", "0.35"))
        self.max_orders_per_cycle = int(os.environ.get("LIVE_MAX_ORDERS_PER_CYCLE", "25"))
        self._validate_operational_limits()

        self.trading_client = TradingClient(api_key, api_secret, paper=True)  # paper=True is load-bearing
        self.data_client = StockHistoricalDataClient(api_key, api_secret)

        from ..infra.persistence import PortfolioStateStore
        from ..infra.audit_log import TamperEvidentAuditLog
        from ..infra.risk_controls import PreTradeRiskEngine, PreTradeLimits
        from ..infra.oms import OrderManager
        from ..infra.fault_tolerance import retry_with_backoff, CircuitBreaker

        self.store = PortfolioStateStore(db_path)
        self.audit_log = TamperEvidentAuditLog()
        self.risk_engine = PreTradeRiskEngine(
            PreTradeLimits(max_position_weight=max_position_weight, max_gross_leverage=1.0)
        )
        self.order_manager = OrderManager()
        self.data_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        self._retry_with_backoff = retry_with_backoff

    def _validate_operational_limits(self):
        if not (0 < self.max_daily_loss_pct < 1):
            raise ValueError("LIVE_MAX_DAILY_LOSS_PCT must be between 0 and 1")
        if not (0 < self.max_drawdown_pct < 1):
            raise ValueError("LIVE_MAX_DRAWDOWN_PCT must be between 0 and 1")
        if not (0 < self.max_order_notional_pct <= 1):
            raise ValueError("LIVE_MAX_ORDER_NOTIONAL_PCT must be > 0 and <= 1")
        if self.max_orders_per_cycle < 1:
            raise ValueError("LIVE_MAX_ORDERS_PER_CYCLE must be >= 1")

    def _emergency_stop_active(self) -> bool:
        return os.environ.get("LIVE_EMERGENCY_STOP", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _check_loss_limits(self, equity: float) -> None:
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError("Unsafe broker equity value; refusing to trade.")
        history = self.store.get_nav_history(self.portfolio_id)
        if not history:
            return
        peak = max(float(row["nav"]) for row in history if float(row["nav"]) > 0)
        drawdown = 1.0 - (equity / peak)
        if drawdown >= self.max_drawdown_pct:
            raise RuntimeError(
                f"MAX_DRAWDOWN_LIMIT: current equity is {drawdown:.2%} below recorded peak; refusing to trade."
            )
        today = datetime.now(timezone.utc).date()
        todays = [row for row in history if datetime.fromisoformat(row["timestamp"]).date() == today and float(row["nav"]) > 0]
        if todays:
            day_start = float(todays[0]["nav"])
            daily_loss = 1.0 - (equity / day_start)
            if daily_loss >= self.max_daily_loss_pct:
                raise RuntimeError(
                    f"MAX_DAILY_LOSS_LIMIT: current equity is {daily_loss:.2%} below today's recorded baseline; refusing to trade."
                )

    def _reconcile_broker_state(self) -> None:
        """Fail closed if broker positions disagree with our persisted state."""
        broker_positions = self.trading_client.get_all_positions()
        broker = {p.symbol: float(p.qty) for p in broker_positions if p.symbol in self.symbols}
        stored = self.store.get_positions(self.portfolio_id)
        stored_qty = {symbol: float(row["quantity"]) for symbol, row in stored.items() if symbol in self.symbols}
        symbols = set(broker) | set(stored_qty)
        mismatches = [
            (symbol, stored_qty.get(symbol, 0.0), broker.get(symbol, 0.0))
            for symbol in symbols
            if abs(stored_qty.get(symbol, 0.0) - broker.get(symbol, 0.0)) > 1e-8
        ]
        if mismatches:
            self.audit_log.record("broker_reconciliation_failed", {"mismatches": mismatches})
            raise RuntimeError(
                "BROKER_RECONCILIATION_FAILED: broker positions differ from local state; "
                "no orders will be submitted until the discrepancy is investigated."
            )
        self.audit_log.record("broker_reconciliation_ok", {"positions": broker})

    def _intent_client_order_id(self, symbol: str, side: str, notional: float,
                                target_weight: float, current_weight: float) -> str:
        payload = {
            "portfolio_id": self.portfolio_id, "symbol": symbol, "side": side,
            "notional": round(notional, 2), "target_weight": round(target_weight, 8),
            "current_weight": round(current_weight, 8),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]
        return f"PO-{digest}"

    def _existing_broker_order(self, client_order_id: str):
        """Find an already-submitted broker order so retries remain idempotent."""
        try:
            orders = self.trading_client.get_orders()
        except Exception as exc:
            raise RuntimeError(f"BROKER_ORDER_LOOKUP_FAILED: refusing retry after lookup failure: {exc}") from exc
        for broker_order in orders or []:
            if str(getattr(broker_order, "client_order_id", "")) == client_order_id:
                return broker_order
        return None

    # -- data ------------------------------------------------------------ #
    def fetch_live_returns(self) -> pd.DataFrame:
        """Pulls real historical daily bars from Alpaca and computes
        returns -- this replaces the synthetic data used everywhere else
        in this repo's demos with actual market history.
        """
        @self._retry_with_backoff(max_attempts=3, base_delay=1.0)
        def _fetch():
            return self.data_circuit_breaker.call(self._do_fetch)

        return _fetch()

    def _do_fetch(self) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(self.lookback_days * 1.6))  # buffer for weekends/holidays
        req = StockBarsRequest(symbol_or_symbols=self.symbols, timeframe=TimeFrame.Day,
                                start=start, end=end)
        bars = self.data_client.get_stock_bars(req).df
        prices = bars.reset_index().pivot(index="timestamp", columns="symbol", values="close")
        prices = prices.dropna(how="all").ffill()

        # basic sanity validation -- this is the "does the data look sane"
        # check flagged as missing from the raw adapters: reject obviously
        # bad ticks (zero/negative prices) and flag insufficient history
        # rather than silently feeding garbage into the optimizer.
        if (prices <= 0).any().any():
            bad_cols = prices.columns[(prices <= 0).any()].tolist()
            raise ValueError(f"Non-positive prices detected for: {bad_cols} -- refusing to proceed.")

        returns = prices.pct_change().dropna(how="all").tail(self.lookback_days)
        if len(returns) < self.lookback_days * 0.5:
            raise ValueError(
                f"Only got {len(returns)} days of return history (wanted ~{self.lookback_days}); "
                f"refusing to optimize on insufficient data."
            )
        return returns

    # -- optimization ------------------------------------------------------- #
    def compute_target_weights(self, returns: pd.DataFrame) -> pd.Series:
        from ..estimators.expected_returns import mean_historical_return
        from ..estimators.covariance import ledoit_wolf_shrinkage
        from ..optimizers.markowitz import MarkowitzOptimizer

        mu = mean_historical_return(returns)
        cov, shrinkage = ledoit_wolf_shrinkage(returns)
        opt = MarkowitzOptimizer(mu, cov, risk_free_rate=0.04,
                                  weight_bounds=(0.0, self.risk_engine.limits.max_position_weight))
        result = opt.max_sharpe()
        if not result.success:
            raise RuntimeError(f"Optimizer failed to converge: {result.message}")
        logger.info(f"Optimizer solved: Sharpe={result.sharpe_ratio:.2f}, "
                    f"shrinkage={shrinkage:.3f}")
        return result.weights

    # -- account state ------------------------------------------------------- #
    def get_current_weights(self):
        """Reads ACTUAL current positions from the broker (not from our own
        state store) -- the broker is always the source of truth for what
        you actually hold, our local persistence is a cache/audit record.
        """
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()
        equity = float(account.equity)

        weights = pd.Series(0.0, index=self.symbols)
        for p in positions:
            if p.symbol in weights.index:
                weights[p.symbol] = float(p.market_value) / equity
        return weights, equity

    # -- execution ------------------------------------------------------- #
    def rebalance(self, target_weights: pd.Series, current_weights: pd.Series, equity: float,
                   min_trade_notional: float = 50.0) -> dict:
        """Submits real (paper) orders to move from current to target
        weights, through the full OMS + risk-control + audit pipeline --
        every trade is a genuine Alpaca paper order, tracked through a
        real order-state-machine, checked against pre-trade limits, and
        recorded in the tamper-evident audit log before and after submission.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        risk_results = self.risk_engine.check_target_portfolio(target_weights, current_weights)
        approved = self.risk_engine.is_approved(risk_results)
        self.audit_log.record("rebalance_risk_check", {
            "target_weights": target_weights.round(4).to_dict(),
            "current_weights": current_weights.round(4).to_dict(),
            "approved": approved,
            "breaches": [r.message for r in risk_results if not r.passed],
        })

        if not approved:
            logger.warning("Rebalance BLOCKED by pre-trade risk checks -- no orders submitted.")
            return {"submitted": False, "reason": "risk_checks_failed",
                    "breaches": [r.message for r in risk_results if not r.passed]}

        submitted_orders = []
        for symbol in self.symbols:
            if len(submitted_orders) >= self.max_orders_per_cycle:
                raise RuntimeError("MAX_ORDERS_PER_CYCLE reached; refusing additional order submissions.")
            trade_weight = target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)
            trade_notional = trade_weight * equity
            if abs(trade_notional) < min_trade_notional:
                continue  # skip dust trades

            side = OrderSide.BUY if trade_notional > 0 else OrderSide.SELL
            if abs(trade_notional) > equity * self.max_order_notional_pct:
                raise RuntimeError(
                    f"MAX_ORDER_NOTIONAL_LIMIT: {symbol} trade is ${abs(trade_notional):,.2f}, "
                    f"above the configured {self.max_order_notional_pct:.0%} of equity."
                )
            client_order_id = self._intent_client_order_id(
                symbol, side.value.lower(), abs(trade_notional),
                float(target_weights.get(symbol, 0.0)), float(current_weights.get(symbol, 0.0)),
            )
            order = self.order_manager.submit(
                symbol, side.value.lower(), abs(trade_notional), client_order_id=client_order_id
            )
            existing = self._existing_broker_order(order.client_order_id)
            if existing is not None:
                order.acknowledge()
                self.store.save_order(self.portfolio_id, order)
                self.audit_log.record("order_retry_deduplicated", {
                    "client_order_id": order.client_order_id, "broker_order_id": str(existing.id),
                })
                submitted_orders.append({"symbol": symbol, "side": side.value,
                                         "notional": abs(trade_notional),
                                         "broker_order_id": str(existing.id), "deduplicated": True})
                continue
            self.audit_log.record("order_submitted", {
                "symbol": symbol, "side": side.value, "notional": abs(trade_notional),
                "client_order_id": order.client_order_id,
            })

            try:
                alpaca_req = MarketOrderRequest(
                    symbol=symbol, notional=round(abs(trade_notional), 2),
                    side=side, time_in_force=TimeInForce.DAY,
                    client_order_id=order.client_order_id,
                )
                alpaca_order = self.trading_client.submit_order(alpaca_req)
                order.acknowledge()
                self.store.save_order(self.portfolio_id, order)
                self.audit_log.record("order_acknowledged", {
                    "client_order_id": order.client_order_id, "broker_order_id": str(alpaca_order.id),
                })
                submitted_orders.append({"symbol": symbol, "side": side.value,
                                          "notional": abs(trade_notional),
                                          "broker_order_id": str(alpaca_order.id)})
                logger.info(f"Submitted {side.value} order: {symbol} ${abs(trade_notional):.2f}")
            except Exception as e:
                order.reject(str(e))
                self.store.save_order(self.portfolio_id, order)
                self.audit_log.record("order_rejected", {
                    "client_order_id": order.client_order_id, "reason": str(e),
                })
                logger.error(f"Order failed for {symbol}: {e}")

        self.store.record_nav(self.portfolio_id, equity, 0.0,
                               {"target_weights": target_weights.round(4).to_dict()})
        return {"submitted": True, "orders": submitted_orders}

    # -- main cycle ------------------------------------------------------- #
    def run_once(self) -> dict:
        logger.info(f"Starting rebalance cycle for {self.symbols}")
        if self._emergency_stop_active():
            self.audit_log.record("emergency_stop_block", {"portfolio_id": self.portfolio_id})
            return {"submitted": False, "reason": "emergency_stop_active"}
        self._reconcile_broker_state()
        returns = self.fetch_live_returns()
        target_weights = self.compute_target_weights(returns)
        current_weights, equity = self.get_current_weights()
        logger.info(f"Account equity: ${equity:,.2f}")
        self._check_loss_limits(equity)
        result = self.rebalance(target_weights, current_weights, equity)

        chain_valid, broken_at = self.audit_log.verify_chain()
        if not chain_valid:
            logger.critical(f"AUDIT LOG INTEGRITY FAILURE at index {broken_at} -- investigate immediately.")

        return {**result, "target_weights": target_weights.round(4).to_dict(),
                "equity": equity, "audit_chain_valid": chain_valid}

    def run_scheduled(self, rebalance_interval_hours: float = 24.0):
        logger.info(f"Starting scheduled loop, rebalancing every {rebalance_interval_hours}h. "
                    f"Ctrl+C to stop.")
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Rebalance cycle failed -- will retry next scheduled interval.")
            time.sleep(rebalance_interval_hours * 3600)


def main():
    parser = argparse.ArgumentParser(description="Alpaca paper-trading loop")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--max-position-weight", type=float, default=0.35)
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    args = parser.parse_args()

    loop = AlpacaPaperTradingLoop(symbols=args.symbols, lookback_days=args.lookback_days,
                                   max_position_weight=args.max_position_weight)
    if args.once:
        result = loop.run_once()
        print(result)
    else:
        loop.run_scheduled(args.interval_hours)


if __name__ == "__main__":
    main()
