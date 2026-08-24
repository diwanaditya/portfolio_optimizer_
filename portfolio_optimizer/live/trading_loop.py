"""
Broker-Agnostic Live Trading Loop.

This supersedes `live/paper_trading_loop.py`'s hard-wired Alpaca-only
design. Data source and execution venue are now two independent,
swappable pieces:

    data adapter   : any `data.adapters.LiveDataAdapter`     (default: Yahoo Finance, no API key)
    broker adapter : any `infra.broker.BrokerAdapter`         (default: SimulatedBroker, no account)

The zero-signup default (`TradingLoop(symbols=[...])` with no other
arguments) needs literally nothing external: Yahoo Finance for real live
prices, a fully local simulated account for execution. Nothing to sign up
for, nothing to configure, no API key anywhere in the path.

If you specifically want real broker order mechanics later, swap in
`AlpacaBroker` (or write your own `BrokerAdapter`) without touching
anything else -- the optimizer, risk engine, OMS, persistence, and audit
log don't know or care which broker (or data source) is behind the
interface.

USAGE:
    from portfolio_optimizer.data.adapters import YahooFinanceAdapter
    from portfolio_optimizer.infra.broker import SimulatedBroker
    from portfolio_optimizer.live.trading_loop import TradingLoop

    loop = TradingLoop(symbols=["AAPL", "MSFT", "GOOGL"])   # yfinance + simulated, zero setup
    loop.run_once()
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_optimizer.live")


class TradingLoop:
    def __init__(self, symbols: list, data_adapter=None, broker_adapter=None,
                 lookback_days: int = 252, max_position_weight: float = 0.35,
                 db_path: str = "live_portfolio.db", portfolio_id: str = "adc_live_v2",
                 min_trade_notional: float = 50.0):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.portfolio_id = portfolio_id
        self.min_trade_notional = min_trade_notional

        if data_adapter is None:
            from ..data.adapters import YahooFinanceAdapter
            data_adapter = YahooFinanceAdapter()
        self.data_adapter = data_adapter

        from ..infra.persistence import PortfolioStateStore
        from ..infra.audit_log import TamperEvidentAuditLog
        from ..infra.risk_controls import PreTradeRiskEngine, PreTradeLimits
        from ..infra.oms import OrderManager
        from ..infra.fault_tolerance import retry_with_backoff, CircuitBreaker

        self.store = PortfolioStateStore(db_path)

        if broker_adapter is None:
            from ..infra.broker import SimulatedBroker
            broker_adapter = SimulatedBroker(store=self.store, portfolio_id=portfolio_id)
        self.broker = broker_adapter

        self.audit_log = TamperEvidentAuditLog()
        self.risk_engine = PreTradeRiskEngine(
            PreTradeLimits(max_position_weight=max_position_weight, max_gross_leverage=1.0)
        )
        self.order_manager = OrderManager()
        self.data_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        self._retry_with_backoff = retry_with_backoff

    # -- data -------------------------------------------------------------- #
    def fetch_live_returns(self) -> pd.DataFrame:
        @self._retry_with_backoff(max_attempts=3, base_delay=1.0)
        def _fetch():
            return self.data_circuit_breaker.call(self._do_fetch)
        return _fetch()

    def _do_fetch(self) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(self.lookback_days * 1.6))
        prices = self.data_adapter.fetch_prices(
            self.symbols, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )
        prices = prices.dropna(how="all").ffill()

        if (prices <= 0).any().any():
            bad_cols = prices.columns[(prices <= 0).any()].tolist()
            raise ValueError(f"Non-positive prices detected for: {bad_cols} -- refusing to proceed.")

        returns = prices.pct_change().dropna(how="all").tail(self.lookback_days)
        if len(returns) < self.lookback_days * 0.5:
            raise ValueError(
                f"Only got {len(returns)} days of return history (wanted ~{self.lookback_days}); "
                f"refusing to optimize on insufficient data."
            )
        self._last_prices = prices
        return returns

    def get_current_prices(self) -> pd.Series:
        """Latest close for each symbol from the same data adapter used
        for history -- kept as a separate, explicit step (not silently
        reused from `_do_fetch`) so brokers get an honestly-labeled price
        to fill against.
        """
        if hasattr(self, "_last_prices") and self._last_prices is not None:
            return self._last_prices.iloc[-1]
        prices = self.data_adapter.fetch_prices(
            self.symbols,
            (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d"),
            datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        return prices.iloc[-1]

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
        logger.info(f"Optimizer solved: Sharpe={result.sharpe_ratio:.2f}, shrinkage={shrinkage:.3f}")
        return result.weights

    # -- account state ------------------------------------------------------- #
    def get_current_weights(self):
        """Reads current positions from the BROKER (whichever one is
        wired in) -- always the source of truth for what's actually held,
        regardless of whether that's a real Alpaca account or a local
        simulated one.
        """
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        equity = account.equity

        weights = pd.Series(0.0, index=self.symbols)
        for symbol, pos in positions.items():
            if symbol in weights.index and equity > 0:
                weights[symbol] = pos.market_value / equity
        return weights, equity

    # -- execution ------------------------------------------------------- #
    def rebalance(self, target_weights: pd.Series, current_weights: pd.Series, equity: float,
                   current_prices: pd.Series) -> dict:
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
            trade_weight = target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)
            trade_notional = trade_weight * equity
            if abs(trade_notional) < self.min_trade_notional:
                continue

            side = "buy" if trade_notional > 0 else "sell"
            order = self.order_manager.submit(symbol, side, abs(trade_notional))
            self.audit_log.record("order_submitted", {
                "symbol": symbol, "side": side, "notional": abs(trade_notional),
                "client_order_id": order.client_order_id,
            })

            try:
                price = float(current_prices.get(symbol, np.nan))
                result = self.broker.submit_order(symbol, side, abs(trade_notional), current_price=price)
                order.acknowledge()
                order.apply_fill(result.filled_quantity, result.filled_price)
                self.store.save_order(self.portfolio_id, order)
                self.audit_log.record("order_filled", {
                    "client_order_id": order.client_order_id,
                    "broker_order_id": result.broker_order_id,
                    "filled_quantity": result.filled_quantity, "filled_price": result.filled_price,
                })
                submitted_orders.append({"symbol": symbol, "side": side,
                                          "notional": abs(trade_notional),
                                          "broker_order_id": result.broker_order_id})
                logger.info(f"Filled {side} order: {symbol} ${abs(trade_notional):.2f} "
                            f"@ ${result.filled_price:.2f}")
            except Exception as e:
                order.reject(str(e))
                self.store.save_order(self.portfolio_id, order)
                self.audit_log.record("order_rejected", {
                    "client_order_id": order.client_order_id, "reason": str(e),
                })
                logger.error(f"Order failed for {symbol}: {e}")

        final_equity = self.broker.get_account().equity
        self.store.record_nav(self.portfolio_id, final_equity, self.broker.get_account().cash,
                               {"target_weights": target_weights.round(4).to_dict()})
        return {"submitted": True, "orders": submitted_orders}

    # -- main cycle ------------------------------------------------------- #
    def run_once(self) -> dict:
        logger.info(f"Starting rebalance cycle for {self.symbols} "
                    f"(data: {type(self.data_adapter).__name__}, broker: {type(self.broker).__name__})")
        returns = self.fetch_live_returns()
        target_weights = self.compute_target_weights(returns)
        current_prices = self.get_current_prices()
        current_weights, equity = self.get_current_weights()
        logger.info(f"Account equity: ${equity:,.2f}")
        result = self.rebalance(target_weights, current_weights, equity, current_prices)

        chain_valid, broken_at = self.audit_log.verify_chain()
        if not chain_valid:
            logger.critical(f"AUDIT LOG INTEGRITY FAILURE at index {broken_at} -- investigate immediately.")

        return {**result, "target_weights": target_weights.round(4).to_dict(),
                "equity": equity, "audit_chain_valid": chain_valid}

    def run_scheduled(self, rebalance_interval_hours: float = 24.0):
        import time
        logger.info(f"Starting scheduled loop, rebalancing every {rebalance_interval_hours}h.")
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Rebalance cycle failed -- will retry next scheduled interval.")
            time.sleep(rebalance_interval_hours * 3600)
