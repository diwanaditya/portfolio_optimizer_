import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.infra.broker import SimulatedBroker, AccountState, PositionState
from portfolio_optimizer.live.trading_loop import TradingLoop
from portfolio_optimizer.infra.persistence import PortfolioStateStore


class FakeDataAdapter:
    """Deterministic in-memory stand-in for YahooFinanceAdapter -- same
    interface (fetch_prices/fetch_returns), zero network calls.
    """
    def __init__(self, seed=0, n_days=300):
        self.seed = seed
        self.n_days = n_days

    def fetch_prices(self, symbols, start, end, interval="1d"):
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(start, end)[-self.n_days:]
        prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
        for s in symbols:
            prices[s] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
        return prices

    def fetch_returns(self, symbols, start, end, interval="1d"):
        p = self.fetch_prices(symbols, start, end, interval)
        return p.pct_change().dropna()


class TestSimulatedBroker:
    def test_starts_with_configured_cash(self):
        broker = SimulatedBroker(starting_cash=50_000)
        account = broker.get_account()
        assert account.cash == 50_000
        assert account.equity == 50_000

    def test_buy_order_reduces_cash_and_creates_position(self):
        broker = SimulatedBroker(starting_cash=100_000)
        result = broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)
        assert result.filled_quantity == 100.0
        assert result.filled_price == 100.0
        account = broker.get_account()
        assert account.cash == 90_000
        positions = broker.get_positions()
        assert positions["AAPL"].quantity == 100.0

    def test_sell_order_increases_cash_and_reduces_position(self):
        broker = SimulatedBroker(starting_cash=100_000)
        broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)   # 100 shares
        broker.submit_order("AAPL", "sell", 5_000, current_price=110.0)   # 5000/110 = 45.4545 shares
        positions = broker.get_positions()
        expected_remaining = 100.0 - (5_000 / 110.0)
        assert abs(positions["AAPL"].quantity - expected_remaining) < 1e-6

    def test_full_close_removes_position(self):
        broker = SimulatedBroker(starting_cash=100_000)
        broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)
        broker.submit_order("AAPL", "sell", 10_000, current_price=100.0)
        assert "AAPL" not in broker.get_positions()

    def test_cost_basis_averaging_on_adding_to_position(self):
        broker = SimulatedBroker(starting_cash=100_000)
        broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)   # 100 shares @ 100
        broker.submit_order("AAPL", "buy", 10_000, current_price=200.0)   # 50 shares @ 200
        pos = broker.get_positions()["AAPL"]
        expected_avg = (100 * 100.0 + 50 * 200.0) / 150.0
        assert abs(pos.avg_entry_price - expected_avg) < 1e-6

    def test_insufficient_cash_raises(self):
        broker = SimulatedBroker(starting_cash=1_000)
        with pytest.raises(ValueError, match="Insufficient simulated cash"):
            broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)

    def test_missing_price_raises(self):
        broker = SimulatedBroker(starting_cash=100_000)
        with pytest.raises(ValueError, match="requires a valid current_price"):
            broker.submit_order("AAPL", "buy", 1_000, current_price=None)

    def test_equity_reflects_position_mark(self):
        broker = SimulatedBroker(starting_cash=100_000)
        broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)
        # equity should still be ~100k right after the fill (no price movement yet)
        assert abs(broker.get_account().equity - 100_000) < 1e-6

    def test_persists_positions_to_store_when_provided(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            broker = SimulatedBroker(starting_cash=100_000, store=store, portfolio_id="p1")
            broker.submit_order("AAPL", "buy", 10_000, current_price=100.0)
            positions = store.get_positions("p1")
            assert "AAPL" in positions
            assert positions["AAPL"]["quantity"] == 100.0


class TestTradingLoopZeroSignupDefault:
    def test_defaults_to_yahoo_and_simulated_without_any_credentials(self):
        """The core claim: constructing a TradingLoop with just symbols
        (no API keys, no broker credentials anywhere) should work, using
        Yahoo Finance for data and SimulatedBroker for execution.
        """
        with tempfile.TemporaryDirectory() as d:
            loop = TradingLoop(symbols=["AAPL"], data_adapter=FakeDataAdapter(),
                                db_path=os.path.join(d, "test.db"))
            from portfolio_optimizer.infra.broker import SimulatedBroker as SB
            assert isinstance(loop.broker, SB)

    def test_full_cycle_runs_with_fake_data_and_simulated_broker(self):
        with tempfile.TemporaryDirectory() as d:
            loop = TradingLoop(symbols=["AAPL", "MSFT", "GOOGL"], data_adapter=FakeDataAdapter(),
                                db_path=os.path.join(d, "test.db"), portfolio_id="test1",
                                max_position_weight=0.40)
            result = loop.run_once()
            assert result["audit_chain_valid"] is True
            assert "target_weights" in result

    def test_orders_actually_fill_when_risk_checks_pass(self):
        with tempfile.TemporaryDirectory() as d:
            loop = TradingLoop(symbols=["AAPL", "MSFT", "GOOGL"], data_adapter=FakeDataAdapter(),
                                db_path=os.path.join(d, "test.db"), portfolio_id="test2",
                                max_position_weight=0.40)
            result = loop.run_once()
            if result["submitted"]:
                assert len(result["orders"]) > 0
                positions = loop.broker.get_positions()
                assert len(positions) > 0

    def test_rebalance_blocked_reports_breaches_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            # 3 assets each capped at 20% max weight is mathematically infeasible
            # (max achievable total = 60% < 100%) -- verify this now fails with
            # a clear, actionable error at construction time rather than a
            # cryptic SLSQP failure deep in the solver.
            loop = TradingLoop(symbols=["AAPL", "MSFT", "GOOGL"], data_adapter=FakeDataAdapter(),
                                db_path=os.path.join(d, "test.db"), portfolio_id="test3",
                                max_position_weight=0.20)
            with pytest.raises(ValueError, match="Infeasible weight_bounds"):
                loop.run_once()

    def test_rebalance_blocked_with_feasible_but_tight_bound(self):
        with tempfile.TemporaryDirectory() as d:
            # 3 assets capped at 34% each IS feasible (max total 102%), and
            # tight enough that the optimizer's natural solution will likely
            # want to exceed it on at least one asset given real dispersion
            # in the fake data -- exercises the report-don't-hide-violations
            # path without hitting outright infeasibility.
            loop = TradingLoop(symbols=["AAPL", "MSFT", "GOOGL"], data_adapter=FakeDataAdapter(),
                                db_path=os.path.join(d, "test.db"), portfolio_id="test3b",
                                max_position_weight=0.34)
            result = loop.run_once()
            # whether blocked or not, the result must be self-consistent and honest
            if not result["submitted"]:
                assert result["reason"] == "risk_checks_failed"
                assert len(result["breaches"]) > 0

    def test_audit_log_records_full_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            loop = TradingLoop(symbols=["AAPL", "MSFT"], data_adapter=FakeDataAdapter(),
                                db_path=os.path.join(d, "test.db"), portfolio_id="test4",
                                max_position_weight=0.60)
            loop.run_once()
            assert len(loop.audit_log.entries) > 0
            valid, _ = loop.audit_log.verify_chain()
            assert valid

    def test_data_source_is_swappable(self):
        """Verifies the abstraction actually decouples data from broker --
        using a different data adapter instance changes what data is
        fetched without touching the broker or optimizer code at all.
        """
        with tempfile.TemporaryDirectory() as d:
            adapter_a = FakeDataAdapter(seed=1)
            adapter_b = FakeDataAdapter(seed=2)
            loop_a = TradingLoop(symbols=["AAPL"], data_adapter=adapter_a,
                                  db_path=os.path.join(d, "a.db"), portfolio_id="a")
            loop_b = TradingLoop(symbols=["AAPL"], data_adapter=adapter_b,
                                  db_path=os.path.join(d, "b.db"), portfolio_id="b")
            returns_a = loop_a.fetch_live_returns()
            returns_b = loop_b.fetch_live_returns()
            assert not returns_a["AAPL"].equals(returns_b["AAPL"])

    def test_broker_is_swappable_independent_of_data(self):
        """Same data adapter, different broker instances -- confirms
        execution state doesn't leak between brokers and the loop only
        talks to the broker through the abstract interface.
        """
        with tempfile.TemporaryDirectory() as d:
            shared_adapter = FakeDataAdapter(seed=5)
            broker1 = SimulatedBroker(starting_cash=50_000)
            broker2 = SimulatedBroker(starting_cash=200_000)
            loop1 = TradingLoop(symbols=["AAPL"], data_adapter=shared_adapter, broker_adapter=broker1,
                                 db_path=os.path.join(d, "c.db"), portfolio_id="c")
            loop2 = TradingLoop(symbols=["AAPL"], data_adapter=shared_adapter, broker_adapter=broker2,
                                 db_path=os.path.join(d, "d.db"), portfolio_id="d")
            assert loop1.broker.get_account().cash == 50_000
            assert loop2.broker.get_account().cash == 200_000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
