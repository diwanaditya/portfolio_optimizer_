import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")

from portfolio_optimizer.live.paper_trading_loop import AlpacaPaperTradingLoop


def _make_loop(tmp_path, symbols=("A", "B", "C")):
    with patch("alpaca.trading.client.TradingClient"), \
         patch("alpaca.data.historical.StockHistoricalDataClient"):
        loop = AlpacaPaperTradingLoop(
            symbols=list(symbols), lookback_days=100,
            db_path=str(tmp_path / "test_live.db"), portfolio_id="test_port",
        )
    return loop


class TestConstruction:
    def test_refuses_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            AlpacaPaperTradingLoop(symbols=["AAPL"])

    def test_constructs_with_credentials(self, tmp_path):
        loop = _make_loop(tmp_path)
        assert loop.symbols == ["A", "B", "C"]
        assert loop.risk_engine is not None
        assert loop.order_manager is not None


class TestDataFetching:
    def test_do_fetch_rejects_non_positive_prices(self, tmp_path):
        loop = _make_loop(tmp_path)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        bad_prices = pd.DataFrame(
            {"A": [100.0] * 50, "B": [-5.0] * 50, "C": [50.0] * 50}, index=dates
        )
        mock_bars_df = bad_prices.stack().rename("close").reset_index()
        mock_bars_df.columns = ["timestamp", "symbol", "close"]
        mock_bars_df = mock_bars_df.set_index(["symbol", "timestamp"])

        mock_response = MagicMock()
        mock_response.df = mock_bars_df
        loop.data_client.get_stock_bars = MagicMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Non-positive prices"):
            loop._do_fetch()

    def test_do_fetch_rejects_insufficient_history(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.lookback_days = 200
        dates = pd.date_range("2024-01-01", periods=5, freq="D")  # way too little
        prices = pd.DataFrame({"A": [100.0] * 5, "B": [50.0] * 5}, index=dates)
        mock_bars_df = prices.stack().rename("close").reset_index()
        mock_bars_df.columns = ["timestamp", "symbol", "close"]
        mock_bars_df = mock_bars_df.set_index(["symbol", "timestamp"])

        mock_response = MagicMock()
        mock_response.df = mock_bars_df
        loop.data_client.get_stock_bars = MagicMock(return_value=mock_response)

        with pytest.raises(ValueError, match="insufficient data"):
            loop._do_fetch()

    def test_do_fetch_accepts_valid_data(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.lookback_days = 50
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        prices = pd.DataFrame({
            "A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 120)),
            "B": 50 * np.cumprod(1 + rng.normal(0, 0.01, 120)),
        }, index=dates)
        mock_bars_df = prices.stack().rename("close").reset_index()
        mock_bars_df.columns = ["timestamp", "symbol", "close"]
        mock_bars_df = mock_bars_df.set_index(["symbol", "timestamp"])

        mock_response = MagicMock()
        mock_response.df = mock_bars_df
        loop.data_client.get_stock_bars = MagicMock(return_value=mock_response)

        returns = loop._do_fetch()
        assert set(returns.columns) == {"A", "B"}
        assert len(returns) > 0
        assert not returns.isna().any().any()


class TestOptimization:
    def test_compute_target_weights_valid(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B", "C"))
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(rng.standard_normal((200, 3)) * 0.01, columns=["A", "B", "C"])
        weights = loop.compute_target_weights(returns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-3)
        assert (weights >= -1e-6).all()
        assert (weights <= loop.risk_engine.limits.max_position_weight + 1e-6).all()


class TestRebalanceRiskGating:
    def test_rebalance_blocked_when_risk_check_fails(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        # Force a concentration breach by using a target that exceeds the limit
        target = pd.Series({"A": 0.9, "B": 0.1})
        current = pd.Series({"A": 0.0, "B": 0.0})
        result = loop.rebalance(target, current, equity=100_000)
        assert result["submitted"] is False
        assert result["reason"] == "risk_checks_failed"
        assert len(result["breaches"]) > 0

    def test_rebalance_submits_orders_when_approved(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.trading_client.submit_order = MagicMock(return_value=MagicMock(id="order-123"))

        target = pd.Series({"A": 0.3, "B": 0.3})
        current = pd.Series({"A": 0.0, "B": 0.0})
        result = loop.rebalance(target, current, equity=100_000)
        assert result["submitted"] is True
        assert len(result["orders"]) == 2
        assert loop.trading_client.submit_order.call_count == 2

    def test_rebalance_skips_dust_trades(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.trading_client.submit_order = MagicMock(return_value=MagicMock(id="order-123"))

        target = pd.Series({"A": 0.001, "B": 0.001})  # tiny weights -> tiny notional
        current = pd.Series({"A": 0.0, "B": 0.0})
        result = loop.rebalance(target, current, equity=1000, min_trade_notional=50.0)
        assert result["submitted"] is True
        assert len(result["orders"]) == 0  # both trades below min_trade_notional

    def test_order_rejection_handled_gracefully(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.trading_client.submit_order = MagicMock(side_effect=Exception("broker rejected: insufficient funds"))

        target = pd.Series({"A": 0.3, "B": 0.3})
        current = pd.Series({"A": 0.0, "B": 0.0})
        result = loop.rebalance(target, current, equity=100_000)
        # Should not raise -- failures are caught, logged to audit, and order marked rejected
        assert result["submitted"] is True
        assert len(result["orders"]) == 0  # both orders failed to submit

    def test_audit_log_records_every_stage(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.trading_client.submit_order = MagicMock(return_value=MagicMock(id="order-123"))

        target = pd.Series({"A": 0.3, "B": 0.3})
        current = pd.Series({"A": 0.0, "B": 0.0})
        loop.rebalance(target, current, equity=100_000)

        event_types = [e.event_type for e in loop.audit_log.entries]
        assert "rebalance_risk_check" in event_types
        assert "order_submitted" in event_types
        assert "order_acknowledged" in event_types

        valid, _ = loop.audit_log.verify_chain()
        assert valid


class TestBrokerIsSourceOfTruth:
    def test_get_current_weights_reads_from_broker(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        mock_account = MagicMock(equity="100000.0")
        mock_position_a = MagicMock(symbol="A", market_value="30000.0")
        loop.trading_client.get_account = MagicMock(return_value=mock_account)
        loop.trading_client.get_all_positions = MagicMock(return_value=[mock_position_a])

        weights, equity = loop.get_current_weights()
        assert equity == 100_000.0
        assert abs(weights["A"] - 0.3) < 1e-9
        assert weights["B"] == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

class TestOperationalSafety:
    def test_emergency_stop_blocks_cycle(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        monkeypatch.setenv("LIVE_EMERGENCY_STOP", "1")
        result = loop.run_once()
        assert result["submitted"] is False
        assert result["reason"] == "emergency_stop_active"

    def test_deterministic_order_id_is_stable(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        first = loop._intent_client_order_id("A", "buy", 1000, 0.1, 0.0)
        second = loop._intent_client_order_id("A", "buy", 1000, 0.1, 0.0)
        assert first == second
        assert len(first) <= 48

    def test_reconciliation_fails_closed_on_position_mismatch(self, tmp_path):
        loop = _make_loop(tmp_path, symbols=("A", "B"))
        loop.trading_client.get_all_positions = MagicMock(
            return_value=[MagicMock(symbol="A", qty="10")]
        )
        with pytest.raises(RuntimeError, match="BROKER_RECONCILIATION_FAILED"):
            loop._reconcile_broker_state()
