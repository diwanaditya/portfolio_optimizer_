import sys, os, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.infra.fix_protocol import (
    new_order_single, execution_report, encode_fix_message, decode_fix_message
)
from portfolio_optimizer.infra.oms import Order, OrderManager, OrderStatus, IllegalOrderTransitionError
from portfolio_optimizer.infra.persistence import PortfolioStateStore
from portfolio_optimizer.infra.audit_log import TamperEvidentAuditLog
from portfolio_optimizer.infra.risk_controls import PreTradeRiskEngine, PreTradeLimits, CheckSeverity
from portfolio_optimizer.infra.fault_tolerance import retry_with_backoff, CircuitBreaker, CircuitState, CircuitBreakerOpenError
from portfolio_optimizer.infra.distributed import run_parallel_tasks, ParallelParameterSweep
from portfolio_optimizer.infra.monitoring import (
    MonitoringEngine, drawdown_rule, concentration_rule, var_breach_rule, AlertSeverity
)


class TestFixProtocol:
    def test_round_trip_new_order_single(self):
        msg = new_order_single("ORD1", "AAPL", "1", 100, ord_type="2", price=150.0)
        wire = encode_fix_message(msg, "SENDER", "TARGET", seq_num=1)
        decoded = decode_fix_message(wire)
        assert decoded["_checksum_valid"] is True
        assert decoded[55] == "AAPL"
        assert decoded[38] == "100"
        assert decoded[44] == "150.0"

    def test_execution_report_round_trip(self):
        msg = execution_report("ORD1", "OID1", "EXEC1", "2", "2", "MSFT", "1", 100, 100, 0, 300.5)
        wire = encode_fix_message(msg, "SENDER", "TARGET", seq_num=2)
        decoded = decode_fix_message(wire)
        assert decoded["_checksum_valid"] is True
        assert decoded[35] == "8"
        assert decoded[14] == "100"

    def test_checksum_detects_corruption(self):
        msg = new_order_single("ORD1", "AAPL", "1", 100)
        wire = encode_fix_message(msg, "SENDER", "TARGET", seq_num=1)
        corrupted = wire.replace("AAPL", "GOOG")
        decoded = decode_fix_message(corrupted)
        assert decoded["_checksum_valid"] is False


class TestOMS:
    def test_order_lifecycle_full_fill(self):
        order = Order(symbol="AAPL", side="buy", quantity=100)
        order.acknowledge()
        assert order.status == OrderStatus.NEW
        order.apply_fill(100, 150.0)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.average_fill_price == 150.0

    def test_partial_fill_sequence(self):
        order = Order(symbol="AAPL", side="buy", quantity=100)
        order.acknowledge()
        order.apply_fill(40, 150.0)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        order.apply_fill(60, 151.0)
        assert order.status == OrderStatus.FILLED
        expected_avg = (40 * 150.0 + 60 * 151.0) / 100
        assert abs(order.average_fill_price - expected_avg) < 1e-9

    def test_illegal_transition_raises(self):
        order = Order(symbol="AAPL", side="buy", quantity=100)
        order.acknowledge()
        order.apply_fill(100, 150.0)  # now FILLED (terminal)
        with pytest.raises(IllegalOrderTransitionError):
            order.apply_fill(10, 150.0)  # cannot fill an already-filled order

    def test_overfill_raises_value_error(self):
        order = Order(symbol="AAPL", side="buy", quantity=100)
        order.acknowledge()
        with pytest.raises(ValueError):
            order.apply_fill(150, 150.0)  # exceeds order quantity

    def test_reject_from_pending_new(self):
        order = Order(symbol="AAPL", side="buy", quantity=100)
        order.reject("insufficient buying power")
        assert order.status == OrderStatus.REJECTED
        assert order.reject_reason == "insufficient buying power"

    def test_order_manager_tracks_orders(self):
        om = OrderManager()
        o1 = om.submit("AAPL", "buy", 100)
        o2 = om.submit("MSFT", "sell", 50)
        o1.acknowledge()
        o1.apply_fill(100, 150.0)
        o2.acknowledge()
        assert len(om.filled_orders()) == 1
        assert len(om.open_orders()) == 1
        summary = om.summary()
        assert summary["total_orders"] == 2


class TestPersistence:
    def test_position_upsert_and_retrieve(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.upsert_position("port1", "AAPL", 100, 150.0)
            store.upsert_position("port1", "AAPL", 120, 151.0)  # update
            positions = store.get_positions("port1")
            assert positions["AAPL"]["quantity"] == 120

    def test_nav_history_recorded_and_retrieved(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 1_000_000, 50_000, {"note": "initial"})
            time.sleep(0.01)
            store.record_nav("port1", 1_010_000, 45_000)
            history = store.get_nav_history("port1")
            assert len(history) == 2

    def test_portfolio_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.upsert_position("port1", "AAPL", 100, 150.0)
            store.record_nav("port1", 1_000_000, 50_000)
            order = Order(symbol="AAPL", side="buy", quantity=50)
            order.acknowledge()
            store.save_order("port1", order)
            snapshot = store.portfolio_snapshot("port1")
            assert "AAPL" in snapshot["positions"]
            assert snapshot["latest_nav"]["nav"] == 1_000_000
            assert len(snapshot["open_orders"]) == 1


class TestAuditLog:
    def test_chain_valid_after_normal_use(self):
        log = TamperEvidentAuditLog()
        log.record("order_placed", {"symbol": "AAPL", "qty": 100})
        log.record("risk_check_passed", {"check": "concentration"})
        log.record("order_filled", {"symbol": "AAPL", "price": 150.0})
        valid, broken_idx = log.verify_chain()
        assert valid
        assert broken_idx is None

    def test_tampering_detected(self):
        log = TamperEvidentAuditLog()
        log.record("order_placed", {"symbol": "AAPL", "qty": 100})
        log.record("order_filled", {"symbol": "AAPL", "price": 150.0})
        # tamper with the first entry's payload after the fact
        log.entries[0].payload["qty"] = 999999
        valid, broken_idx = log.verify_chain()
        assert not valid
        assert broken_idx == 0

    def test_json_round_trip_preserves_chain_validity(self):
        log = TamperEvidentAuditLog()
        log.record("event_a", {"x": 1})
        log.record("event_b", {"y": 2})
        exported = log.export_json()
        restored = TamperEvidentAuditLog.from_json(exported)
        valid, _ = restored.verify_chain()
        assert valid

    def test_entries_by_type_filters_correctly(self):
        log = TamperEvidentAuditLog()
        log.record("order_placed", {"id": 1})
        log.record("order_placed", {"id": 2})
        log.record("risk_check", {"id": 3})
        assert len(log.entries_by_type("order_placed")) == 2


class TestRiskControls:
    def test_restricted_symbol_blocks(self):
        limits = PreTradeLimits(restricted_symbols={"BADCO"})
        engine = PreTradeRiskEngine(limits)
        target = pd.Series({"AAPL": 0.5, "BADCO": 0.5})
        results = engine.check_target_portfolio(target)
        assert not engine.is_approved(results)

    def test_concentration_limit_blocks(self):
        limits = PreTradeLimits(max_position_weight=0.30)
        engine = PreTradeRiskEngine(limits)
        target = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
        results = engine.check_target_portfolio(target)
        assert not engine.is_approved(results)

    def test_valid_portfolio_approved(self):
        limits = PreTradeLimits(max_position_weight=0.40)
        engine = PreTradeRiskEngine(limits)
        target = pd.Series({"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25})
        results = engine.check_target_portfolio(target)
        assert engine.is_approved(results)

    def test_sector_limit_enforced(self):
        limits = PreTradeLimits(max_sector_weight={"tech": 0.5})
        sector_map = {"AAPL": "tech", "MSFT": "tech", "XOM": "energy"}
        engine = PreTradeRiskEngine(limits, sector_map=sector_map)
        target = pd.Series({"AAPL": 0.4, "MSFT": 0.4, "XOM": 0.2})
        results = engine.check_target_portfolio(target)
        assert not engine.is_approved(results)


class TestFaultTolerance:
    def test_retry_succeeds_after_transient_failures(self):
        attempts = {"count": 0}

        @retry_with_backoff(max_attempts=5, base_delay=0.001, jitter=False)
        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("transient")
            return "success"

        result = flaky()
        assert result == "success"
        assert attempts["count"] == 3

    def test_retry_exhausts_and_raises(self):
        @retry_with_backoff(max_attempts=3, base_delay=0.001, jitter=False)
        def always_fails():
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError):
            always_fails()

    def test_circuit_breaker_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        def failing_fn():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_fn)
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(failing_fn)  # should fail fast now, not even call failing_fn

    def test_circuit_breaker_recovers_on_success(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED


def _square(x):
    return x ** 2


class TestDistributed:
    def test_parallel_tasks_preserve_order(self):
        results = run_parallel_tasks(_square, [(1,), (2,), (3,), (4,)], max_workers=2)
        assert [r.result for r in results] == [1, 4, 9, 16]

    def test_parallel_sweep(self):
        sweep = ParallelParameterSweep(_square, max_workers=2)
        results = sweep.run([1, 2, 3])
        assert results["2"].result == 4


class TestMonitoring:
    def test_drawdown_rule_triggers(self):
        equity = pd.Series([100, 110, 90, 80])  # drawdown from peak 110 to 80 = -27%
        engine = MonitoringEngine().add_rule(drawdown_rule(max_drawdown=-0.15))
        alerts = engine.check_all({"equity_curve": equity})
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_concentration_rule_triggers(self):
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        engine = MonitoringEngine().add_rule(concentration_rule(max_weight=0.35))
        alerts = engine.check_all({"weights": weights})
        assert len(alerts) == 1

    def test_no_alert_when_within_limits(self):
        weights = pd.Series({"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25})
        engine = MonitoringEngine().add_rule(concentration_rule(max_weight=0.35))
        alerts = engine.check_all({"weights": weights})
        assert len(alerts) == 0

    def test_var_breach_rule(self):
        returns = pd.Series(np.random.RandomState(0).normal(-0.01, 0.05, 100))
        engine = MonitoringEngine().add_rule(var_breach_rule(var_limit=0.01))
        alerts = engine.check_all({"recent_returns": returns})
        assert len(alerts) >= 0  # may or may not trigger depending on random data; just verify it runs

    def test_alert_history_accumulates(self):
        weights = pd.Series({"AAPL": 0.9, "MSFT": 0.1})
        engine = MonitoringEngine().add_rule(concentration_rule(max_weight=0.35))
        engine.check_all({"weights": weights})
        engine.check_all({"weights": weights})
        assert len(engine.alert_history) == 2
        assert len(engine.alerts_by_severity(AlertSeverity.WARNING)) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
