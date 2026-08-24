import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from portfolio_optimizer.dashboard.generator import (
    build_dashboard_data, render_dashboard_html, generate_dashboard,
    _pnl_color, _fmt_money, _fmt_pct, COLOR_GREEN, COLOR_RED, COLOR_BLACK, COLOR_WHITE,
)
from portfolio_optimizer.infra.persistence import PortfolioStateStore
from portfolio_optimizer.infra.oms import Order


class TestColorLogic:
    def test_positive_value_is_green(self):
        assert _pnl_color(1234.56) == COLOR_GREEN

    def test_negative_value_is_red(self):
        assert _pnl_color(-1234.56) == COLOR_RED

    def test_zero_is_black_not_green_or_red(self):
        assert _pnl_color(0.0) == COLOR_BLACK

    def test_tiny_epsilon_values_treated_as_zero(self):
        # values within floating point noise of zero should stay black
        assert _pnl_color(1e-12) == COLOR_BLACK
        assert _pnl_color(-1e-12) == COLOR_BLACK


class TestFormatting:
    def test_money_format_positive(self):
        assert _fmt_money(1234.5) == "+$1,234.50"

    def test_money_format_negative(self):
        assert _fmt_money(-1234.5) == "-$1,234.50"

    def test_money_format_zero(self):
        assert _fmt_money(0.0) == "$0.00"

    def test_pct_format_positive(self):
        assert _fmt_pct(0.1234) == "+12.34%"

    def test_pct_format_negative(self):
        assert _fmt_pct(-0.1234) == "-12.34%"


class TestDashboardDataBuilding:
    def test_raises_without_nav_history(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "empty.db"))
            with pytest.raises(ValueError, match="No NAV history"):
                build_dashboard_data(store, "nonexistent_portfolio")

    def test_computes_correct_pnl(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 100_000, 0.0)
            import time; time.sleep(0.01)
            store.record_nav("port1", 110_000, 0.0)

            data = build_dashboard_data(store, "port1")
            assert data["starting_nav"] == 100_000
            assert data["current_nav"] == 110_000
            assert abs(data["total_pnl"] - 10_000) < 1e-6
            assert abs(data["total_pnl_pct"] - 0.10) < 1e-6

    def test_negative_pnl_computed_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 100_000, 0.0)
            import time; time.sleep(0.01)
            store.record_nav("port1", 90_000, 0.0)

            data = build_dashboard_data(store, "port1")
            assert data["total_pnl"] < 0
            assert data["total_pnl_pct"] < 0

    def test_drawdown_computed_from_running_max(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            import time
            for nav in [100_000, 120_000, 90_000]:  # peak 120k, then drawdown to 90k
                store.record_nav("port1", nav, 0.0)
                time.sleep(0.01)

            data = build_dashboard_data(store, "port1")
            expected_dd = 90_000 / 120_000 - 1
            assert abs(data["current_drawdown"] - expected_dd) < 1e-6
            assert data["max_drawdown"] <= data["current_drawdown"] + 1e-9

    def test_positions_included(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 100_000, 0.0)
            store.upsert_position("port1", "AAPL", 100, 150.0)
            data = build_dashboard_data(store, "port1")
            assert len(data["positions"]) == 1
            assert data["positions"][0]["symbol"] == "AAPL"

    def test_open_vs_filled_orders_separated(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 100_000, 0.0)

            o1 = Order(symbol="AAPL", side="buy", quantity=100)
            o1.acknowledge()
            o1.apply_fill(100, 150.0)
            store.save_order("port1", o1)

            o2 = Order(symbol="MSFT", side="sell", quantity=50)
            o2.acknowledge()
            store.save_order("port1", o2)  # stays open

            data = build_dashboard_data(store, "port1")
            assert data["filled_orders_count"] == 1
            assert len(data["open_orders"]) == 1
            assert data["total_orders"] == 2


class TestHTMLRendering:
    def test_html_contains_required_color_scheme(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 100_000, 0.0)
            import time; time.sleep(0.01)
            store.record_nav("port1", 110_000, 0.0)
            data = build_dashboard_data(store, "port1")
            html = render_dashboard_html(data)

            assert COLOR_WHITE in html
            assert COLOR_GREEN in html  # positive P&L should trigger green
            assert "<html>" in html.lower()
            assert "NAV History" in html

    def test_html_uses_red_for_losses(self):
        with tempfile.TemporaryDirectory() as d:
            store = PortfolioStateStore(os.path.join(d, "test.db"))
            store.record_nav("port1", 100_000, 0.0)
            import time; time.sleep(0.01)
            store.record_nav("port1", 90_000, 0.0)
            data = build_dashboard_data(store, "port1")
            html = render_dashboard_html(data)
            assert COLOR_RED in html

    def test_generate_dashboard_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            store = PortfolioStateStore(db_path)
            store.record_nav("port1", 100_000, 0.0)
            store.upsert_position("port1", "AAPL", 100, 150.0)

            out_path = os.path.join(d, "dashboard.html")
            result_path = generate_dashboard(db_path, "port1", out_path)
            assert os.path.exists(result_path)
            with open(result_path) as f:
                content = f.read()
            assert "AAPL" in content
            assert len(content) > 1000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
