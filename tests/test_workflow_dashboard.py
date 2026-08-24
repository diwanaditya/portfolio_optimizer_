import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.estimators.expected_returns import mean_historical_return
from portfolio_optimizer.estimators.covariance import ledoit_wolf_shrinkage
from portfolio_optimizer.optimizers.markowitz import MarkowitzOptimizer
from portfolio_optimizer.dashboard.workflow_data import (
    build_health_metrics, build_efficient_frontier_data, build_allocation_treemap_data,
    build_rolling_metrics_data, build_factor_exposure_data, build_risk_contribution_data,
    build_weight_change_explanations, build_full_workflow_dashboard_data,
)
from portfolio_optimizer.dashboard.workflow_ui import (
    render_workflow_dashboard, generate_workflow_dashboard, _metric_color,
    COLOR_INK, COLOR_GREEN, COLOR_RED,
)


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=500)


@pytest.fixture(scope="module")
def optimized(returns):
    mu = mean_historical_return(returns)
    cov, _ = ledoit_wolf_shrinkage(returns)
    res = MarkowitzOptimizer(mu, cov, risk_free_rate=0.03, weight_bounds=(0.0, 0.30)).max_sharpe()
    return mu, cov, res.weights


class TestHealthMetrics:
    def test_returns_all_six_metrics(self, returns, optimized):
        mu, cov, weights = optimized
        port_returns = returns @ weights
        health = build_health_metrics(port_returns, weights, returns)
        assert set(health.keys()) == {"expected_return", "volatility", "sharpe_ratio",
                                        "max_drawdown", "var_95", "cvar_95"}

    def test_max_drawdown_is_non_positive(self, returns, optimized):
        mu, cov, weights = optimized
        port_returns = returns @ weights
        health = build_health_metrics(port_returns, weights, returns)
        assert health["max_drawdown"] <= 0

    def test_cvar_at_least_as_extreme_as_var(self, returns, optimized):
        mu, cov, weights = optimized
        port_returns = returns @ weights
        health = build_health_metrics(port_returns, weights, returns, alpha=0.95)
        assert health["cvar_95"] >= health["var_95"] - 1e-9


class TestEfficientFrontierData:
    def test_frontier_has_requested_points(self, optimized):
        mu, cov, weights = optimized
        data = build_efficient_frontier_data(mu, cov, weights, n_points=15)
        assert len(data["frontier"]) <= 15
        assert len(data["frontier"]) > 5

    def test_tangency_point_present(self, optimized):
        mu, cov, weights = optimized
        data = build_efficient_frontier_data(mu, cov, weights)
        assert "risk" in data["tangency"] and "return" in data["tangency"]

    def test_asset_points_cover_all_assets(self, optimized):
        mu, cov, weights = optimized
        data = build_efficient_frontier_data(mu, cov, weights)
        assert len(data["assets"]) == len(mu)

    def test_current_portfolio_point_present_when_weights_given(self, optimized):
        mu, cov, weights = optimized
        data = build_efficient_frontier_data(mu, cov, weights)
        assert data["current"] is not None
        assert data["current"]["risk"] > 0

    def test_current_point_none_without_weights(self, optimized):
        mu, cov, weights = optimized
        data = build_efficient_frontier_data(mu, cov, None)
        assert data["current"] is None


class TestTreemapData:
    def test_excludes_near_zero_weights(self):
        weights = pd.Series({"A": 0.5, "B": 0.5, "C": 1e-9})
        cells = build_allocation_treemap_data(weights)
        names = {c["name"] for c in cells}
        assert "C" not in names
        assert "A" in names and "B" in names

    def test_sorted_by_weight_descending(self):
        weights = pd.Series({"A": 0.2, "B": 0.5, "C": 0.3})
        cells = build_allocation_treemap_data(weights)
        weights_list = [c["weight"] for c in cells]
        assert weights_list == sorted(weights_list, reverse=True)

    def test_expected_return_attached_when_provided(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        er = pd.Series({"A": 0.10, "B": -0.05})
        cells = build_allocation_treemap_data(weights, er)
        by_name = {c["name"]: c["expected_return"] for c in cells}
        assert by_name["A"] == 0.10
        assert by_name["B"] == -0.05

    def test_defaults_to_zero_return_when_not_provided(self):
        weights = pd.Series({"A": 1.0})
        cells = build_allocation_treemap_data(weights)
        assert cells[0]["expected_return"] == 0.0


class TestRollingMetricsData:
    def test_produces_both_series(self, returns, optimized):
        mu, cov, weights = optimized
        port_returns = returns @ weights
        data = build_rolling_metrics_data(port_returns, window=30)
        assert len(data["rolling_sharpe"]) > 0
        assert len(data["rolling_drawdown"]) > 0

    def test_drawdown_series_never_positive(self, returns, optimized):
        mu, cov, weights = optimized
        port_returns = returns @ weights
        data = build_rolling_metrics_data(port_returns, window=30)
        assert all(p["v"] <= 1e-9 for p in data["rolling_drawdown"])


class TestFactorExposureData:
    def test_returns_requested_number_of_factors(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_factor_exposure_data(weights, returns, n_factors=3)
        assert len(data["factors"]) == 3

    def test_r_squared_present_per_asset(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_factor_exposure_data(weights, returns, n_factors=2)
        assert set(data["r_squared"].keys()) == set(returns.columns)


class TestRiskContributionData:
    def test_contributions_sum_to_one(self, optimized):
        mu, cov, weights = optimized
        contributions = build_risk_contribution_data(weights, cov)
        total = sum(c["contribution_pct"] for c in contributions)
        assert abs(total - 1.0) < 1e-6

    def test_covers_all_assets(self, optimized):
        mu, cov, weights = optimized
        contributions = build_risk_contribution_data(weights, cov)
        assert len(contributions) == len(weights)


class TestWeightChangeExplanationsIntegration:
    def test_produces_narratives(self, returns, optimized):
        mu, cov, weights = optimized
        prev_weights = pd.Series(1.0 / len(weights), index=weights.index)
        explanations = build_weight_change_explanations(returns, returns, prev_weights, weights, top_n=3)
        assert len(explanations) == 3
        for e in explanations:
            assert "narrative" in e and len(e["narrative"]) > 0


class TestFullPipeline:
    def test_build_full_dashboard_data_structure(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        assert set(data.keys()) >= {"health", "efficient_frontier", "treemap", "rolling",
                                      "factor_exposure", "risk_contribution", "weight_changes"}

    def test_full_pipeline_with_previous_weights_populates_changes(self, returns, optimized):
        mu, cov, weights = optimized
        prev_weights = pd.Series(1.0 / len(weights), index=weights.index)
        data = build_full_workflow_dashboard_data(
            returns, weights, mu, cov, previous_weights=prev_weights, previous_returns=returns,
        )
        assert len(data["weight_changes"]) > 0

    def test_full_pipeline_without_previous_weights_empty_changes(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        assert data["weight_changes"] == []


class TestHealthCardColorLogic:
    def test_positive_sharpe_is_green(self):
        assert _metric_color("sharpe_ratio", 1.5) == COLOR_GREEN

    def test_negative_sharpe_is_red(self):
        assert _metric_color("sharpe_ratio", -0.5) == COLOR_RED

    def test_drawdown_is_red_when_nonzero(self):
        assert _metric_color("max_drawdown", -0.15) == COLOR_RED

    def test_drawdown_is_black_when_zero(self):
        assert _metric_color("max_drawdown", 0.0) == COLOR_INK

    def test_volatility_is_always_neutral_black(self):
        """Volatility/VaR/CVaR are magnitude-only metrics -- neither
        'good' nor 'bad' in isolation, so they must never be colored
        green or red regardless of value, unlike Sharpe/return/drawdown.
        """
        assert _metric_color("volatility", 0.30) == COLOR_INK
        assert _metric_color("var_95", 0.05) == COLOR_INK
        assert _metric_color("cvar_95", 0.08) == COLOR_INK

    def test_nan_is_black(self):
        assert _metric_color("sharpe_ratio", float("nan")) == COLOR_INK


class TestHTMLRendering:
    def test_render_produces_valid_html(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        html = render_workflow_dashboard(data)
        assert "<html>" in html.lower()
        assert "Chart.js" in html or "chart.umd" in html

    def test_all_five_workflow_cards_present(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        html = render_workflow_dashboard(data)
        for label in ["Create Portfolio", "Optimize", "Backtest", "Paper Trade", "Generate Report"]:
            assert label in html

    def test_generate_writes_file(self, returns, optimized, tmp_path):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        out_path = str(tmp_path / "dashboard.html")
        result_path = generate_workflow_dashboard(data, out_path)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 5000

    def test_explanations_rendered_when_present(self, returns, optimized):
        mu, cov, weights = optimized
        prev_weights = pd.Series(1.0 / len(weights), index=weights.index)
        data = build_full_workflow_dashboard_data(
            returns, weights, mu, cov, previous_weights=prev_weights, previous_returns=returns,
        )
        html = render_workflow_dashboard(data)
        assert "explain-row" in html

    def test_empty_state_shown_without_explanations(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        html = render_workflow_dashboard(data)
        assert "No prior allocation supplied" in html

    def test_workflow_cards_use_custom_svg_icons_not_unicode_symbols(self, returns, optimized):
        """Regression test for the design pass: workflow cards must use
        custom-drawn SVG icon shapes, not unicode/emoji glyphs (the
        original version used characters like the ones checked for
        absence below).
        """
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        html = render_workflow_dashboard(data)
        assert html.count("workflow-icon") >= 5
        assert "<svg" in html
        # the old unicode glyph set this replaced
        for glyph in ["◎", "⚙", "↻", "▶", "▤"]:
            assert glyph not in html

    def test_typography_system_loaded(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        html = render_workflow_dashboard(data)
        assert "fonts.googleapis.com" in html
        assert "Inter" in html
        assert "JetBrains+Mono" in html

    def test_spacing_scale_variables_defined(self, returns, optimized):
        mu, cov, weights = optimized
        data = build_full_workflow_dashboard_data(returns, weights, mu, cov)
        html = render_workflow_dashboard(data)
        for var in ["--sp-1", "--sp-3", "--sp-5", "--sp-7"]:
            assert var in html


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
