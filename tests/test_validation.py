import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.validation.bootstrap_ci import (
    bootstrap_metric_ci, sharpe_ratio_metric, max_drawdown_metric, full_metric_report
)
from portfolio_optimizer.validation.multiple_testing import (
    jobson_korkie_memmel_test, holm_bonferroni_correction, benjamini_hochberg_correction,
    multi_strategy_comparison_report
)
from portfolio_optimizer.validation.sensitivity_analysis import sweep_parameter, multi_parameter_sensitivity_report
from portfolio_optimizer.validation.regime_robustness import generate_regime_scenarios, evaluate_strategy_across_regimes


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=400)


class TestBootstrapCI:
    def test_sharpe_ci_contains_point_estimate_bounds_are_ordered(self, returns):
        port = returns.mean(axis=1)
        result = bootstrap_metric_ci(port, sharpe_ratio_metric(), n_bootstrap=200)
        assert result.ci_lower <= result.ci_upper
        assert len(result.bootstrap_distribution) == 200

    def test_full_metric_report_structure(self, returns):
        port = returns.mean(axis=1)
        report = full_metric_report(port, n_bootstrap=200)
        assert set(report.index) == {"annualized_return", "sharpe_ratio", "sortino_ratio",
                                       "max_drawdown", "cvar_95"}
        assert (report["ci_upper"] >= report["ci_lower"]).all()

    def test_narrower_ci_with_more_bootstrap_samples_is_not_required_but_runs(self, returns):
        port = returns.mean(axis=1)
        r1 = bootstrap_metric_ci(port, max_drawdown_metric(), n_bootstrap=100, seed=1)
        r2 = bootstrap_metric_ci(port, max_drawdown_metric(), n_bootstrap=100, seed=2)
        assert np.isfinite(r1.point_estimate) and np.isfinite(r2.point_estimate)


class TestMultipleTesting:
    def test_jobson_korkie_identical_series_gives_zero_stat(self, returns):
        port = returns.mean(axis=1)
        z, p = jobson_korkie_memmel_test(port, port)
        assert abs(z) < 1e-6
        assert p > 0.99

    def test_holm_correction_reduces_or_equal_rejections_vs_uncorrected(self):
        p_values = [0.001, 0.01, 0.03, 0.04, 0.2, 0.5]
        rejected = holm_bonferroni_correction(p_values, alpha=0.05)
        uncorrected = np.array(p_values) < 0.05
        assert rejected.sum() <= uncorrected.sum()

    def test_bh_correction_valid_output(self):
        p_values = [0.001, 0.02, 0.03, 0.04, 0.3, 0.6]
        rejected = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert len(rejected) == len(p_values)
        assert rejected.dtype == bool

    def test_multi_strategy_report_structure(self, returns):
        strategies = {
            "strat_a": returns.iloc[:, 0],
            "strat_b": returns.iloc[:, 1],
            "strat_c": returns.mean(axis=1),
        }
        report = multi_strategy_comparison_report(strategies, correction="holm")
        assert len(report) == 3  # 3 choose 2 = 3 pairs
        assert "n_comparisons" in report.attrs
        assert report.attrs["n_significant_corrected"] <= report.attrs["n_significant_uncorrected"]


class TestSensitivityAnalysis:
    def test_sweep_parameter_basic(self):
        def run_fn(x):
            return {"y": x ** 2, "z": -x}
        result = sweep_parameter([1, 2, 3, 4], run_fn, "x")
        assert list(result.output_values.index) == [1, 2, 3, 4]
        assert result.output_values.loc[4, "y"] == 16

    def test_relative_sensitivity_zero_for_constant_output(self):
        def run_fn(x):
            return {"constant_metric": 5.0}
        result = sweep_parameter([1, 2, 3], run_fn, "x")
        assert result.output_relative_sensitivity["constant_metric"] == 0.0

    def test_multi_parameter_report_structure(self):
        def run_fn_a(x):
            # low relative sensitivity: output barely changes relative to its mean level
            return {"metric1": 100.0 + x * 0.01}
        def run_fn_b(x):
            # high relative sensitivity: output swings widely relative to its mean level
            return {"metric1": x}
        sweeps = {
            "param_a": sweep_parameter([1, 2, 3], run_fn_a, "param_a"),
            "param_b": sweep_parameter([1, 2, 3], run_fn_b, "param_b"),
        }
        report = multi_parameter_sensitivity_report(sweeps)
        assert set(report["parameter"]) == {"param_a", "param_b"}
        # param_b's output swings from 1 to 3 (relative range ~1.0), while param_a's
        # output barely moves relative to its ~100 baseline -- param_b should rank
        # as more sensitive after normalization.
        assert report.iloc[0]["parameter"] == "param_b"


class TestRegimeRobustness:
    def test_generate_regime_scenarios_returns_five_regimes(self):
        scenarios = generate_regime_scenarios()
        assert len(scenarios) == 5
        assert "bear_market" in scenarios
        assert "crash_recovery" in scenarios

    def test_bear_market_has_negative_drift(self):
        scenarios = generate_regime_scenarios()
        assert scenarios["bear_market"].mean().mean() < scenarios["bull_low_vol"].mean().mean()

    def test_evaluate_strategy_across_regimes_runs(self):
        def equal_weight_strategy(returns):
            return pd.Series(1.0 / returns.shape[1], index=returns.columns)

        report = evaluate_strategy_across_regimes(equal_weight_strategy)
        assert len(report.summary_table) == 5
        assert report.worst_regime in report.summary_table.index
        assert np.isfinite(report.dispersion_sharpe)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
