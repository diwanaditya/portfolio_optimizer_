import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.benchmarking.benchmark_suite import (
    benchmark_max_sharpe, benchmark_min_cvar, benchmark_hrp, summarize, robustness_battery
)


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=300)


class TestBenchmarkSuite:
    def test_max_sharpe_all_three_converge(self, returns):
        results = benchmark_max_sharpe(returns, n_repeats=1)
        libraries = {r.library for r in results}
        assert libraries == {"this_library", "pypfopt", "riskfolio-lib"}
        for r in results:
            assert r.converged, f"{r.library} failed to converge: {r.error}"

    def test_min_cvar_objective_values_agree_across_libraries(self, returns):
        """The key correctness check: since CVaR uses the same scenario
        data with no estimator-choice ambiguity, all three solvers should
        converge to the same objective value if they're all correctly
        implementing the Rockafellar-Uryasev formulation.
        """
        results = benchmark_min_cvar(returns, alpha=0.95, n_repeats=1)
        objectives = {r.library: r.objective_value for r in results if r.converged}
        assert len(objectives) == 3
        values = list(objectives.values())
        # all three should agree to within a small tolerance
        assert max(values) - min(values) < 0.001, f"CVaR objectives disagree: {objectives}"

    def test_hrp_all_three_produce_valid_weights(self, returns):
        results = benchmark_hrp(returns, n_repeats=1)
        for r in results:
            assert r.converged
            assert np.isclose(r.weights.sum(), 1.0, atol=1e-3)

    def test_robustness_battery_this_library_never_fails(self):
        """This library's ridge regularization is specifically designed to
        survive these pathological cases -- verify that design goal holds.
        """
        df = robustness_battery()
        this_lib_rows = df[df["library"] == "this_library"]
        assert this_lib_rows["converged"].all(), \
            f"this_library failed on: {this_lib_rows[~this_lib_rows['converged']]['case'].tolist()}"

    def test_summarize_produces_valid_dataframe(self, returns):
        results = benchmark_max_sharpe(returns, n_repeats=1)
        table = summarize(results)
        assert set(table.columns) >= {"library", "method", "runtime_ms", "converged"}
        assert len(table) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
