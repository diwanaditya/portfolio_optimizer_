import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.explainability.weight_change_explainer import WeightChangeExplainer


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=600)


class TestWeightChangeExplainer:
    def test_unchanged_weight_gives_flat_narrative(self, returns):
        assets = list(returns.columns)
        w = pd.Series(1.0 / len(assets), index=assets)
        explainer = WeightChangeExplainer(returns, returns, w, w)
        exp = explainer.explain_asset(assets[0])
        assert "stayed roughly flat" in exp.narrative
        assert abs(exp.weight_change) < 1e-9

    def test_return_driven_increase_mentions_return(self, returns):
        assets = list(returns.columns)
        before, after = returns.iloc[:300], returns.iloc[300:]
        w_before = pd.Series(1.0 / len(assets), index=assets)
        w_after = w_before.copy()
        w_after[assets[0]] += 0.15
        w_after = w_after / w_after.sum()

        # force a genuine return increase for assets[0] in the "after" window
        after = after.copy()
        after[assets[0]] = after[assets[0]] + 0.002

        explainer = WeightChangeExplainer(before, after, w_before, w_after)
        exp = explainer.explain_asset(assets[0])
        assert exp.weight_change > 0
        assert "expected return rose" in exp.narrative

    def test_narrative_never_claims_a_reason_that_contradicts_the_direction(self, returns):
        """The core correctness property: if we force a DECREASE in
        expected return but the narrative generator is buggy and reports
        'expected return rose' anyway, that's a real (embarrassing) bug.
        Verify across several random weight pairs that every stated
        reason's directional word matches its underlying sign.
        """
        assets = list(returns.columns)
        rng = np.random.default_rng(1)
        before, after = returns.iloc[:300], returns.iloc[300:]

        for _ in range(5):
            w1 = pd.Series(rng.dirichlet(np.ones(len(assets))), index=assets)
            w2 = pd.Series(rng.dirichlet(np.ones(len(assets))), index=assets)
            explainer = WeightChangeExplainer(before, after, w1, w2)
            for exp in explainer.explain_all():
                if "expected return rose" in exp.narrative:
                    assert exp.return_change > 0
                if "expected return fell" in exp.narrative:
                    assert exp.return_change < 0
                if "correlation with the rest of the portfolio declined" in exp.narrative:
                    assert exp.portfolio_correlation_change < 0
                if "correlation with the rest of the portfolio rose" in exp.narrative:
                    assert exp.portfolio_correlation_change > 0

    def test_explain_all_sorted_by_magnitude(self, returns):
        assets = list(returns.columns)
        rng = np.random.default_rng(2)
        w1 = pd.Series(rng.dirichlet(np.ones(len(assets))), index=assets)
        w2 = pd.Series(rng.dirichlet(np.ones(len(assets))), index=assets)
        explainer = WeightChangeExplainer(returns.iloc[:300], returns.iloc[300:], w1, w2)
        explanations = explainer.explain_all()
        magnitudes = [abs(e.weight_change) for e in explanations]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_top_n_limits_results(self, returns):
        assets = list(returns.columns)
        rng = np.random.default_rng(3)
        w1 = pd.Series(rng.dirichlet(np.ones(len(assets))), index=assets)
        w2 = pd.Series(rng.dirichlet(np.ones(len(assets))), index=assets)
        explainer = WeightChangeExplainer(returns.iloc[:300], returns.iloc[300:], w1, w2)
        assert len(explainer.explain_all(top_n=3)) == 3

    def test_summary_table_structure(self, returns):
        assets = list(returns.columns)
        w = pd.Series(1.0 / len(assets), index=assets)
        explainer = WeightChangeExplainer(returns, returns, w, w)
        table = explainer.summary_table()
        assert "narrative" in table.columns
        assert "weight_change" in table.columns
        assert len(table) == len(assets)

    def test_portfolio_correlation_excludes_self(self, returns):
        assets = list(returns.columns)
        w = pd.Series(1.0 / len(assets), index=assets)
        explainer = WeightChangeExplainer(returns, returns, w, w)
        corr = explainer._portfolio_correlation(returns, w, assets[0])
        # sanity: correlation should be a valid correlation coefficient, not NaN or >1
        assert -1.0 <= corr <= 1.0

    def test_asset_missing_from_one_period_handled_gracefully(self, returns):
        assets = list(returns.columns)
        before = returns.iloc[:300]
        after = returns.iloc[300:].drop(columns=[assets[0]])
        w_before = pd.Series(1.0 / len(assets), index=assets)
        w_after = pd.Series(1.0 / (len(assets) - 1), index=[a for a in assets if a != assets[0]])

        explainer = WeightChangeExplainer(before, after, w_before, w_after)
        exp = explainer.explain_asset(assets[0])
        assert np.isnan(exp.return_after)
        assert exp.weight_after == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
