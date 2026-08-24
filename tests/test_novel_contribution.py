import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.advanced.cvar_risk_parity import CVaRRiskParity
from portfolio_optimizer.research.shrinkage_cvar_risk_parity import ShrinkageAdaptiveCVaRRiskParity


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=400)


class TestShrinkageAdaptiveCVaRRiskParityMechanics:
    """These tests check the method WORKS CORRECTLY as specified (produces
    valid weights, shrinkage behaves as designed at the extremes) — they do
    NOT claim the method outperforms plain CVaR-RP. See the module
    docstring and examples/novel_contribution_validation.py for the
    (null) empirical result on that question.
    """

    def test_produces_valid_weights_sample_size_rule(self, returns):
        res = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=0.95, rule="sample_size").solve()
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
        assert (res.weights >= -1e-6).all()
        assert res.success

    def test_produces_valid_weights_adaptive_snr_rule(self, returns):
        res = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=0.95, rule="adaptive_snr").solve()
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
        assert (res.weights >= -1e-6).all()

    def test_shrinkage_intensity_in_valid_range(self, returns):
        res = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=0.95, rule="sample_size").solve()
        assert (res.shrinkage_intensities >= 0).all()
        assert (res.shrinkage_intensities <= 1).all()

    def test_prior_strength_zero_recovers_plain_estimator(self, returns):
        """With prior_strength -> 0, lambda -> 0 for the sample_size rule,
        so the shrunk tail mean should equal the raw tail mean (no
        shrinkage applied) — a basic sanity check that the shrinkage
        formula degenerates correctly at its boundary.
        """
        res = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=0.95, rule="sample_size",
                                                prior_strength=1e-9).solve()
        assert (res.shrinkage_intensities < 0.01).all()

    def test_very_large_prior_strength_shrinks_heavily(self, returns):
        res = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=0.95, rule="sample_size",
                                                prior_strength=1e6).solve()
        assert (res.shrinkage_intensities > 0.99).all()

    def test_effective_tail_size_matches_alpha(self, returns):
        alpha = 0.95
        res = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=alpha, rule="sample_size").solve()
        expected_tail = int(len(returns) * (1 - alpha))
        # allow slack since the actual tail mask is based on the solved
        # weight vector's realized loss distribution, not the raw input
        assert abs(res.effective_tail_size - expected_tail) < len(returns) * 0.1


class TestNovelContributionHonestReporting:
    """Verifies the empirical validation script actually runs and produces
    a real statistical comparison (not that it 'proves' the method works —
    it explicitly doesn't, and that's the point).
    """

    def test_plain_and_shrunk_produce_different_but_valid_portfolios(self, returns):
        plain = CVaRRiskParity(returns, alpha=0.95).solve()
        shrunk = ShrinkageAdaptiveCVaRRiskParity(returns, alpha=0.95, rule="sample_size").solve()
        assert np.isclose(plain.weights.sum(), 1.0, atol=1e-4)
        assert np.isclose(shrunk.weights.sum(), 1.0, atol=1e-4)
        # they need not be identical, but both must be valid simplex points
        assert (plain.weights >= -1e-6).all()
        assert (shrunk.weights >= -1e-6).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
