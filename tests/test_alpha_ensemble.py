import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.alpha.signal_ensemble import (
    MomentumSignal, ShortTermReversalSignal, VolatilityCarrySignal, QualityTrendSignal,
    SignalEnsemble, Signal,
)
from portfolio_optimizer.advanced.entropy_pooling import EntropyPooling
from portfolio_optimizer.optimizers.markowitz import MarkowitzOptimizer


@pytest.fixture(scope="module")
def prices():
    returns = generate_synthetic_universe(n_days=600)
    return (1 + returns).cumprod() * 100


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=600)


class TestIndividualSignals:
    def test_momentum_signal_shape_and_zscore(self, prices):
        sig = MomentumSignal(lookback=126, skip=10)
        scores = sig.compute(prices)
        assert scores.shape == prices.shape
        # cross-sectional zscore per row should have ~0 mean (where not all-NaN)
        valid_rows = scores.dropna(how="all")
        row_means = valid_rows.mean(axis=1).dropna()
        assert (row_means.abs() < 0.5).all()

    def test_short_term_reversal_signal_runs(self, prices):
        sig = ShortTermReversalSignal(lookback=5)
        scores = sig.compute(prices)
        assert scores.shape == prices.shape

    def test_volatility_carry_signal_favors_low_vol(self, prices):
        sig = VolatilityCarrySignal(lookback=63)
        scores = sig.compute(prices)
        # sanity: the asset with lowest realized vol recently should score higher than
        # the asset with highest realized vol, on average
        returns_est = prices.pct_change()
        recent_vol = returns_est.tail(63).std()
        lowest_vol_asset = recent_vol.idxmin()
        highest_vol_asset = recent_vol.idxmax()
        last_scores = scores.iloc[-1]
        assert last_scores[lowest_vol_asset] > last_scores[highest_vol_asset]

    def test_quality_trend_signal_runs(self, prices):
        sig = QualityTrendSignal(lookback=90)
        scores = sig.compute(prices)
        assert scores.shape == prices.shape


class TestSignalEnsemble:
    def test_calibrate_weights_sum_to_one_or_equal_fallback(self, prices):
        ensemble = SignalEnsemble([
            MomentumSignal(lookback=126, skip=10),
            ShortTermReversalSignal(lookback=5),
        ], forward_period=21)
        ew = ensemble.calibrate_weights(prices, method="ic_weighted")
        assert np.isclose(ew.weights.sum(), 1.0, atol=1e-6)
        assert (ew.weights >= 0).all()

    def test_equal_weight_calibration(self, prices):
        ensemble = SignalEnsemble([
            MomentumSignal(), ShortTermReversalSignal(), VolatilityCarrySignal(),
        ], forward_period=21)
        ew = ensemble.calibrate_weights(prices, method="equal")
        assert np.allclose(ew.weights.values, 1.0 / 3, atol=1e-6)

    def test_information_coefficient_bounded(self, prices):
        ensemble = SignalEnsemble([MomentumSignal()], forward_period=21)
        scores = ensemble.compute_all_signals(prices)["momentum_12_1"]
        ic = ensemble.information_coefficient(scores, prices)
        assert (ic >= -1.0).all() and (ic <= 1.0).all()

    def test_blended_score_shape(self, prices):
        ensemble = SignalEnsemble([MomentumSignal(), ShortTermReversalSignal()], forward_period=21)
        blended = ensemble.blended_score(prices)
        assert blended.shape == prices.shape

    def test_latest_ranking_covers_all_assets(self, prices):
        ensemble = SignalEnsemble([MomentumSignal(), VolatilityCarrySignal()], forward_period=21)
        ranking = ensemble.latest_ranking(prices)
        assert set(ranking.index) == set(prices.columns)
        assert ranking.is_monotonic_decreasing

    def test_zero_ic_signals_get_zero_weight_not_negative(self, prices):
        """A signal with negative historical IC should be floored at zero
        weight, not flipped to negative -- verifies the conservative
        design choice documented in the module docstring.
        """
        ensemble = SignalEnsemble([
            MomentumSignal(), ShortTermReversalSignal(), VolatilityCarrySignal(), QualityTrendSignal(),
        ], forward_period=21)
        ew = ensemble.calibrate_weights(prices, method="ic_weighted")
        assert (ew.weights >= 0).all()  # never negative, by construction


class TestSignalToViewsToOptimizer:
    """The full pipeline test: signals -> views -> Entropy Pooling ->
    optimizer, verifying the hand-off actually works end-to-end and stays
    within this repo's existing risk-aware optimization layer rather than
    bypassing it.
    """

    def test_scores_to_views_produces_valid_tuples(self, prices, returns):
        ensemble = SignalEnsemble([MomentumSignal(), ShortTermReversalSignal()], forward_period=21)
        mu = returns.mean() * 252
        views = ensemble.scores_to_views(prices, mu, view_strength=0.02, top_n=2)
        assert len(views) > 0
        for asset, tilted_return, confidence in views:
            assert asset in prices.columns
            assert np.isfinite(tilted_return)
            assert 0.0 <= confidence <= 1.0

    def test_full_pipeline_signal_to_entropy_pooling_to_markowitz(self, prices, returns):
        ensemble = SignalEnsemble([
            MomentumSignal(lookback=126, skip=10), VolatilityCarrySignal(lookback=63),
        ], forward_period=21)
        mu = returns.mean() * 252
        views = ensemble.scores_to_views(prices, mu, view_strength=0.03, top_n=2)

        ep = EntropyPooling(returns)
        for asset, tilted_return, confidence in views:
            ep.add_mean_view(asset, value=tilted_return / 252, kind="=")

        post_mu, post_cov = ep.posterior_moments()
        opt = MarkowitzOptimizer(post_mu, post_cov, risk_free_rate=0.03)
        result = opt.max_sharpe()

        assert result.success
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-3)
        assert (result.weights >= -1e-6).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
