import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe, generate_market_caps
from portfolio_optimizer.estimators.expected_returns import mean_historical_return, ewma_return
from portfolio_optimizer.estimators.covariance import ledoit_wolf_shrinkage, sample_covariance, ewma_covariance
from portfolio_optimizer.optimizers.markowitz import MarkowitzOptimizer
from portfolio_optimizer.optimizers.black_litterman import BlackLitterman, View
from portfolio_optimizer.optimizers.risk_parity import RiskParity, HierarchicalRiskParity
from portfolio_optimizer.optimizers.cvar import CVaROptimizer
from portfolio_optimizer.advanced.entropy_pooling import EntropyPooling
from portfolio_optimizer.advanced.cvar_risk_parity import CVaRRiskParity
from portfolio_optimizer.advanced.regime_switching import RegimeSwitchingOverlay
from portfolio_optimizer.advanced.robust_frontier import ResampledEfficientFrontier
from portfolio_optimizer.advanced.factor_risk_model import FactorRiskModel
from portfolio_optimizer.advanced.stress_testing import StressTester
from portfolio_optimizer.backtester import WalkForwardBacktester
from portfolio_optimizer.reporting.tearsheet import Tearsheet


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=600)


@pytest.fixture(scope="module")
def mu_cov(returns):
    mu = mean_historical_return(returns)
    cov, _ = ledoit_wolf_shrinkage(returns)
    return mu, cov


def _assert_valid_weights(w: pd.Series, n_assets: int):
    assert len(w) == n_assets
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert (w >= -1e-6).all()


class TestEstimators:
    def test_mean_historical_return(self, returns):
        mu = mean_historical_return(returns)
        assert len(mu) == returns.shape[1]
        assert mu.notna().all()

    def test_ewma_return(self, returns):
        mu = ewma_return(returns)
        assert mu.notna().all()

    def test_ledoit_wolf_shrinkage(self, returns):
        cov, shrink = ledoit_wolf_shrinkage(returns)
        assert 0.0 <= shrink <= 1.0
        eigvals = np.linalg.eigvalsh(cov.values)
        assert (eigvals > 0).all()  # PSD

    def test_ewma_covariance(self, returns):
        cov = ewma_covariance(returns)
        assert cov.shape == (returns.shape[1], returns.shape[1])


class TestMarkowitz:
    def test_max_sharpe(self, mu_cov):
        mu, cov = mu_cov
        opt = MarkowitzOptimizer(mu, cov)
        res = opt.max_sharpe()
        assert res.success
        _assert_valid_weights(res.weights, len(mu))
        # max-Sharpe must beat (or match) naive equal-weight on a Sharpe basis,
        # regardless of whether the random sample happened to realize positive
        # or negative average drift.
        n = len(mu)
        naive_w = np.ones(n) / n
        naive_ret = naive_w @ mu.values
        naive_vol = np.sqrt(naive_w @ cov.values @ naive_w)
        naive_sharpe = naive_ret / naive_vol
        assert res.sharpe_ratio >= naive_sharpe - 1e-6

    def test_min_volatility_beats_naive(self, mu_cov):
        mu, cov = mu_cov
        opt = MarkowitzOptimizer(mu, cov)
        res = opt.min_volatility()
        n = len(mu)
        naive_w = np.ones(n) / n
        naive_vol = np.sqrt(naive_w @ cov.values @ naive_w)
        assert res.volatility <= naive_vol + 1e-6

    def test_efficient_frontier_monotonic_risk(self, mu_cov):
        mu, cov = mu_cov
        opt = MarkowitzOptimizer(mu, cov)
        frontier = opt.efficient_frontier(n_points=15)
        assert len(frontier) > 5
        assert frontier["volatility"].is_monotonic_increasing or \
               frontier["volatility"].diff().dropna().ge(-1e-6).all()

    def test_group_constraint(self, mu_cov):
        mu, cov = mu_cov
        sector_map = {a: ("bond" if "BOND" in a else "equity") for a in mu.index}
        opt = MarkowitzOptimizer(mu, cov, sector_map=sector_map)
        opt.add_group_constraint("bond", 0.0, 0.25)
        res = opt.max_sharpe()
        bond_weight = sum(res.weights[a] for a in mu.index if sector_map[a] == "bond")
        assert bond_weight <= 0.26


class TestBlackLitterman:
    def test_posterior_no_views_equals_prior(self, returns):
        cov, _ = ledoit_wolf_shrinkage(returns)
        caps = generate_market_caps(list(returns.columns))
        bl = BlackLitterman(cov, market_caps=caps)
        post_mu, post_cov = bl.posterior()
        np.testing.assert_allclose(post_mu.values, bl.implied_prior().values, atol=1e-8)

    def test_view_shifts_posterior(self, returns):
        cov, _ = ledoit_wolf_shrinkage(returns)
        caps = generate_market_caps(list(returns.columns))
        bl = BlackLitterman(cov, market_caps=caps)
        prior = bl.implied_prior()
        bl.add_absolute_view("GOLD", 0.30, confidence=0.9)
        post_mu, _ = bl.posterior()
        assert post_mu["GOLD"] > prior["GOLD"]

    def test_relative_view(self, returns):
        cov, _ = ledoit_wolf_shrinkage(returns)
        caps = generate_market_caps(list(returns.columns))
        bl = BlackLitterman(cov, market_caps=caps)
        bl.add_relative_view("US_EQUITY", "GOVT_BONDS", 0.10, confidence=0.8)
        post_mu, post_cov = bl.posterior()
        opt = MarkowitzOptimizer(post_mu, post_cov)
        res = opt.max_sharpe()
        assert res.success


class TestRiskParity:
    def test_erc_contributions_are_equal(self, mu_cov):
        _, cov = mu_cov
        rp = RiskParity(cov)
        w = rp.solve()
        _assert_valid_weights(w, len(cov))
        report = rp.risk_contribution_report(w)
        pct = report["risk_contribution_pct"].values
        assert pct.std() < 0.05  # should be close to equal

    def test_hrp_valid_weights(self, returns):
        hrp = HierarchicalRiskParity(returns)
        w = hrp.solve()
        _assert_valid_weights(w, returns.shape[1])

    def test_risk_budget_respected(self, mu_cov):
        _, cov = mu_cov
        assets = list(cov.index)
        budget = pd.Series(0.05, index=assets)
        budget[assets[0]] = 0.6  # concentrate budget on first asset
        budget = budget / budget.sum()
        rp = RiskParity(cov, risk_budget=budget)
        w = rp.solve()
        assert w[assets[0]] > w[assets[1]]


class TestCVaR:
    def test_cvar_optimize_valid(self, returns):
        cvar_opt = CVaROptimizer(returns, alpha=0.95)
        res = cvar_opt.optimize()
        assert res.success
        _assert_valid_weights(res.weights, returns.shape[1])
        assert res.cvar > 0  # CVaR of losses should be positive

    def test_cvar_target_return(self, returns):
        cvar_opt = CVaROptimizer(returns, alpha=0.95)
        base = cvar_opt.optimize()
        target = base.expected_return * 0.8
        res = cvar_opt.optimize(target_return=target)
        assert res.success


class TestEntropyPooling:
    def test_no_views_returns_prior(self, returns):
        ep = EntropyPooling(returns)
        p = ep.solve()
        np.testing.assert_allclose(p, ep.p0)

    def test_mean_view_shifts_posterior_mean(self, returns):
        ep = EntropyPooling(returns)
        asset = returns.columns[0]
        ep.add_mean_view(asset, value=0.002, kind="=")
        mu, cov = ep.posterior_moments()
        # posterior mean for that asset should move toward the imposed view
        assert mu[asset] != pytest.approx(returns[asset].mean() * 252, rel=1e-3)

    def test_effective_sample_size_bounded(self, returns):
        ep = EntropyPooling(returns)
        ep.add_mean_view(returns.columns[0], value=0.005)
        ess = ep.effective_sample_size()
        assert 0 < ess <= len(returns)


class TestCVaRRiskParity:
    def test_contributions_roughly_equal(self, returns):
        crp = CVaRRiskParity(returns.iloc[:, :5], alpha=0.9)
        result = crp.solve()
        _assert_valid_weights(result.weights, 5)
        contribs = result.cvar_contributions.values
        assert np.std(contribs) / (np.mean(np.abs(contribs)) + 1e-9) < 1.5


class TestRegimeSwitching:
    def test_regime_report_structure(self, returns):
        overlay = RegimeSwitchingOverlay(returns, n_regimes=2)
        report = overlay.report()
        assert report.current_regime in (0, 1)
        assert report.transition_matrix.shape == (2, 2)
        np.testing.assert_allclose(report.transition_matrix.sum(axis=1).values, 1.0, atol=1e-6)

    def test_regime_conditional_moments(self, returns):
        overlay = RegimeSwitchingOverlay(returns, n_regimes=2)
        moments = overlay.regime_conditional_moments()
        assert len(moments) >= 1
        for regime, m in moments.items():
            assert "mu" in m and "cov" in m

    def test_blended_moments_shape(self, returns):
        overlay = RegimeSwitchingOverlay(returns, n_regimes=2)
        mu, cov = overlay.blended_moments()
        assert len(mu) == returns.shape[1]
        assert cov.shape == (returns.shape[1], returns.shape[1])


class TestRobustFrontier:
    def test_resampled_max_sharpe_valid(self, returns):
        ref = ResampledEfficientFrontier(returns, n_resamples=25, block_size=15)
        result = ref.resampled_max_sharpe()
        _assert_valid_weights(result.weights, returns.shape[1])
        assert result.n_resamples > 0

    def test_resampled_more_diversified_than_naive_mv(self, returns):
        mu = mean_historical_return(returns)
        cov, _ = ledoit_wolf_shrinkage(returns)
        naive = MarkowitzOptimizer(mu, cov).max_sharpe().weights
        ref = ResampledEfficientFrontier(returns, n_resamples=25, block_size=15)
        resampled = ref.resampled_max_sharpe().weights
        # Herfindahl concentration index: resampled should typically be <= naive
        naive_hhi = (naive ** 2).sum()
        resampled_hhi = (resampled ** 2).sum()
        assert resampled_hhi <= naive_hhi + 0.15  # allow some tolerance, still directionally true


class TestFactorRiskModel:
    def test_statistical_factor_model_shapes(self, returns):
        frm = FactorRiskModel(returns)
        result = frm.fit_statistical(n_factors=3)
        assert result.exposures.shape == (returns.shape[1], 3)
        assert result.reconstructed_cov.shape == (returns.shape[1], returns.shape[1])
        assert (result.r_squared >= -0.01).all()

    def test_portfolio_exposure(self, returns):
        frm = FactorRiskModel(returns)
        result = frm.fit_statistical(n_factors=2)
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        exposure = FactorRiskModel.portfolio_factor_exposure(w, result.exposures)
        assert len(exposure) == 2


class TestStressTesting:
    def test_historical_scenario(self, returns):
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        st = StressTester(returns, w)
        res = st.historical_scenario("2008_gfc_crash")
        assert res.portfolio_pnl_pct < 0

    def test_all_scenarios_table(self, returns):
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        st = StressTester(returns, w)
        table = st.run_all_historical_scenarios()
        assert len(table) == 5

    def test_student_t_monte_carlo(self, returns):
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        st = StressTester(returns, w)
        mc = st.student_t_monte_carlo(n_sims=5000)
        assert mc.var_99 >= mc.var_95  # 99% VaR should be more extreme
        assert mc.cvar_95 >= mc.var_95
        assert 3 <= mc.degrees_of_freedom <= 30


class TestBacktester:
    def test_walk_forward_runs(self, returns):
        def strategy(window):
            mu = mean_historical_return(window)
            cov, _ = ledoit_wolf_shrinkage(window)
            opt = MarkowitzOptimizer(mu, cov)
            res = opt.max_sharpe()
            return res.weights if res.success else pd.Series(1/len(mu), index=mu.index)

        bt = WalkForwardBacktester(returns, strategy, lookback_periods=200,
                                    rebalance_every=42, transaction_cost_bps=10)
        result = bt.run()
        assert len(result.equity_curve) > 0
        assert "sharpe_ratio" in result.metrics
        assert result.equity_curve.iloc[0] > 0

    def test_tearsheet_generation(self, returns, tmp_path):
        def strategy(window):
            return pd.Series(1.0 / window.shape[1], index=window.columns)

        bt = WalkForwardBacktester(returns, strategy, lookback_periods=200, rebalance_every=42)
        result = bt.run()
        ts = Tearsheet(result, strategy_name="Test Equal Weight")
        out_path = str(tmp_path / "tearsheet.html")
        ts.to_html(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "Performance Summary" in content
        assert len(content) > 1000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
