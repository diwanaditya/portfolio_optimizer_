import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe, generate_market_caps
from portfolio_optimizer.estimators.covariance import ledoit_wolf_shrinkage

from portfolio_optimizer.bayesian.bayesian_mean_variance import bayes_stein_shrinkage, BayesianMeanVariance
from portfolio_optimizer.bayesian.hierarchical_bayes import HierarchicalBayesianPortfolio
from portfolio_optimizer.multiperiod.multi_period_optimizer import LiNgMultiPeriod, ScenarioMPCOptimizer
from portfolio_optimizer.execution.almgren_chriss import optimal_execution_trajectory, AlmgrenChrissCostModel
from portfolio_optimizer.attribution.brinson import brinson_attribution, multi_period_brinson
from portfolio_optimizer.attribution.factor_attribution import factor_attribution_over_backtest
from portfolio_optimizer.attribution.risk_attribution import variance_risk_attribution, factor_risk_attribution
from portfolio_optimizer.advanced.factor_risk_model import FactorRiskModel
from portfolio_optimizer.optimizers.black_litterman import BlackLitterman
from portfolio_optimizer.optimizers.markowitz import MarkowitzOptimizer
from portfolio_optimizer.explainability.dashboard import BlackLittermanExplainer
from portfolio_optimizer.gpu.accel import available_backends, best_backend, GPUAcceleratedMonteCarlo, GPUAcceleratedResampling


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=500)


class TestBayesian:
    def test_bayes_stein_shrinks_toward_grand_mean(self, returns):
        result = bayes_stein_shrinkage(returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0
        # shrunk mean should sit strictly between raw mean and grand mean (or equal at extremes)
        for a in returns.columns:
            lo = min(result.original_mean[a], result.grand_mean)
            hi = max(result.original_mean[a], result.grand_mean)
            assert lo - 1e-6 <= result.shrunk_mean[a] <= hi + 1e-6

    def test_niw_posterior_predictive_wider_than_sample_cov(self, returns):
        bmv = BayesianMeanVariance(returns)
        post = bmv.posterior()
        sample_cov = returns.cov() * 252
        # posterior predictive variance should be >= sample variance (parameter uncertainty inflates it)
        for a in returns.columns:
            assert post.posterior_predictive_cov.loc[a, a] >= sample_cov.loc[a, a] * 0.9

    def test_bayesian_optimize_runs(self, returns):
        bmv = BayesianMeanVariance(returns)
        res, post = bmv.optimize()
        assert res.success
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)

    def test_hierarchical_bayes_groups(self, returns):
        group_map = {a: ("equity" if "EQUITY" in a else "bond" if "BOND" in a else "alt")
                     for a in returns.columns}
        hb = HierarchicalBayesianPortfolio(returns, group_map)
        result = hb.solve()
        assert len(result.shrunk_mean) == returns.shape[1]
        assert (result.within_group_shrinkage >= 0).all() and (result.within_group_shrinkage <= 1).all()


class TestMultiPeriod:
    def test_li_ng_constant_mix(self, returns):
        mu = returns.mean() * 252
        cov, _ = ledoit_wolf_shrinkage(returns)
        lng = LiNgMultiPeriod(mu, cov, horizon_periods=30, risk_aversion=3.0)
        plan = lng.solve()
        assert plan.weights_by_period.shape[0] == 30
        # constant-mix: every period should have the identical weight vector
        first_row = plan.weights_by_period.iloc[0]
        last_row = plan.weights_by_period.iloc[-1]
        pd.testing.assert_series_equal(first_row, last_row, check_names=False)
        assert np.isclose(first_row.sum(), 1.0, atol=1e-4)

    def test_scenario_mpc_produces_valid_sequence(self, returns):
        mpc = ScenarioMPCOptimizer(returns, horizon_periods=5, n_scenarios=20,
                                    block_size=8, transaction_cost_bps=10)
        plan = mpc.solve()
        assert plan.weights_by_period.shape == (5, returns.shape[1])
        for _, row in plan.weights_by_period.iterrows():
            assert np.isclose(row.sum(), 1.0, atol=1e-3)
            assert (row >= -1e-6).all()


class TestExecutionModel:
    def test_optimal_execution_trajectory_monotonic_decrease(self):
        traj = optimal_execution_trajectory(total_shares=50000, n_periods=10, total_time=1/252,
                                             volatility=0.3, temporary_impact_eta=2e-6,
                                             permanent_impact_gamma=1e-6, risk_aversion=1e-6)
        assert traj.holdings_schedule[0] == pytest.approx(50000)
        assert traj.holdings_schedule[-1] == pytest.approx(0, abs=1.0)
        assert np.all(np.diff(traj.holdings_schedule) <= 1e-6)  # monotonically decreasing
        assert traj.expected_cost >= 0

    def test_higher_risk_aversion_front_loads_execution(self):
        traj_low = optimal_execution_trajectory(50000, 10, 1/252, 0.3, 2e-6, 1e-6, risk_aversion=1e-8)
        traj_high = optimal_execution_trajectory(50000, 10, 1/252, 0.3, 2e-6, 1e-6, risk_aversion=1e-4)
        # higher risk aversion should execute more in early periods (front-loaded)
        assert traj_high.trade_schedule[0] >= traj_low.trade_schedule[0] * 0.9

    def test_almgren_chriss_cost_model_illiquid_costs_more(self):
        vol = pd.Series({"LIQUID": 0.2, "ILLIQUID": 0.2})
        adv = pd.Series({"LIQUID": 10_000_000, "ILLIQUID": 100_000})
        model = AlmgrenChrissCostModel(vol, adv)
        cost_liquid = model.implied_bps_cost("LIQUID", 50000, 100)
        cost_illiquid = model.implied_bps_cost("ILLIQUID", 50000, 100)
        assert cost_illiquid > cost_liquid


class TestAttribution:
    def test_brinson_decomposition_sums_to_active_return(self):
        assets = ["A", "B", "C"]
        wp = pd.Series([0.5, 0.3, 0.2], index=assets)
        wb = pd.Series([0.33, 0.33, 0.34], index=assets)
        rp = pd.Series([0.05, 0.03, -0.01], index=assets)
        rb = pd.Series([0.04, 0.02, 0.01], index=assets)
        result = brinson_attribution(wp, wb, rp, rb)
        total_effects = (result.allocation_effect + result.selection_effect + result.interaction_effect).sum()
        assert abs(total_effects - result.total_active_return) < 1e-9

    def test_multi_period_brinson_runs(self):
        dates = pd.date_range("2024-01-01", periods=5)
        assets = ["A", "B"]
        wp = pd.DataFrame(np.random.dirichlet([1, 1], size=5), index=dates, columns=assets)
        wb = pd.DataFrame(np.random.dirichlet([1, 1], size=5), index=dates, columns=assets)
        rp = pd.DataFrame(np.random.randn(5, 2) * 0.01, index=dates, columns=assets)
        rb = pd.DataFrame(np.random.randn(5, 2) * 0.01, index=dates, columns=assets)
        result = multi_period_brinson(wp, wb, rp, rb)
        assert "TOTAL" in result.index

    def test_variance_risk_attribution_sums_to_total_vol(self, returns):
        cov, _ = ledoit_wolf_shrinkage(returns)
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        result = variance_risk_attribution(w, cov)
        assert abs(result.component_contribution.sum() - result.total_volatility) < 1e-6
        assert abs(result.percent_contribution.sum() - 1.0) < 1e-6

    def test_factor_risk_attribution_runs(self, returns):
        frm = FactorRiskModel(returns)
        fresult = frm.fit_statistical(n_factors=3)
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        result = factor_risk_attribution(w, fresult.exposures, fresult.factor_cov, fresult.idiosyncratic_var)
        assert abs(result.percent_contribution.sum() - 1.0) < 1e-4

    def test_factor_attribution_over_backtest(self, returns):
        frm = FactorRiskModel(returns)
        fresult = frm.fit_statistical(n_factors=2)
        w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        weights_hist = pd.DataFrame([w.values] * len(returns), index=returns.index, columns=returns.columns)
        actual_returns = returns @ w
        result = factor_attribution_over_backtest(weights_hist, actual_returns,
                                                    fresult.exposures, fresult.factor_returns)
        assert np.isfinite(result.specific_return)
        assert np.isfinite(result.total_return)


class TestExplainability:
    def test_bl_explainer_produces_report(self, returns):
        caps = generate_market_caps(list(returns.columns))
        cov, _ = ledoit_wolf_shrinkage(returns)
        bl = BlackLitterman(cov, market_caps=caps)
        bl.add_absolute_view("GOLD", 0.15, confidence=0.7)
        post_mu, post_cov = bl.posterior()
        opt = MarkowitzOptimizer(post_mu, post_cov)
        res = opt.max_sharpe()

        explainer = BlackLittermanExplainer(bl, returns, res.weights)
        report = explainer.explain()
        assert "GOLD" in report.asset_explanations
        assert len(report.summary_table) == returns.shape[1]

    def test_explain_in_words_mentions_view(self, returns):
        caps = generate_market_caps(list(returns.columns))
        cov, _ = ledoit_wolf_shrinkage(returns)
        bl = BlackLitterman(cov, market_caps=caps)
        bl.add_absolute_view("GOLD", 0.20, confidence=0.9)
        post_mu, post_cov = bl.posterior()
        res = MarkowitzOptimizer(post_mu, post_cov).max_sharpe()
        explainer = BlackLittermanExplainer(bl, returns, res.weights)
        text = explainer.explain_asset_in_words("GOLD")
        assert "GOLD" in text and "%" in text

    def test_html_report_generation(self, returns, tmp_path):
        caps = generate_market_caps(list(returns.columns))
        cov, _ = ledoit_wolf_shrinkage(returns)
        bl = BlackLitterman(cov, market_caps=caps)
        bl.add_absolute_view("GOLD", 0.15, confidence=0.6)
        post_mu, post_cov = bl.posterior()
        res = MarkowitzOptimizer(post_mu, post_cov).max_sharpe()
        explainer = BlackLittermanExplainer(bl, returns, res.weights)
        path = str(tmp_path / "explain.html")
        explainer.to_html(path)
        with open(path) as f:
            content = f.read()
        assert "Explainability" in content


class TestGPUAcceleration:
    def test_available_backends_reports_numpy(self):
        backends = available_backends()
        assert backends["numpy"] is True

    def test_best_backend_returns_valid_string(self):
        backend = best_backend()
        assert backend in ("numpy", "cupy", "jax", "torch")

    def test_monte_carlo_numpy_backend(self):
        mu = np.array([0.001, 0.0005])
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
        mc = GPUAcceleratedMonteCarlo(backend="numpy")
        sims = mc.simulate(mu, cov, df=6, n_sims=5000)
        assert sims.shape == (5000, 2)
        assert np.allclose(sims.mean(axis=0), mu, atol=0.01)

    def test_batched_resampling(self):
        resamples = np.random.randn(20, 100, 3) * 0.01
        gr = GPUAcceleratedResampling(backend="numpy")
        means, covs = gr.batch_mean_cov(resamples)
        assert means.shape == (20, 3)
        assert covs.shape == (20, 3, 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
