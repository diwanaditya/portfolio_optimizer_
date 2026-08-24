"""
End-to-end demo: runs every engine in the Portfolio Optimizer on a synthetic
8-asset universe (US/Intl/EM equity, govt/corp bonds, gold, commodities,
REITs) and prints a full research-desk style report, then generates an
HTML tearsheet backed by a real walk-forward backtest.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from synthetic_data import generate_synthetic_universe, generate_market_caps

from portfolio_optimizer.estimators.expected_returns import mean_historical_return
from portfolio_optimizer.estimators.covariance import ledoit_wolf_shrinkage
from portfolio_optimizer.optimizers.markowitz import MarkowitzOptimizer
from portfolio_optimizer.optimizers.black_litterman import BlackLitterman
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

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
SEP = "=" * 78


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def main():
    returns = generate_synthetic_universe(n_days=1000)
    assets = list(returns.columns)
    market_caps = generate_market_caps(assets)

    section("1. DATA")
    print(f"Universe: {assets}")
    print(f"Period: {returns.index[0].date()} -> {returns.index[-1].date()} "
          f"({len(returns)} trading days)")

    # ---------------------------------------------------------------- #
    mu = mean_historical_return(returns)
    cov, shrinkage = ledoit_wolf_shrinkage(returns)
    section("2. ESTIMATORS")
    print(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.3f}")
    print("\nAnnualized historical returns:\n", mu.round(4))

    # ---------------------------------------------------------------- #
    section("3. MARKOWITZ MEAN-VARIANCE")
    mv = MarkowitzOptimizer(mu, cov, risk_free_rate=0.03)
    max_sharpe = mv.max_sharpe()
    min_vol = mv.min_volatility()
    print(f"Max-Sharpe portfolio -> return={max_sharpe.expected_return:.2%}, "
          f"vol={max_sharpe.volatility:.2%}, Sharpe={max_sharpe.sharpe_ratio:.2f}")
    print(max_sharpe.weights.round(3))
    print(f"\nMin-volatility portfolio -> return={min_vol.expected_return:.2%}, "
          f"vol={min_vol.volatility:.2%}")

    # ---------------------------------------------------------------- #
    section("4. BLACK-LITTERMAN")
    bl = BlackLitterman(cov, market_caps=market_caps, risk_aversion=2.5, tau=0.05)
    bl.add_absolute_view("EM_EQUITY", value=0.14, confidence=0.6)
    bl.add_relative_view("US_EQUITY", "GOVT_BONDS", value=0.06, confidence=0.7)
    post_mu, post_cov = bl.posterior()
    print("Prior (equilibrium) returns:\n", bl.implied_prior().round(4))
    print("\nPosterior (view-adjusted) returns:\n", post_mu.round(4))
    bl_opt = MarkowitzOptimizer(post_mu, post_cov, risk_free_rate=0.03)
    bl_res = bl_opt.max_sharpe()
    print(f"\nBL-implied max-Sharpe portfolio -> Sharpe={bl_res.sharpe_ratio:.2f}")
    print(bl_res.weights.round(3))

    # ---------------------------------------------------------------- #
    section("5. RISK PARITY (ERC + HRP)")
    erc = RiskParity(cov)
    erc_w = erc.solve()
    print("ERC weights:\n", erc_w.round(3))
    print("\nERC risk contribution %:\n",
          erc.risk_contribution_report(erc_w)["risk_contribution_pct"].round(3))

    hrp = HierarchicalRiskParity(returns)
    hrp_w = hrp.solve()
    print("\nHRP weights (clustering-based, no matrix inversion):\n", hrp_w.round(3))

    # ---------------------------------------------------------------- #
    section("6. CVaR OPTIMIZATION")
    cvar_opt = CVaROptimizer(returns, alpha=0.95)
    cvar_res = cvar_opt.optimize()
    print(f"Min-CVaR(95%) portfolio -> return={cvar_res.expected_return:.2%}, "
          f"CVaR={cvar_res.cvar:.2%}, VaR={cvar_res.var:.2%}")
    print(cvar_res.weights.round(3))

    # ================================================================ #
    section("7. [NEW] ENTROPY POOLING — generalized views beyond BL")
    ep = EntropyPooling(returns)
    ep.add_mean_view("GOLD", value=0.10 / 252, kind=">=")   # daily-scale view
    ep.add_volatility_view("EM_EQUITY", annualized_vol=0.35, kind="=")
    ep.add_correlation_view("US_EQUITY", "INTL_EQUITY", correlation=0.85, kind="=")
    ep_mu, ep_cov = ep.posterior_moments()
    ess = ep.effective_sample_size()
    print(f"Effective sample size after view-tilting: {ess:.0f} / {len(returns)} scenarios")
    print("Posterior means (entropy-pooled):\n", ep_mu.round(4))

    # ---------------------------------------------------------------- #
    section("8. [NEW] CVaR RISK PARITY — equalize tail-risk contribution")
    crp = CVaRRiskParity(returns, alpha=0.95)
    crp_res = crp.solve()
    print("CVaR-parity weights:\n", crp_res.weights.round(3))
    print("\nCVaR contribution by asset:\n", crp_res.cvar_contributions.round(5))

    # ---------------------------------------------------------------- #
    section("9. [NEW] REGIME-SWITCHING OVERLAY (HMM)")
    regime = RegimeSwitchingOverlay(returns, n_regimes=2)
    rep = regime.report()
    print(f"Current regime: {rep.regime_labels[rep.current_regime]}")
    print("\nRegime transition matrix:\n", rep.transition_matrix.round(3))
    print("\nFiltered regime probabilities (today):\n", rep.filtered_probabilities.round(3))
    blended_mu, blended_cov = regime.blended_moments()
    print("\nRegime-blended expected returns:\n", blended_mu.round(4))

    # ---------------------------------------------------------------- #
    section("10. [NEW] ROBUST / RESAMPLED EFFICIENT FRONTIER (Michaud)")
    resampler = ResampledEfficientFrontier(returns, n_resamples=150, block_size=20,
                                            risk_free_rate=0.03)
    resampled = resampler.resampled_max_sharpe()
    naive_hhi = (max_sharpe.weights ** 2).sum()
    resampled_hhi = (resampled.weights ** 2).sum()
    print(f"Naive Markowitz concentration (HHI): {naive_hhi:.3f}")
    print(f"Resampled portfolio concentration (HHI): {resampled_hhi:.3f}  "
          f"({'more' if resampled_hhi < naive_hhi else 'less'} diversified)")
    print("\nResampled max-Sharpe weights:\n", resampled.weights.round(3))
    print("\nWeight stability (std dev across resamples):\n",
          resampled.weight_std_across_resamples.round(4))

    # ---------------------------------------------------------------- #
    section("11. [NEW] FACTOR RISK MODEL (statistical / PCA)")
    frm = FactorRiskModel(returns)
    factor_result = frm.fit_statistical(n_factors=3)
    print("Factor exposures (assets x PCA factors):\n", factor_result.exposures.round(3))
    print("\nR-squared explained by factors:\n", factor_result.r_squared.round(3))
    port_exposure = FactorRiskModel.portfolio_factor_exposure(max_sharpe.weights, factor_result.exposures)
    print("\nMax-Sharpe portfolio's net factor exposure:\n", port_exposure.round(3))

    # ---------------------------------------------------------------- #
    section("12. [NEW] STRESS TESTING & TAIL-RISK SIMULATION")
    stress = StressTester(returns, max_sharpe.weights)
    scenario_table = stress.run_all_historical_scenarios()
    print("Historical scenario replay (portfolio P&L):\n", scenario_table)
    mc = stress.student_t_monte_carlo(n_sims=50_000)
    print(f"\nStudent-t Monte Carlo (fat-tailed, df={mc.degrees_of_freedom:.1f}):")
    print(f"  1-day VaR 95% / 99%:  {mc.var_95:.2%} / {mc.var_99:.2%}")
    print(f"  1-day CVaR 95% / 99%: {mc.cvar_95:.2%} / {mc.cvar_99:.2%}")
    print("  Tail-loss probabilities:", {k: f"{v:.3%}" for k, v in mc.prob_loss_exceeds.items()})

    # ---------------------------------------------------------------- #
    section("13. WALK-FORWARD BACKTEST + TEARSHEET (Ledoit-Wolf Max-Sharpe strategy)")

    def strategy(window: pd.DataFrame) -> pd.Series:
        mu_w = mean_historical_return(window)
        cov_w, _ = ledoit_wolf_shrinkage(window)
        opt = MarkowitzOptimizer(mu_w, cov_w, risk_free_rate=0.03)
        res = opt.max_sharpe()
        return res.weights if res.success else pd.Series(1 / len(mu_w), index=mu_w.index)

    backtester = WalkForwardBacktester(
        returns, strategy_fn=strategy, lookback_periods=252,
        rebalance_every=21, transaction_cost_bps=8, no_trade_band=0.02,
    )
    bt_result = backtester.run()
    for k, v in bt_result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    tearsheet = Tearsheet(bt_result, strategy_name="ADC Ledoit-Wolf Max-Sharpe Strategy")
    path = tearsheet.to_html(os.path.join(out_dir, "portfolio_tearsheet.html"))
    print(f"\nTearsheet written to: {path}")

    section("DONE — all 4 core engines + 7 research extensions executed successfully")


if __name__ == "__main__":
    main()
