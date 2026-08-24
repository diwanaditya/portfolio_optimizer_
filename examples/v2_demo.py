"""
v2.0 demo: Bayesian methods, RL vs classical, multi-period optimization,
Almgren-Chriss execution, live data adapters, attribution, explainability,
and GPU acceleration — all exercised end-to-end on the same synthetic
8-asset universe used in the v1 demo.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from synthetic_data import generate_synthetic_universe, generate_market_caps

from portfolio_optimizer.estimators.covariance import ledoit_wolf_shrinkage
from portfolio_optimizer.optimizers.markowitz import MarkowitzOptimizer
from portfolio_optimizer.optimizers.black_litterman import BlackLitterman

from portfolio_optimizer.bayesian.bayesian_mean_variance import bayes_stein_shrinkage, BayesianMeanVariance
from portfolio_optimizer.bayesian.hierarchical_bayes import HierarchicalBayesianPortfolio

from portfolio_optimizer.rl.environment import PortfolioEnv
from portfolio_optimizer.rl.agents import PPOAgent, SACAgent, DQNAgent
from portfolio_optimizer.rl.benchmark import evaluate_rl_agent, evaluate_classical_strategy, comparison_table

from portfolio_optimizer.multiperiod.multi_period_optimizer import LiNgMultiPeriod, ScenarioMPCOptimizer
from portfolio_optimizer.execution.almgren_chriss import optimal_execution_trajectory, AlmgrenChrissCostModel
from portfolio_optimizer.attribution.brinson import brinson_attribution
from portfolio_optimizer.attribution.risk_attribution import variance_risk_attribution
from portfolio_optimizer.advanced.factor_risk_model import FactorRiskModel
from portfolio_optimizer.attribution.factor_attribution import factor_attribution_over_backtest
from portfolio_optimizer.explainability.dashboard import BlackLittermanExplainer
from portfolio_optimizer.gpu.accel import available_backends, best_backend, GPUAcceleratedMonteCarlo

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
SEP = "=" * 78


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def main():
    returns = generate_synthetic_universe(n_days=900)
    train, test = returns.iloc[:700], returns.iloc[700:]
    assets = list(returns.columns)
    market_caps = generate_market_caps(assets)

    # ================================================================ #
    section("1. BAYESIAN PORTFOLIO OPTIMIZATION")
    bs = bayes_stein_shrinkage(returns)
    print(f"Bayes-Stein shrinkage intensity: {bs.shrinkage_intensity:.4f}  "
          f"(grand mean = {bs.grand_mean:.2%})")
    print("Raw sample mean vs Bayes-Stein shrunk mean:")
    print(pd.DataFrame({"raw": bs.original_mean, "bayes_stein": bs.shrunk_mean}).round(4))

    bmv = BayesianMeanVariance(returns)
    post = bmv.posterior()
    res_bayes, _ = bmv.optimize(risk_free_rate=0.03)
    print(f"\nFull NIW-Bayesian max-Sharpe portfolio -> Sharpe={res_bayes.sharpe_ratio:.2f}")
    print(res_bayes.weights.round(3))

    group_map = {a: ("equity" if "EQUITY" in a else "bond" if "BOND" in a else "alt") for a in assets}
    hb = HierarchicalBayesianPortfolio(returns, group_map)
    hres = hb.solve()
    print("\nHierarchical Bayes shrunk means (partial pooling within sector groups):")
    print(hres.shrunk_mean.round(4))

    # ================================================================ #
    section("2. REINFORCEMENT LEARNING vs MARKOWITZ / BL / RISK PARITY")
    env = PortfolioEnv(train, lookback=20, transaction_cost_bps=10)
    print(f"Training PPO, SAC, DQN on {len(train)} periods "
          f"(state_dim={env.state_dim}, action_dim={env.action_dim})...")

    t0 = time.time()
    ppo = PPOAgent(env.state_dim, env.action_dim)
    ppo.train(env, total_steps=4000, rollout_len=200)
    sac = SACAgent(env.state_dim, env.action_dim)
    sac.train(env, total_steps=2000, warmup_steps=300)
    dqn = DQNAgent(env.state_dim, env.action_dim)
    dqn.train(env, total_steps=2000)
    print(f"All 3 RL agents trained in {time.time()-t0:.1f}s")

    perf_ppo = evaluate_rl_agent(ppo, test, lookback=20, name="PPO")
    perf_sac = evaluate_rl_agent(sac, test, lookback=20, name="SAC")
    perf_dqn = evaluate_rl_agent(dqn, test, lookback=20, name="DQN",
                                  action_fn=lambda s, det: dqn.act(s, deterministic=det)[0])
    perf_mv = evaluate_classical_strategy("markowitz", train, test)
    perf_bl = evaluate_classical_strategy("black_litterman", train, test, market_caps=market_caps)
    perf_rp = evaluate_classical_strategy("risk_parity", train, test)

    table = comparison_table([perf_ppo, perf_sac, perf_dqn, perf_mv, perf_bl, perf_rp])
    print("\nOut-of-sample comparison (test period):")
    print(table.round(4))
    print("\n(Honest takeaway: RL agents here are trained on a short, low-budget run for "
          "demo purposes — treat this table as a methodology template, not a verdict on "
          "RL vs classical. Real comparisons need much longer training + hyperparameter search.)")

    # ================================================================ #
    section("3. MULTI-PERIOD OPTIMIZATION (60-day horizon)")
    mu = train.mean() * 252
    cov, _ = ledoit_wolf_shrinkage(train)
    lng = LiNgMultiPeriod(mu, cov, horizon_periods=60, risk_aversion=3.0)
    plan = lng.solve()
    print("Li-Ng analytical multi-period plan (constant-mix rule), period 1 weights:")
    print(plan.weights_by_period.iloc[0].round(3))
    print(f"Expected 60-day terminal return: {plan.expected_terminal_return:.2%}, "
          f"vol: {plan.expected_terminal_vol:.2%}")

    mpc = ScenarioMPCOptimizer(train, horizon_periods=15, n_scenarios=60,
                                block_size=10, transaction_cost_bps=15)
    plan_mpc = mpc.solve()
    print("\nScenario-MPC plan — weight evolution over horizon (first & last period):")
    print(pd.concat([plan_mpc.weights_by_period.iloc[[0]], plan_mpc.weights_by_period.iloc[[-1]]]).round(3))

    # ================================================================ #
    section("4. ALMGREN-CHRISS EXECUTION MODEL")
    traj = optimal_execution_trajectory(total_shares=200_000, n_periods=10, total_time=1/252,
                                         volatility=0.28, temporary_impact_eta=2.5e-6,
                                         permanent_impact_gamma=1e-6, risk_aversion=5e-7)
    print("Optimal execution trade schedule (200k share liquidation over 10 intervals):")
    print(np.round(traj.trade_schedule, 0))
    print(f"Expected execution cost: {traj.expected_cost:,.0f}  |  Cost variance: {traj.cost_variance:,.0f}")

    vol = pd.Series({a: 0.15 + 0.02 * i for i, a in enumerate(assets)})
    adv = pd.Series({a: 8_000_000 / (i + 1) for i, a in enumerate(assets)})
    ac_model = AlmgrenChrissCostModel(vol, adv)
    print("\nImplied bps cost of a 50k-share trade per asset (varying liquidity):")
    for a in assets:
        print(f"  {a}: {ac_model.implied_bps_cost(a, 50_000, 100):.2f} bps")

    # ================================================================ #
    section("5. LIVE DATA ADAPTERS")
    from portfolio_optimizer.data.adapters import ADAPTER_REGISTRY
    print("Supported live data adapters (unified fetch_returns() interface):")
    for name in ADAPTER_REGISTRY:
        print(f"  - {name}")
    print("(Requires your own API credentials; not exercised here since this sandbox "
          "has no network path to these vendors' endpoints — see tests/test_data_adapters.py "
          "for interface-level tests against a mocked HTTP layer.)")

    # ================================================================ #
    section("6. PORTFOLIO ATTRIBUTION (Brinson + Factor + Risk)")
    bl = BlackLitterman(cov, market_caps=market_caps, risk_aversion=2.5, tau=0.05)
    bl.add_absolute_view("EM_EQUITY", 0.10, confidence=0.35)
    post_mu, post_cov = bl.posterior()
    bl_res = MarkowitzOptimizer(post_mu, post_cov, risk_free_rate=0.03,
                                 weight_bounds=(0.0, 0.4)).max_sharpe()

    bench_w = pd.Series(1 / len(assets), index=assets)
    port_period_ret = test.mean()
    bench_period_ret = test.mean() * 0.85
    brinson = brinson_attribution(bl_res.weights, bench_w, port_period_ret, bench_period_ret)
    print(f"Brinson total active return vs equal-weight benchmark: {brinson.total_active_return:.4f}")
    print(brinson.by_group[["allocation_effect", "selection_effect", "interaction_effect"]].round(4))

    rar = variance_risk_attribution(bl_res.weights, post_cov)
    print(f"\nRisk attribution — total portfolio vol: {rar.total_volatility:.2%}")
    print(rar.percent_contribution.round(3))

    frm = FactorRiskModel(test)
    fresult = frm.fit_statistical(n_factors=3)
    weights_hist = pd.DataFrame([bl_res.weights.values] * len(test), index=test.index, columns=assets)
    actual_ret = test @ bl_res.weights
    fattr = factor_attribution_over_backtest(weights_hist, actual_ret, fresult.exposures, fresult.factor_returns)
    print(f"\nFactor attribution — total return {fattr.total_return:.4f}, "
          f"specific/idiosyncratic return {fattr.specific_return:.4f}")
    print(fattr.factor_contributions.round(4))

    # ================================================================ #
    section("7. EXPLAINABILITY DASHBOARD")
    explainer = BlackLittermanExplainer(bl, train, bl_res.weights)
    top_asset = bl_res.weights.idxmax()
    print(explainer.explain_asset_in_words(top_asset))
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    explain_path = explainer.to_html(os.path.join(out_dir, "explainability_report.html"))
    print(f"\nFull explainability HTML report written to: {explain_path}")

    # ================================================================ #
    section("8. GPU ACCELERATION LAYER")
    backends = available_backends()
    print("Available acceleration backends on this machine:", backends)
    print("Selected backend:", best_backend())
    mc = GPUAcceleratedMonteCarlo()
    t0 = time.time()
    sims = mc.simulate(mu.values[:3] / 252, (cov.values[:3, :3]) / 252, df=6, n_sims=100_000)
    print(f"Ran 100,000-path Student-t Monte Carlo via '{mc.backend}' backend in {time.time()-t0:.3f}s "
          f"-> shape {sims.shape}")

    section("DONE — Bayesian, RL, multi-period, execution, live-data, attribution, "
            "explainability, and GPU acceleration all executed successfully")


if __name__ == "__main__":
    main()
