"""
NEW #6 — Stress Testing & Tail-Risk Simulation Engine.

Two complementary approaches, both standard at institutional risk desks:

1. Historical scenario replay: apply the actual historical shock
   (e.g. 2008 GFC, 2020 COVID crash, 2022 rate-shock) — measured on a proxy
   index or supplied directly — to the *current* portfolio weights via each
   asset's estimated beta to that shock, producing a P&L estimate.

2. Fat-tailed Monte Carlo: real returns have fatter tails and more joint
   crash risk than a Gaussian assumption implies. This simulates portfolio
   outcomes using a multivariate Student-t distribution (fit via method-of-
   moments on kurtosis) instead of Normal, giving materially more realistic
   VaR/CVaR tail estimates — the standard fix for "my backtested VaR keeps
   getting breached way more than 5% of the time."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class StressResult:
    scenario_name: str
    portfolio_pnl_pct: float
    asset_pnl_pct: pd.Series


@dataclass
class MonteCarloResult:
    simulated_returns: np.ndarray      # n_sims-length array of portfolio returns
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    prob_loss_exceeds: dict            # {threshold: probability}
    degrees_of_freedom: float


# A small library of canonical historical shocks (approximate broad-market
# single-day/period moves) usable as generic stress proxies when
# asset-specific historical data for the exact event isn't available.
HISTORICAL_SCENARIOS = {
    "2008_gfc_crash": -0.42,      # ~Sept 2008-Mar 2009 equity drawdown
    "2020_covid_crash": -0.34,    # Feb-Mar 2020 drawdown
    "2022_rate_shock": -0.24,     # 2022 bear market
    "1987_black_monday": -0.20,   # single-day equivalent shock magnitude
    "dotcom_bust": -0.49,         # 2000-2002 drawdown
}


class StressTester:
    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns
        self.weights = weights.reindex(returns.columns).fillna(0.0)
        self.assets = list(returns.columns)

    def _asset_betas_to_market(self, market_proxy: pd.Series | None = None) -> pd.Series:
        proxy = market_proxy if market_proxy is not None else self.returns.mean(axis=1)
        var_m = proxy.var()
        return self.returns.apply(lambda col: col.cov(proxy) / var_m if var_m > 0 else 1.0)

    def historical_scenario(self, scenario_name: str,
                             market_proxy: pd.Series | None = None,
                             custom_shock: float | None = None) -> StressResult:
        """Apply a named historical market shock scaled by each asset's beta."""
        shock = custom_shock if custom_shock is not None else HISTORICAL_SCENARIOS.get(scenario_name)
        if shock is None:
            raise ValueError(f"Unknown scenario '{scenario_name}'. "
                              f"Available: {list(HISTORICAL_SCENARIOS)} or pass custom_shock.")
        betas = self._asset_betas_to_market(market_proxy)
        asset_pnl = betas * shock
        port_pnl = float((self.weights * asset_pnl).sum())
        return StressResult(scenario_name=scenario_name, portfolio_pnl_pct=port_pnl,
                             asset_pnl_pct=asset_pnl)

    def run_all_historical_scenarios(self, market_proxy: pd.Series | None = None) -> pd.DataFrame:
        rows = []
        for name in HISTORICAL_SCENARIOS:
            r = self.historical_scenario(name, market_proxy=market_proxy)
            rows.append({"scenario": name, "portfolio_pnl_pct": r.portfolio_pnl_pct})
        return pd.DataFrame(rows).sort_values("portfolio_pnl_pct")

    def student_t_monte_carlo(self, n_sims: int = 100_000, horizon_days: int = 1,
                               random_state: int = 42) -> MonteCarloResult:
        """Simulate portfolio returns from a fitted multivariate Student-t
        distribution (fat tails + joint crash risk), then compute VaR/CVaR.

        Degrees of freedom estimated via excess-kurtosis matching on the
        historical portfolio return series (method of moments):
            kurtosis_excess = 6 / (df - 4)   =>   df = 6/kurtosis_excess + 4
        Clipped to a sane range since noisy kurtosis estimates can blow up.
        """
        rng = np.random.default_rng(random_state)
        port_hist = (self.returns @ self.weights).values
        excess_kurt = max(stats.kurtosis(port_hist, fisher=True), 0.05)
        df = np.clip(6.0 / excess_kurt + 4.0, 3.0, 30.0)

        mu = self.returns.mean().values
        cov = self.returns.cov().values
        # scale covariance so that the t-distribution's implied covariance
        # (df/(df-2) * scale) matches the empirical covariance
        scale_matrix = cov * (df - 2) / df
        # ensure PSD
        scale_matrix = (scale_matrix + scale_matrix.T) / 2
        eigval, eigvec = np.linalg.eigh(scale_matrix)
        eigval = np.clip(eigval, 1e-12, None)
        scale_matrix = eigvec @ np.diag(eigval) @ eigvec.T

        n_assets = len(self.assets)
        g = rng.chisquare(df, size=n_sims) / df
        z = rng.multivariate_normal(np.zeros(n_assets), scale_matrix, size=n_sims)
        asset_sims = mu + z / np.sqrt(g)[:, None]

        # horizon scaling (sqrt-of-time on returns; simple approximation)
        if horizon_days != 1:
            asset_sims = asset_sims * np.sqrt(horizon_days)

        port_sims = asset_sims @ self.weights.values

        var_95 = -np.quantile(port_sims, 0.05)
        var_99 = -np.quantile(port_sims, 0.01)
        cvar_95 = -port_sims[port_sims <= -var_95].mean()
        cvar_99 = -port_sims[port_sims <= -var_99].mean()

        thresholds = [-0.05, -0.10, -0.15, -0.20]
        prob_loss = {f"loss_exceeds_{abs(t)*100:.0f}pct": float((port_sims < t).mean())
                     for t in thresholds}

        return MonteCarloResult(
            simulated_returns=port_sims, var_95=var_95, var_99=var_99,
            cvar_95=cvar_95, cvar_99=cvar_99, prob_loss_exceeds=prob_loss,
            degrees_of_freedom=float(df),
        )
