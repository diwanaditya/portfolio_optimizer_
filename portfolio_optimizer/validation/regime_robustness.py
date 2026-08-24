"""
Robustness Across Market Regimes.

A strategy that looks great on one historical window (e.g. a long calm
bull market) can fail badly in a different regime (a sharp crash, a
choppy sideways market, a slow grinding bear market, a high-inflation/
high-rate regime). This module systematically tests a strategy across
MULTIPLE distinct synthetic regimes (not just multiple random seeds of
the SAME regime, which is what the bootstrap/multi-seed tests elsewhere
in this repo do) and reports the worst-case and dispersion of performance
— because a strategy's median performance across regimes matters much
less than whether it has a regime where it blows up.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class RegimeTestResult:
    regime_name: str
    metrics: dict


@dataclass
class RegimeRobustnessReport:
    per_regime_results: list
    summary_table: pd.DataFrame
    worst_regime: str
    worst_case_sharpe: float
    dispersion_sharpe: float  # std dev of Sharpe across regimes -- lower is more robust


def generate_regime_scenarios(n_assets: int = 6, n_days: int = 400, seed: int = 0) -> dict:
    """Generates several structurally DIFFERENT synthetic market regimes
    (not just different random seeds of the same generating process):
      - bull_low_vol: steady positive drift, low volatility
      - bear_market: persistent negative drift
      - high_vol_choppy: zero drift, high volatility, no trend
      - crash_recovery: sharp drawdown followed by a V-shaped recovery
      - stagflation: low/negative real returns, elevated correlations (everything sells off together)
    """
    rng = np.random.default_rng(seed)
    assets = [f"Asset_{i}" for i in range(n_assets)]
    scenarios = {}

    # Bull, low vol
    drift = rng.uniform(0.0003, 0.0006, size=n_assets)
    vol = rng.uniform(0.006, 0.010, size=n_assets)
    noise = rng.standard_normal((n_days, n_assets)) * vol + drift
    scenarios["bull_low_vol"] = pd.DataFrame(noise, columns=assets)

    # Bear market
    drift = rng.uniform(-0.0008, -0.0002, size=n_assets)
    vol = rng.uniform(0.012, 0.018, size=n_assets)
    noise = rng.standard_normal((n_days, n_assets)) * vol + drift
    scenarios["bear_market"] = pd.DataFrame(noise, columns=assets)

    # High-vol choppy, no trend
    vol = rng.uniform(0.02, 0.03, size=n_assets)
    noise = rng.standard_t(5, size=(n_days, n_assets)) * vol / np.sqrt(5 / 3)
    scenarios["high_vol_choppy"] = pd.DataFrame(noise, columns=assets)

    # Crash + V-shaped recovery
    base_vol = rng.uniform(0.008, 0.012, size=n_assets)
    noise = rng.standard_normal((n_days, n_assets)) * base_vol
    crash_point = n_days // 2
    crash_len = n_days // 10
    noise[crash_point:crash_point + crash_len] -= 0.02
    noise[crash_point + crash_len:crash_point + 2 * crash_len] += 0.018
    scenarios["crash_recovery"] = pd.DataFrame(noise, columns=assets)

    # Stagflation: near-zero/negative drift, elevated cross-asset correlation
    common = rng.standard_normal(n_days) * 0.012
    idio = rng.standard_normal((n_days, n_assets)) * 0.006
    noise = -0.0001 + 0.8 * common[:, None] + 0.4 * idio
    scenarios["stagflation_high_correlation"] = pd.DataFrame(noise, columns=assets)

    return scenarios


def evaluate_strategy_across_regimes(strategy_fn: Callable[[pd.DataFrame], pd.Series],
                                       scenarios: dict | None = None,
                                       periods_per_year: int = 252) -> RegimeRobustnessReport:
    """Runs `strategy_fn(returns) -> weights` on each regime scenario,
    applies the resulting static weights through that same scenario
    (in-sample allocation quality check -- a stricter out-of-sample walk-
    forward test is available via WalkForwardBacktester if needed), and
    reports per-regime performance plus overall robustness statistics.
    """
    scenarios = scenarios or generate_regime_scenarios()
    results = []
    for name, returns in scenarios.items():
        try:
            w = strategy_fn(returns)
            w = w.reindex(returns.columns).fillna(0.0)
            if w.sum() > 0:
                w = w / w.sum()
            port_returns = returns @ w
            ann_ret = port_returns.mean() * periods_per_year
            ann_vol = port_returns.std() * np.sqrt(periods_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            equity = (1 + port_returns).cumprod()
            max_dd = (equity / equity.cummax() - 1).min()
            metrics = {"annualized_return": ann_ret, "annualized_vol": ann_vol,
                       "sharpe_ratio": sharpe, "max_drawdown": max_dd}
        except Exception as e:
            metrics = {"annualized_return": np.nan, "annualized_vol": np.nan,
                       "sharpe_ratio": np.nan, "max_drawdown": np.nan, "error": str(e)}
        results.append(RegimeTestResult(regime_name=name, metrics=metrics))

    summary = pd.DataFrame({r.regime_name: r.metrics for r in results}).T
    sharpe_col = summary["sharpe_ratio"].dropna()
    worst_regime = sharpe_col.idxmin() if len(sharpe_col) else "N/A"
    worst_sharpe = sharpe_col.min() if len(sharpe_col) else np.nan
    dispersion = sharpe_col.std() if len(sharpe_col) > 1 else np.nan

    return RegimeRobustnessReport(per_regime_results=results, summary_table=summary,
                                    worst_regime=worst_regime, worst_case_sharpe=worst_sharpe,
                                    dispersion_sharpe=dispersion)
