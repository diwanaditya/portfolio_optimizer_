"""
Bootstrap Confidence Intervals for portfolio performance metrics.

A single Sharpe ratio computed from one backtest is a point estimate with
real sampling uncertainty — it is not "the Sharpe ratio," it's one draw
from a distribution. This module computes block-bootstrap confidence
intervals for Sharpe, Sortino, max drawdown, CVaR, and arbitrary
user-supplied metrics, so results can be reported honestly as
"Sharpe = 1.2, 95% CI [0.7, 1.8]" rather than a bare, falsely-precise number.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class BootstrapCIResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_distribution: np.ndarray
    confidence_level: float


def _block_bootstrap_sample(returns: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    T = len(returns)
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, max(T - block_size, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
    idx = np.clip(idx, 0, T - 1)
    return returns[idx]


def bootstrap_metric_ci(returns: pd.Series, metric_fn: Callable[[np.ndarray], float],
                         n_bootstrap: int = 2000, block_size: int = 20,
                         confidence_level: float = 0.95, seed: int = 42) -> BootstrapCIResult:
    """Generic block-bootstrap CI for any scalar metric function of a return series.

    Parameters
    ----------
    returns : periodic return series
    metric_fn : function mapping a numpy array of returns -> a scalar metric
    n_bootstrap : number of bootstrap resamples
    block_size : block length for the block bootstrap (preserves
                 autocorrelation/volatility clustering, unlike iid resampling)
    confidence_level : e.g. 0.95 for a 95% CI
    """
    rng = np.random.default_rng(seed)
    r = returns.values
    point_estimate = metric_fn(r)

    distribution = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = _block_bootstrap_sample(r, block_size, rng)
        distribution[i] = metric_fn(sample)

    alpha = 1 - confidence_level
    lo = np.percentile(distribution, 100 * alpha / 2)
    hi = np.percentile(distribution, 100 * (1 - alpha / 2))

    return BootstrapCIResult(point_estimate=float(point_estimate), ci_lower=float(lo),
                              ci_upper=float(hi), bootstrap_distribution=distribution,
                              confidence_level=confidence_level)


# --- Standard metric functions, ready to plug into bootstrap_metric_ci --- #

def sharpe_ratio_metric(periods_per_year: int = 252, risk_free_rate: float = 0.0) -> Callable:
    def fn(r: np.ndarray) -> float:
        if r.std() == 0:
            return np.nan
        ann_ret = r.mean() * periods_per_year
        ann_vol = r.std() * np.sqrt(periods_per_year)
        return (ann_ret - risk_free_rate) / ann_vol
    return fn


def sortino_ratio_metric(periods_per_year: int = 252, risk_free_rate: float = 0.0) -> Callable:
    def fn(r: np.ndarray) -> float:
        downside = r[r < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.nan
        ann_ret = r.mean() * periods_per_year
        downside_vol = downside.std() * np.sqrt(periods_per_year)
        return (ann_ret - risk_free_rate) / downside_vol
    return fn


def max_drawdown_metric() -> Callable:
    def fn(r: np.ndarray) -> float:
        equity = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(equity)
        dd = equity / running_max - 1
        return float(dd.min())
    return fn


def cvar_metric(alpha: float = 0.95) -> Callable:
    def fn(r: np.ndarray) -> float:
        losses = -r
        var_thresh = np.quantile(losses, alpha)
        tail = losses[losses >= var_thresh]
        return float(tail.mean()) if len(tail) > 0 else np.nan
    return fn


def annualized_return_metric(periods_per_year: int = 252) -> Callable:
    def fn(r: np.ndarray) -> float:
        return float(np.prod(1 + r) ** (periods_per_year / len(r)) - 1)
    return fn


def full_metric_report(returns: pd.Series, n_bootstrap: int = 1000, block_size: int = 20,
                        confidence_level: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Convenience: compute bootstrap CIs for the standard suite of
    performance metrics in one call, returned as a tidy DataFrame.
    """
    metrics = {
        "annualized_return": annualized_return_metric(),
        "sharpe_ratio": sharpe_ratio_metric(),
        "sortino_ratio": sortino_ratio_metric(),
        "max_drawdown": max_drawdown_metric(),
        "cvar_95": cvar_metric(0.95),
    }
    rows = []
    for name, fn in metrics.items():
        result = bootstrap_metric_ci(returns, fn, n_bootstrap, block_size, confidence_level, seed)
        rows.append({
            "metric": name, "point_estimate": result.point_estimate,
            "ci_lower": result.ci_lower, "ci_upper": result.ci_upper,
            "ci_width": result.ci_upper - result.ci_lower,
        })
    return pd.DataFrame(rows).set_index("metric")
