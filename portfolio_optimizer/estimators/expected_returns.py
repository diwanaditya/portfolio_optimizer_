"""
Expected return estimators.

Raw sample means are famously the noisiest input in mean-variance
optimization (Merton, 1980; Michaud, 1989) — this module offers several
estimators so you're not stuck with the naive one.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def mean_historical_return(returns: pd.DataFrame, periods_per_year: int = 252,
                            compounding: bool = True) -> pd.Series:
    """Annualized historical mean return per asset.

    Parameters
    ----------
    returns : DataFrame of periodic (e.g. daily) simple returns, assets in columns.
    periods_per_year : trading periods per year (252 daily, 12 monthly, 52 weekly).
    compounding : if True, use geometric (CAGR-style) annualization; else arithmetic.
    """
    if compounding:
        growth = (1.0 + returns).prod()
        n_periods = returns.shape[0]
        return growth ** (periods_per_year / n_periods) - 1.0
    return returns.mean() * periods_per_year


def ewma_return(returns: pd.DataFrame, span: int = 180,
                 periods_per_year: int = 252) -> pd.Series:
    """Exponentially-weighted mean return — reacts faster to recent regime shifts
    than an equal-weighted historical average.
    """
    lam = 2.0 / (span + 1.0)
    weights = (1 - lam) ** np.arange(len(returns))[::-1]
    weights /= weights.sum()
    mu = returns.mul(weights, axis=0).sum()
    return mu * periods_per_year


def capm_return(returns: pd.DataFrame, market_returns: pd.Series,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> pd.Series:
    """CAPM-implied expected return: rf + beta * (E[Rm] - rf).

    Useful as a *prior* for Black-Litterman, and generally more stable
    than trailing sample means since it only requires estimating beta.
    """
    market_var = market_returns.var()
    betas = returns.apply(lambda col: col.cov(market_returns) / market_var)
    market_premium = market_returns.mean() * periods_per_year - risk_free_rate
    return risk_free_rate + betas * market_premium
