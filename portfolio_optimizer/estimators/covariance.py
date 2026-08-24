"""
Covariance estimators.

Sample covariance is the other classic culprit behind unstable, corner-heavy
Markowitz weights. This module implements shrinkage estimators that pull the
noisy sample covariance toward a structured, low-variance target.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Plain annualized sample covariance matrix."""
    return returns.cov() * periods_per_year


def ledoit_wolf_shrinkage(returns: pd.DataFrame, periods_per_year: int = 252
                           ) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) constant-correlation shrinkage estimator.

    Shrinks the sample covariance matrix toward a structured target (the
    constant-correlation matrix implied by average pairwise correlation),
    with the shrinkage intensity chosen analytically to minimize expected
    Frobenius-norm estimation error. This is *the* standard fix used at
    every serious quant shop for the "N assets, not-much-more-than-N history"
    problem, and it never requires a solver or hyperparameter search.

    Returns
    -------
    (shrunk_covariance, shrinkage_intensity)
    """
    X = returns.values
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)

    sample = (X.T @ X) / T                      # sample covariance (unannualized)
    var = np.diag(sample)
    std = np.sqrt(var)
    _outer = np.outer(std, std)
    _outer[_outer == 0] = 1e-12
    corr = sample / _outer
    avg_corr = (corr.sum() - N) / (N * (N - 1)) if N > 1 else 0.0

    # Structured target: constant correlation matrix
    target = avg_corr * _outer
    np.fill_diagonal(target, var)

    # Frobenius distance between sample and target
    diff = sample - target
    delta = (diff ** 2).sum()

    # Estimate pi_hat: asymptotic variance of sample covariance entries
    y = X ** 2
    phi_mat = (y.T @ y) / T - sample ** 2
    phi = phi_mat.sum()

    # rho_hat term (cross term for the constant-correlation target), Ledoit-Wolf (2003/2004)
    theta_sum = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            term1 = ((X[:, i] ** 2 * X[:, i] * X[:, j]).mean() - sample[i, i] * sample[i, j])
            term2 = ((X[:, j] ** 2 * X[:, j] * X[:, i]).mean() - sample[j, j] * sample[i, j])
            if std[i] > 0 and std[j] > 0:
                theta_sum += avg_corr * 0.5 * (
                    (std[j] / std[i]) * term1 + (std[i] / std[j]) * term2
                )
    rho = np.diag(phi_mat).sum() + theta_sum

    kappa = (phi - rho) / delta if delta > 1e-12 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / T))

    shrunk = shrinkage * target + (1 - shrinkage) * sample
    shrunk_annualized = pd.DataFrame(shrunk * periods_per_year,
                                      index=returns.columns, columns=returns.columns)
    return shrunk_annualized, shrinkage


def ewma_covariance(returns: pd.DataFrame, span: int = 60,
                     periods_per_year: int = 252) -> pd.DataFrame:
    """RiskMetrics-style exponentially-weighted covariance — adapts to
    volatility clustering much faster than a flat lookback window.
    """
    lam = 2.0 / (span + 1.0)
    X = returns.values - returns.values.mean(axis=0, keepdims=True)
    T = X.shape[0]
    weights = (1 - lam) ** np.arange(T)[::-1]
    weights /= weights.sum()
    cov = (X * weights[:, None]).T @ X
    return pd.DataFrame(cov * periods_per_year, index=returns.columns, columns=returns.columns)


def semicovariance(returns: pd.DataFrame, benchmark: float = 0.0,
                    periods_per_year: int = 252) -> pd.DataFrame:
    """Downside semi-covariance: only co-movements below the benchmark count.
    Useful when the objective genuinely cares about downside risk (paired
    with CVaR / Sortino-style objectives) rather than symmetric variance.
    """
    downside = returns.clip(upper=benchmark) - benchmark
    return (downside.T @ downside) / len(returns) * periods_per_year
