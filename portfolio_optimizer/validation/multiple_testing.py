"""
Multiple-Testing-Corrected Strategy Comparison.

When comparing K strategies pairwise (e.g. this repo alone has ~15
optimizers — Markowitz, BL, Risk Parity, HRP, CVaR, Bayesian variants, RL
agents, etc.), running K*(K-1)/2 pairwise significance tests without
correction inflates the false-positive rate badly: at 15 strategies,
that's 105 pairwise tests, and at a naive 5% significance level you'd
expect ~5 "significant" differences purely by chance even if every
strategy had identical true Sharpe ratios.

This module implements:
  1. The Jobson & Korkie (1981) / Memmel (2003) test for whether two
     Sharpe ratios are statistically significantly different (accounts
     for the correlation between the two strategies' returns, which a
     naive two-sample t-test on Sharpe ratios ignores).
  2. Holm-Bonferroni and Benjamini-Hochberg (FDR) corrections applied
     across all pairwise comparisons, so the reported "significant"
     differences actually control the intended error rate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from itertools import combinations


@dataclass
class PairwiseTestResult:
    strategy_a: str
    strategy_b: str
    sharpe_a: float
    sharpe_b: float
    test_statistic: float
    p_value: float


def jobson_korkie_memmel_test(returns_a: pd.Series, returns_b: pd.Series,
                                periods_per_year: int = 252) -> tuple[float, float]:
    """Memmel's (2003) corrected Jobson-Korkie test for equality of two
    Sharpe ratios, accounting for the covariance between the two return
    series (they're often the same underlying assets with different
    weights, so their returns are correlated — ignoring that correlation,
    as a naive t-test would, gives the wrong test size).

    Returns (z_statistic, two_sided_p_value).
    """
    common_idx = returns_a.index.intersection(returns_b.index)
    a = returns_a.reindex(common_idx).values
    b = returns_b.reindex(common_idx).values
    T = len(a)

    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(ddof=1), b.std(ddof=1)
    sharpe_a = mu_a / sig_a * np.sqrt(periods_per_year) if sig_a > 0 else np.nan
    sharpe_b = mu_b / sig_b * np.sqrt(periods_per_year) if sig_b > 0 else np.nan

    sigma_ab = np.cov(a, b, ddof=1)[0, 1]
    rho = sigma_ab / (sig_a * sig_b) if sig_a > 0 and sig_b > 0 else 0.0

    # Memmel (2003) corrected asymptotic variance of (SR_a - SR_b)
    theta = (1 / T) * (
        2 - 2 * rho + 0.5 * (sharpe_a / np.sqrt(periods_per_year)) ** 2
        + 0.5 * (sharpe_b / np.sqrt(periods_per_year)) ** 2
        - (rho * (sharpe_a / np.sqrt(periods_per_year)) * (sharpe_b / np.sqrt(periods_per_year)))
    )
    theta = max(theta, 1e-12)

    z = (sharpe_a - sharpe_b) / np.sqrt(periods_per_year * theta) if theta > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def pairwise_sharpe_comparison(strategy_returns: dict) -> list[PairwiseTestResult]:
    """Run the Jobson-Korkie-Memmel test for every pair of strategies in
    `strategy_returns` (dict of name -> pd.Series of returns).
    """
    results = []
    names = list(strategy_returns.keys())
    for name_a, name_b in combinations(names, 2):
        ra, rb = strategy_returns[name_a], strategy_returns[name_b]
        z, p = jobson_korkie_memmel_test(ra, rb)
        sharpe_a = ra.mean() / ra.std() * np.sqrt(252) if ra.std() > 0 else np.nan
        sharpe_b = rb.mean() / rb.std() * np.sqrt(252) if rb.std() > 0 else np.nan
        results.append(PairwiseTestResult(
            strategy_a=name_a, strategy_b=name_b, sharpe_a=float(sharpe_a),
            sharpe_b=float(sharpe_b), test_statistic=z, p_value=p,
        ))
    return results


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down correction: controls the family-wise
    error rate (probability of ANY false positive) while being less
    conservative than plain Bonferroni. Returns a boolean array of which
    hypotheses are rejected (i.e. declared significant) at level alpha.
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    reject_sorted = np.zeros(m, dtype=bool)
    for i in range(m):
        threshold = alpha / (m - i)
        if sorted_p[i] <= threshold:
            reject_sorted[i] = True
        else:
            break  # Holm's procedure stops at the first non-rejection
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def benjamini_hochberg_correction(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction: controls the expected proportion
    of false positives among rejected hypotheses (less conservative than
    family-wise-error control, appropriate when you can tolerate a
    controlled fraction of false positives among many comparisons —
    common in exploratory strategy screening).
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    max_i = np.max(np.where(below)[0])
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[:max_i + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def multi_strategy_comparison_report(strategy_returns: dict, alpha: float = 0.05,
                                       correction: str = "holm") -> pd.DataFrame:
    """Full report: pairwise Sharpe comparisons across all strategies,
    with multiple-testing correction applied, so you get an honest answer
    to "which of these K strategies are ACTUALLY significantly different
    from each other" rather than an inflated list of false positives.
    """
    pairwise = pairwise_sharpe_comparison(strategy_returns)
    p_values = [r.p_value for r in pairwise]

    if correction == "holm":
        rejected = holm_bonferroni_correction(p_values, alpha)
    elif correction == "bh":
        rejected = benjamini_hochberg_correction(p_values, alpha)
    else:
        raise ValueError("correction must be 'holm' or 'bh'")

    rows = []
    for r, rej in zip(pairwise, rejected):
        rows.append({
            "strategy_a": r.strategy_a, "strategy_b": r.strategy_b,
            "sharpe_a": r.sharpe_a, "sharpe_b": r.sharpe_b,
            "sharpe_diff": r.sharpe_a - r.sharpe_b,
            "z_statistic": r.test_statistic, "raw_p_value": r.p_value,
            "significant_after_correction": bool(rej),
        })
    df = pd.DataFrame(rows)
    n_sig_raw = (df["raw_p_value"] < alpha).sum()
    n_sig_corrected = df["significant_after_correction"].sum()
    df.attrs["n_comparisons"] = len(df)
    df.attrs["n_significant_uncorrected"] = int(n_sig_raw)
    df.attrs["n_significant_corrected"] = int(n_sig_corrected)
    df.attrs["correction_method"] = correction
    return df
