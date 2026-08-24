"""
Risk Attribution.

Decomposes total portfolio risk (variance, volatility, or CVaR) into
per-asset and per-factor contributions using Euler's theorem for
positively-homogeneous risk measures — the same mathematical device
already used inside `RiskParity` and `CVaRRiskParity`, generalized here
into a standalone reporting/attribution module usable on *any* portfolio
(not just ones explicitly optimized for equal risk contribution).

Answers: "of my portfolio's total 14% annualized volatility, how much
comes from EM equity vs govt bonds vs gold?" and the factor-level
analogue: "how much of my risk is market-factor risk vs idiosyncratic?"
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RiskAttributionResult:
    total_volatility: float
    marginal_contribution: pd.Series      # d(vol)/d(w_i)
    component_contribution: pd.Series     # w_i * marginal_i, sums to total_volatility
    percent_contribution: pd.Series       # component / total, sums to 1.0


def variance_risk_attribution(weights: pd.Series, cov_matrix: pd.DataFrame) -> RiskAttributionResult:
    """Standard Euler decomposition of portfolio volatility by asset:
        vol(w) = sqrt(w' Sigma w)
        d(vol)/d(w_i) = (Sigma w)_i / vol(w)     (marginal contribution)
        component_i = w_i * marginal_i            (sums exactly to vol(w))
    """
    assets = list(weights.index)
    w = weights.reindex(assets).fillna(0.0).values
    cov = cov_matrix.reindex(index=assets, columns=assets).values
    port_var = w @ cov @ w
    vol = np.sqrt(max(port_var, 1e-16))
    marginal = (cov @ w) / vol
    component = w * marginal
    pct = component / vol

    return RiskAttributionResult(
        total_volatility=float(vol),
        marginal_contribution=pd.Series(marginal, index=assets),
        component_contribution=pd.Series(component, index=assets),
        percent_contribution=pd.Series(pct, index=assets),
    )


def factor_risk_attribution(weights: pd.Series, factor_exposures: pd.DataFrame,
                             factor_cov: pd.DataFrame, idiosyncratic_var: pd.Series
                             ) -> RiskAttributionResult:
    """Risk attribution at the FACTOR level (rather than asset level), using
    the factor-structured covariance Sigma = B F B' + D from
    `advanced/factor_risk_model.py`. Reports how much of total portfolio
    variance is explained by each systematic factor vs idiosyncratic risk
    (the latter reported as a single extra 'idiosyncratic' bucket).
    """
    assets = factor_exposures.index
    w = weights.reindex(assets).fillna(0.0).values
    B = factor_exposures.values          # N x K
    F = factor_cov.values                # K x K
    D = idiosyncratic_var.reindex(assets).fillna(0.0).values  # N

    port_factor_exposure = w @ B          # length K
    factor_var = port_factor_exposure @ F @ port_factor_exposure
    idio_var = float((w ** 2 * D).sum())
    total_var = factor_var + idio_var
    total_vol = np.sqrt(max(total_var, 1e-16))

    # Euler decomposition within factor space: marginal_k = (F @ exposure)_k / total_vol
    marginal_factor = (F @ port_factor_exposure) / total_vol
    component_factor = port_factor_exposure * marginal_factor

    names = list(factor_cov.columns) + ["idiosyncratic"]
    components = np.append(component_factor, idio_var / total_vol)
    marginals = np.append(marginal_factor, np.nan)  # marginal not well-defined for the aggregate idio bucket

    return RiskAttributionResult(
        total_volatility=float(total_vol),
        marginal_contribution=pd.Series(marginals, index=names),
        component_contribution=pd.Series(components, index=names),
        percent_contribution=pd.Series(components / total_vol, index=names),
    )


def risk_attribution_over_time(weights_history: pd.DataFrame, returns: pd.DataFrame,
                                 window: int = 60) -> pd.DataFrame:
    """Rolling risk attribution: recompute component risk contributions at
    each rebalance date using a trailing covariance window, tracking how
    each asset's share of total portfolio risk evolves — useful for
    spotting risk concentration building up before it shows up in returns.
    """
    dates = weights_history.index
    rows = []
    for date in dates:
        window_data = returns.loc[:date].tail(window)
        if len(window_data) < window // 2:
            continue
        cov = window_data.cov() * 252
        result = variance_risk_attribution(weights_history.loc[date], cov)
        row = result.percent_contribution.to_dict()
        row["_total_volatility"] = result.total_volatility
        rows.append(pd.Series(row, name=date))
    return pd.DataFrame(rows)
