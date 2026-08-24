"""
Factor Attribution.

Given a factor model (statistical/PCA or fundamental — either result from
`advanced/factor_risk_model.py`), decomposes portfolio *return* over a
period into:

    portfolio_return = sum_k (portfolio_exposure_k * factor_return_k) + specific_return

i.e. how much of what the portfolio actually earned came from being
exposed to each systematic factor, versus stock-specific (idiosyncratic)
performance. This is the return-side complement to the risk-side factor
decomposition already available via `FactorRiskModel.reconstructed_cov`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FactorAttributionResult:
    factor_contributions: pd.Series      # contribution to total return, per factor
    specific_return: float
    total_return: float
    factor_contributions_by_period: pd.DataFrame


def factor_attribution(weights: pd.Series, factor_exposures: pd.DataFrame,
                        factor_returns: pd.DataFrame) -> FactorAttributionResult:
    """
    Parameters
    ----------
    weights : portfolio weights (assets)
    factor_exposures : assets x factors (betas), e.g. FactorRiskModel output
    factor_returns : time x factors, the realized factor return series over
                      the attribution period
    """
    assets = factor_exposures.index
    w = weights.reindex(assets).fillna(0.0).values

    # Portfolio-level factor exposure at each point in time is constant here
    # (static weights) — for a time-varying weights history, call this once
    # per rebalance period and sum (see `factor_attribution_over_backtest`).
    port_exposure = w @ factor_exposures.values  # length = n_factors

    factor_period_contrib = factor_returns.values * port_exposure[None, :]  # T x K
    factor_contributions = pd.Series(factor_period_contrib.sum(axis=0),
                                      index=factor_exposures.columns, name="factor_contribution")

    total_factor_return = factor_contributions.sum()
    # total portfolio return implied purely by the factor model over this window
    # (specific return here is a residual placeholder unless actual portfolio
    # returns are supplied — see the backtest-integrated version below)
    return FactorAttributionResult(
        factor_contributions=factor_contributions,
        specific_return=np.nan, total_return=float(total_factor_return),
        factor_contributions_by_period=pd.DataFrame(factor_period_contrib, index=factor_returns.index,
                                                      columns=factor_exposures.columns),
    )


def factor_attribution_over_backtest(weights_history: pd.DataFrame, actual_portfolio_returns: pd.Series,
                                       factor_exposures: pd.DataFrame, factor_returns: pd.DataFrame
                                       ) -> FactorAttributionResult:
    """Full attribution using actual realized portfolio returns, so the
    residual (specific/idiosyncratic return) is a genuine plug figure
    rather than undefined.
    """
    common_dates = weights_history.index.intersection(factor_returns.index)
    assets = factor_exposures.index
    contribs = []
    for date in common_dates:
        w = weights_history.loc[date].reindex(assets).fillna(0.0).values
        exposure = w @ factor_exposures.values
        contribs.append(exposure * factor_returns.loc[date].values)
    contrib_df = pd.DataFrame(contribs, index=common_dates, columns=factor_exposures.columns)

    factor_contributions = contrib_df.sum(axis=0)
    factor_contributions.name = "factor_contribution"
    total_factor_return = float(factor_contributions.sum())
    total_actual_return = float(actual_portfolio_returns.reindex(common_dates).sum())
    specific_return = total_actual_return - total_factor_return

    return FactorAttributionResult(
        factor_contributions=factor_contributions, specific_return=specific_return,
        total_return=total_actual_return, factor_contributions_by_period=contrib_df,
    )
