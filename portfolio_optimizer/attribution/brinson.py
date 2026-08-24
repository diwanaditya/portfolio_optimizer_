"""
Brinson Attribution (Brinson, Hood & Beebower, 1986; Brinson & Fachler, 1985).

Decomposes a portfolio's active return (portfolio return - benchmark
return) over a period into the three classic effects:

  - **Allocation effect**: return from over/underweighting a sector/asset
    class relative to the benchmark, regardless of security selection
    within it.
  - **Selection effect**: return from picking better/worse-performing
    securities within a sector, holding the allocation weight fixed.
  - **Interaction effect**: the cross-term capturing that allocation and
    selection decisions aren't actually independent (a large allocation
    tilt amplifies the impact of good/bad selection within it).

This is *the* standard attribution framework used to explain a fund's
performance to LPs/allocators — "we beat the benchmark by 140bps: 90bps
came from being overweight EM equity (allocation), and 50bps came from
picking better EM names than the benchmark (selection)."
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass
class BrinsonResult:
    allocation_effect: pd.Series
    selection_effect: pd.Series
    interaction_effect: pd.Series
    total_active_return: float
    by_group: pd.DataFrame


def brinson_attribution(portfolio_weights: pd.Series, benchmark_weights: pd.Series,
                         portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> BrinsonResult:
    """
    All four inputs are indexed by group (sector/asset-class/security) for
    a single period. For multi-period attribution, compute this per period
    and compound/sum the effects (see `multi_period_brinson`).

    Formulas (Brinson-Fachler, benchmark-relative allocation):
        Allocation_g  = (w_p,g - w_b,g) * (R_b,g - R_b_total)
        Selection_g   = w_b,g * (R_p,g - R_b,g)
        Interaction_g = (w_p,g - w_b,g) * (R_p,g - R_b,g)
    """
    groups = portfolio_weights.index.union(benchmark_weights.index)
    wp = portfolio_weights.reindex(groups).fillna(0.0)
    wb = benchmark_weights.reindex(groups).fillna(0.0)
    rp = portfolio_returns.reindex(groups).fillna(0.0)
    rb = benchmark_returns.reindex(groups).fillna(0.0)

    r_b_total = (wb * rb).sum()

    allocation = (wp - wb) * (rb - r_b_total)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)

    total_active = float((wp * rp).sum() - (wb * rb).sum())

    by_group = pd.DataFrame({
        "portfolio_weight": wp, "benchmark_weight": wb,
        "portfolio_return": rp, "benchmark_return": rb,
        "allocation_effect": allocation, "selection_effect": selection,
        "interaction_effect": interaction,
        "total_effect": allocation + selection + interaction,
    })

    return BrinsonResult(allocation_effect=allocation, selection_effect=selection,
                          interaction_effect=interaction, total_active_return=total_active,
                          by_group=by_group)


def multi_period_brinson(portfolio_weights_history: pd.DataFrame, benchmark_weights_history: pd.DataFrame,
                          portfolio_returns_history: pd.DataFrame, benchmark_returns_history: pd.DataFrame
                          ) -> pd.DataFrame:
    """Run single-period Brinson attribution at every date and sum the
    effects across periods (arithmetic linking — the standard simple
    approach; geometric/compounded linking is a known refinement if you
    need multiplicative consistency with total compounded returns).
    """
    dates = portfolio_weights_history.index
    rows = []
    for date in dates:
        try:
            result = brinson_attribution(
                portfolio_weights_history.loc[date], benchmark_weights_history.loc[date],
                portfolio_returns_history.loc[date], benchmark_returns_history.loc[date],
            )
            rows.append({
                "date": date,
                "allocation": result.allocation_effect.sum(),
                "selection": result.selection_effect.sum(),
                "interaction": result.interaction_effect.sum(),
                "total_active": result.total_active_return,
            })
        except KeyError:
            continue
    df = pd.DataFrame(rows).set_index("date")
    df.loc["TOTAL"] = df.sum()
    return df
