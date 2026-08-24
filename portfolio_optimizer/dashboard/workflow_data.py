"""
Workflow Dashboard Data Assembly.

Pulls real numbers out of this repo's existing optimizer/backtest/
attribution modules and shapes them into exactly what
`workflow_ui.py`'s HTML template needs. Kept separate from HTML
rendering (same split as `generator.py`/`build_dashboard_data`) so every
number on the dashboard is independently testable Python, not JS that
silently swallows a bad computation.

Nothing here invents data -- every chart's numbers trace back to a real
call into `optimizers/`, `backtester.py`, `advanced/factor_risk_model.py`,
`attribution/risk_attribution.py`, or `explainability/`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


def build_health_metrics(portfolio_returns: pd.Series, weights: pd.Series,
                          returns: pd.DataFrame, alpha: float = 0.95,
                          periods_per_year: int = 252) -> dict:
    """The six headline numbers for the Portfolio Health Card."""
    r = portfolio_returns.dropna()
    ann_return = float(r.mean() * periods_per_year)
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + r).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())

    losses = -r
    var_thresh = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_thresh]
    cvar = float(tail.mean()) if len(tail) > 0 else np.nan

    return {
        "expected_return": ann_return, "volatility": ann_vol, "sharpe_ratio": sharpe,
        "max_drawdown": max_dd, "var_95": var_thresh, "cvar_95": cvar,
    }


def build_efficient_frontier_data(expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                                    current_weights: pd.Series | None = None,
                                    n_points: int = 30, risk_free_rate: float = 0.0) -> dict:
    """Real efficient frontier points (not illustrative curve-fitting) via
    the actual MarkowitzOptimizer, plus where the current portfolio and
    individual assets sit relative to it.
    """
    from ..optimizers.markowitz import MarkowitzOptimizer

    opt = MarkowitzOptimizer(expected_returns, cov_matrix, risk_free_rate=risk_free_rate)
    frontier = opt.efficient_frontier(n_points=n_points)
    frontier_points = [{"risk": float(row["volatility"]), "return": float(row["return"])}
                        for _, row in frontier.iterrows()]

    max_sharpe = opt.max_sharpe()
    tangency_point = {"risk": float(max_sharpe.volatility), "return": float(max_sharpe.expected_return)}

    asset_points = []
    for asset in expected_returns.index:
        vol = float(np.sqrt(cov_matrix.loc[asset, asset]))
        ret = float(expected_returns[asset])
        asset_points.append({"name": asset, "risk": vol, "return": ret})

    current_point = None
    if current_weights is not None:
        w = current_weights.reindex(expected_returns.index).fillna(0.0).values
        port_ret = float(w @ expected_returns.values)
        port_vol = float(np.sqrt(w @ cov_matrix.values @ w))
        current_point = {"risk": port_vol, "return": port_ret}

    return {"frontier": frontier_points, "tangency": tangency_point,
            "assets": asset_points, "current": current_point}


def build_allocation_treemap_data(weights: pd.Series, expected_returns: pd.Series | None = None) -> list:
    """Treemap cells: size = weight, color signal = expected return sign
    (green/red/black, matching the rest of this repo's fixed color
    contract) so the treemap communicates both allocation SIZE and return
    OUTLOOK in one view rather than just size alone.
    """
    cells = []
    for asset, w in weights.items():
        if abs(w) < 1e-6:
            continue
        er = float(expected_returns.get(asset, 0.0)) if expected_returns is not None else 0.0
        cells.append({"name": asset, "weight": float(w), "expected_return": er})
    cells.sort(key=lambda c: c["weight"], reverse=True)
    return cells


def build_rolling_metrics_data(returns: pd.Series, window: int = 63,
                                 periods_per_year: int = 252) -> dict:
    """Rolling Sharpe and rolling drawdown series -- real rolling-window
    computations on the actual return series, not smoothed/illustrative.
    """
    r = returns.dropna()
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)

    equity = (1 + r).cumprod()
    roll_dd = (equity / equity.cummax() - 1)

    sharpe_points = [{"t": str(t.date()) if hasattr(t, "date") else str(t), "v": float(v)}
                      for t, v in roll_sharpe.dropna().items()]
    dd_points = [{"t": str(t.date()) if hasattr(t, "date") else str(t), "v": float(v)}
                 for t, v in roll_dd.dropna().items()]
    return {"rolling_sharpe": sharpe_points, "rolling_drawdown": dd_points}


def build_factor_exposure_data(weights: pd.Series, returns: pd.DataFrame, n_factors: int = 3) -> dict:
    """Portfolio-level factor exposures via the real PCA statistical
    factor model, not hand-waved category buckets.
    """
    from ..advanced.factor_risk_model import FactorRiskModel

    frm = FactorRiskModel(returns)
    result = frm.fit_statistical(n_factors=n_factors)
    exposure = FactorRiskModel.portfolio_factor_exposure(weights, result.exposures)
    return {
        "factors": [{"name": name, "exposure": float(val)} for name, val in exposure.items()],
        "r_squared": {a: float(v) for a, v in result.r_squared.items()},
    }


def build_risk_contribution_data(weights: pd.Series, cov_matrix: pd.DataFrame) -> list:
    """Per-asset contribution to total portfolio risk via the real Euler
    decomposition (same math backing RiskParity, not a rough proxy).
    """
    from ..attribution.risk_attribution import variance_risk_attribution

    result = variance_risk_attribution(weights, cov_matrix)
    return [
        {"name": asset, "contribution_pct": float(result.percent_contribution[asset]),
         "weight": float(weights.get(asset, 0.0))}
        for asset in result.percent_contribution.index
    ]


def build_weight_change_explanations(returns_before: pd.DataFrame, returns_after: pd.DataFrame,
                                       weights_before: pd.Series, weights_after: pd.Series,
                                       top_n: int = 6) -> list:
    from ..explainability.weight_change_explainer import WeightChangeExplainer

    explainer = WeightChangeExplainer(returns_before, returns_after, weights_before, weights_after)
    return [asdict(e) for e in explainer.explain_all(top_n=top_n)]


def build_full_workflow_dashboard_data(
    returns: pd.DataFrame, weights: pd.Series, expected_returns: pd.Series,
    cov_matrix: pd.DataFrame, backtest_result=None, previous_weights: pd.Series | None = None,
    previous_returns: pd.DataFrame | None = None, portfolio_name: str = "ADC Portfolio",
) -> dict:
    """One call that assembles everything the dashboard needs. `backtest_result`
    (a `backtester.BacktestResult`, optional) supplies the rolling-metrics
    and health-card series if provided; otherwise those are computed
    directly from a static-weight application of `returns`/`weights`.
    """
    if backtest_result is not None:
        portfolio_returns = backtest_result.returns
    else:
        portfolio_returns = returns @ weights.reindex(returns.columns).fillna(0.0)

    data = {
        "portfolio_name": portfolio_name,
        "health": build_health_metrics(portfolio_returns, weights, returns),
        "efficient_frontier": build_efficient_frontier_data(expected_returns, cov_matrix, weights),
        "treemap": build_allocation_treemap_data(weights, expected_returns),
        "rolling": build_rolling_metrics_data(portfolio_returns),
        "factor_exposure": build_factor_exposure_data(weights, returns),
        "risk_contribution": build_risk_contribution_data(weights, cov_matrix),
    }

    if previous_weights is not None and previous_returns is not None:
        data["weight_changes"] = build_weight_change_explanations(
            previous_returns, returns, previous_weights, weights
        )
    else:
        data["weight_changes"] = []

    return data
