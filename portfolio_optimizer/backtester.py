"""
NEW #7 — Walk-Forward Backtester.

The one thing that makes a portfolio-optimization repo "toy" vs "production"
is whether it can be honestly backtested. This engine:

  - Refits the chosen optimizer on a rolling/expanding lookback window,
    never touching future data (no look-ahead bias).
  - Rebalances on a configurable schedule (e.g. monthly, quarterly).
  - Charges realistic transaction costs (bps per unit turnover) and
    optionally a turnover cap / no-trade band to reduce needless churn.
  - Reports standard performance tearsheet metrics + a full equity curve.

Works with *any* callable optimizer strategy — pass in a function
`(returns_window) -> weights: pd.Series` and it plugs in Markowitz,
Black-Litterman, Risk Parity, HRP, CVaR, or any advanced module above.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    turnover_history: pd.Series
    transaction_costs: pd.Series
    metrics: dict


class WalkForwardBacktester:
    def __init__(self, returns: pd.DataFrame,
                 strategy_fn: Callable[[pd.DataFrame], pd.Series],
                 lookback_periods: int = 252, rebalance_every: int = 21,
                 transaction_cost_bps: float = 10.0, no_trade_band: float = 0.0,
                 expanding_window: bool = False, initial_capital: float = 1.0):
        """
        Parameters
        ----------
        returns : full historical returns panel
        strategy_fn : callable taking a returns DataFrame (the lookback window)
                      and returning target weights as a pd.Series
        lookback_periods : rolling (or minimum, if expanding) estimation window
        rebalance_every : periods between rebalances (21 ~ monthly for daily data)
        transaction_cost_bps : one-way cost in basis points per unit of turnover
        no_trade_band : ignore weight drifts smaller than this before trading
                        (reduces needless turnover from noise-level rebalances)
        expanding_window : if True, lookback grows from the start rather than rolling
        """
        self.returns = returns
        self.strategy_fn = strategy_fn
        self.lookback = lookback_periods
        self.rebalance_every = rebalance_every
        self.cost_bps = transaction_cost_bps / 10_000.0
        self.no_trade_band = no_trade_band
        self.expanding = expanding_window
        self.initial_capital = initial_capital

    def run(self) -> BacktestResult:
        dates = self.returns.index
        assets = self.returns.columns
        n = len(dates)

        current_weights = pd.Series(0.0, index=assets)
        weights_log = []
        turnover_log = []
        cost_log = []
        port_returns = pd.Series(0.0, index=dates)

        rebalance_points = list(range(self.lookback, n, self.rebalance_every))

        for t in range(self.lookback, n):
            date = dates[t]

            if t in rebalance_points:
                start = 0 if self.expanding else t - self.lookback
                window = self.returns.iloc[start:t]
                try:
                    target = self.strategy_fn(window)
                    target = target.reindex(assets).fillna(0.0)
                    target = target.clip(lower=0)
                    if target.sum() > 0:
                        target = target / target.sum()
                    else:
                        target = current_weights
                except Exception:
                    target = current_weights

                drift = (target - current_weights).abs()
                trade_mask = drift > self.no_trade_band
                new_weights = current_weights.copy()
                new_weights[trade_mask] = target[trade_mask]
                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                turnover = (new_weights - current_weights).abs().sum()
                cost = turnover * self.cost_bps
                current_weights = new_weights
                turnover_log.append((date, turnover))
                cost_log.append((date, cost))
            else:
                cost = 0.0

            day_return = (current_weights * self.returns.iloc[t]).sum() - cost
            port_returns.iloc[t] = day_return
            weights_log.append((date, current_weights.copy()))

            # drift weights forward with realized returns (buy-and-hold between rebalances)
            grown = current_weights * (1 + self.returns.iloc[t])
            if grown.sum() > 0:
                current_weights = grown / grown.sum()

        valid_returns = port_returns.iloc[self.lookback:]
        equity_curve = self.initial_capital * (1 + valid_returns).cumprod()

        weights_df = pd.DataFrame({d: w for d, w in weights_log}).T
        turnover_series = pd.Series(dict(turnover_log))
        cost_series = pd.Series(dict(cost_log))

        metrics = self._compute_metrics(valid_returns, equity_curve, turnover_series)

        return BacktestResult(
            equity_curve=equity_curve, returns=valid_returns, weights_history=weights_df,
            turnover_history=turnover_series, transaction_costs=cost_series, metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(returns: pd.Series, equity: pd.Series, turnover: pd.Series,
                          periods_per_year: int = 252) -> dict:
        if len(returns) == 0 or returns.std() == 0:
            return {}
        ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        downside = returns[returns < 0]
        sortino = ann_return / (downside.std() * np.sqrt(periods_per_year)) if len(downside) > 0 else np.nan
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        max_dd = drawdown.min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
        var_95 = -np.quantile(returns, 0.05)
        cvar_95 = -returns[returns <= -var_95].mean() if (returns <= -var_95).any() else var_95
        avg_turnover = turnover.mean() if len(turnover) else 0.0
        return {
            "annualized_return": ann_return, "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown": max_dd,
            "calmar_ratio": calmar, "var_95": var_95, "cvar_95": cvar_95,
            "avg_rebalance_turnover": avg_turnover, "n_periods": len(returns),
        }
