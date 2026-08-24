"""
Almgren-Chriss Optimal Execution Model (Almgren & Chriss, 2000/2001).

A flat basis-point transaction cost (as used in the backtester's default
mode) is a reasonable first-order approximation but hides two structurally
different costs that behave very differently as trade size grows:

  - **Temporary impact**: the price concession you pay for demanding
    immediate liquidity (crossing the spread, walking the book). This
    reverts after your trade — it does not permanently move the price.
    Modeled as increasing (here, linearly) in trading *rate*.

  - **Permanent impact**: the lasting price shift your trading itself
    causes (information leakage / inventory effects). This does NOT
    revert, and scales with total *quantity* traded, not rate.

Almgren-Chriss finds the trade execution schedule (how to break a large
order into smaller pieces over an execution window) that minimizes
expected cost plus a risk-aversion penalty on execution shortfall
variance — trading faster reduces exposure to price risk during execution
but costs more in temporary impact; trading slower does the opposite.

This module provides:
  1. `optimal_execution_trajectory` — the closed-form AC optimal trading
     trajectory for liquidating/acquiring a position over N sub-periods.
  2. `AlmgrenChrissCostModel` — a cost-per-trade function pluggable into
     the backtester or multi-period optimizer as an alternative to the
     flat-bps model, given each asset's volatility and liquidity
     parameters (average daily volume, or an assumed impact coefficient).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ExecutionTrajectory:
    holdings_schedule: np.ndarray     # length N+1: shares held at each time step (X_0 ... X_N)
    trade_schedule: np.ndarray        # length N: shares traded in each interval
    expected_cost: float              # total expected execution cost ($ or return-space, per input units)
    cost_variance: float              # variance of execution cost (risk term)


def optimal_execution_trajectory(total_shares: float, n_periods: int, total_time: float,
                                  volatility: float, temporary_impact_eta: float,
                                  permanent_impact_gamma: float,
                                  risk_aversion: float = 1e-6) -> ExecutionTrajectory:
    """
    Closed-form Almgren-Chriss optimal trajectory.

    Parameters
    ----------
    total_shares : X_0, total position to execute (positive = sell/liquidate,
                   negative = buy/acquire — the math is symmetric)
    n_periods : number of discrete trading intervals (N)
    total_time : total execution horizon in years (e.g. 1/252 for one trading day)
    volatility : annualized volatility of the asset's price
    temporary_impact_eta : eta, temporary impact coefficient ($ per share^2/time,
                           roughly: cost rate = eta * (shares traded / tau))
    permanent_impact_gamma : gamma, permanent impact coefficient ($ per share^2)
    risk_aversion : lambda, risk-aversion parameter trading off cost vs
                    variance of execution shortfall. lambda -> 0 recovers
                    the linear (VWAP-like) trajectory; larger lambda front-
                    loads execution to reduce price-risk exposure.

    Returns
    -------
    ExecutionTrajectory with the holdings path X_0...X_N, per-interval
    trades, and expected cost / cost variance of the optimal schedule.
    """
    tau = total_time / n_periods
    sigma = volatility
    eta_tilde = temporary_impact_eta - 0.5 * permanent_impact_gamma * tau
    if eta_tilde <= 0:
        eta_tilde = temporary_impact_eta * 0.5  # numerical safety floor

    kappa_bar_sq = (risk_aversion * sigma ** 2) / max(eta_tilde, 1e-12)
    kappa = np.arccosh(0.5 * kappa_bar_sq * tau ** 2 + 1) / tau if kappa_bar_sq > 0 else 1e-9
    kappa = max(kappa, 1e-9)

    j = np.arange(0, n_periods + 1)
    if risk_aversion <= 1e-12:
        # degenerates to the linear (uniform-rate) trajectory as lambda -> 0
        holdings = total_shares * (1 - j / n_periods)
    else:
        holdings = total_shares * (np.sinh(kappa * (n_periods - j)) / np.sinh(kappa * n_periods))

    trades = -np.diff(holdings)  # shares sold (positive) in each interval

    # Expected cost: permanent impact (path-independent) + temporary impact (path-dependent)
    expected_cost = 0.5 * permanent_impact_gamma * total_shares ** 2
    expected_cost += eta_tilde / tau * np.sum(trades ** 2)

    # Variance of execution shortfall from price risk during the holding period
    cost_variance = sigma ** 2 * tau * np.sum(holdings[:-1] ** 2)

    return ExecutionTrajectory(holdings_schedule=holdings, trade_schedule=trades,
                                expected_cost=float(expected_cost), cost_variance=float(cost_variance))


class AlmgrenChrissCostModel:
    """Per-asset Almgren-Chriss cost model usable as a drop-in replacement
    for the flat-bps cost assumption in the backtester / multi-period
    optimizer. Estimates eta (temporary) and gamma (permanent) impact
    coefficients from each asset's volatility and average daily volume
    (ADV) using standard microstructure rules of thumb when explicit
    calibration data isn't available:

        gamma_i ~ c_perm * sigma_i / ADV_i
        eta_i   ~ c_temp * sigma_i / ADV_i

    (Both impact coefficients scale with volatility-per-unit-liquidity —
    the standard "impact is bigger in illiquid, volatile names" intuition;
    the two constants let you calibrate temporary vs permanent impact
    magnitude if you have TCA (transaction cost analysis) data to fit to.)
    """

    def __init__(self, volatility: pd.Series, average_daily_volume: pd.Series,
                 c_temp: float = 0.1, c_perm: float = 0.05):
        self.assets = list(volatility.index)
        self.sigma = volatility.reindex(self.assets)
        self.adv = average_daily_volume.reindex(self.assets).clip(lower=1.0)
        self.eta = c_temp * self.sigma / self.adv
        self.gamma = c_perm * self.sigma / self.adv

    def trade_cost(self, asset: str, shares_traded: float, participation_rate: float = 0.1) -> float:
        """Estimated $ cost of executing `shares_traded` of `asset`, assuming
        the trade is spread out so that it represents `participation_rate`
        of ADV per period (higher participation = faster, costlier execution).
        """
        eta_i, gamma_i = self.eta[asset], self.gamma[asset]
        trade_rate = abs(shares_traded) * participation_rate
        temporary_cost = eta_i * trade_rate * abs(shares_traded)
        permanent_cost = 0.5 * gamma_i * shares_traded ** 2
        return float(temporary_cost + permanent_cost)

    def portfolio_trade_cost(self, trades_shares: pd.Series, participation_rate: float = 0.1) -> float:
        return sum(self.trade_cost(a, trades_shares.get(a, 0.0), participation_rate)
                   for a in self.assets)

    def implied_bps_cost(self, asset: str, shares_traded: float, price: float,
                          participation_rate: float = 0.1) -> float:
        """Convenience: express the AC dollar cost as an equivalent bps-of-
        notional cost, so it can be compared directly against the flat-bps
        assumption used elsewhere in the codebase."""
        notional = abs(shares_traded) * price
        if notional == 0:
            return 0.0
        cost = self.trade_cost(asset, shares_traded, participation_rate)
        return cost / notional * 10_000
