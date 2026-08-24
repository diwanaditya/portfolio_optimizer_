"""
CVaR (Conditional Value-at-Risk) Optimization — Rockafellar & Uryasev (2000).

Why this matters for production: Markowitz variance penalizes upside and
downside symmetrically. CVaR optimization directly targets tail losses,
which is what actually blows up a fund. This is solved as a genuine linear
program on the historical (or simulated) return scenarios — no distributional
assumption required, and it scales to thousands of scenarios easily.

Also includes CDaR (Conditional Drawdown-at-Risk), the path-dependent
analogue that controls sustained drawdowns rather than single-period losses
— arguably more relevant to how allocators actually judge a track record.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from dataclasses import dataclass


@dataclass
class CVaRResult:
    weights: pd.Series
    expected_return: float
    cvar: float
    var: float
    success: bool


class CVaROptimizer:
    """Minimize portfolio CVaR at confidence level `alpha`, subject to a
    minimum expected return target, using historical return scenarios.

    Formulation (Rockafellar-Uryasev):
        min_{w, zeta, u}   zeta + 1/((1-alpha) T) * sum(u_t)
        s.t.               u_t >= -(r_t . w) - zeta,   u_t >= 0
                            sum(w) = 1, w >= 0 (or custom bounds)
                            mu . w >= target_return   (optional)

    `zeta` at the optimum is the portfolio VaR; the objective value is CVaR.
    """

    def __init__(self, returns: pd.DataFrame, alpha: float = 0.95,
                 weight_bounds: tuple = (0.0, 1.0)):
        self.returns = returns
        self.assets = list(returns.columns)
        self.R = returns.values  # T x N scenario matrix
        self.T, self.N = self.R.shape
        self.alpha = alpha
        self.bounds = weight_bounds
        self.mu = returns.mean().values * 252  # annualized for convenience

    def optimize(self, target_return: float | None = None,
                 max_weight_per_asset: float | None = None) -> CVaRResult:
        N, T = self.N, self.T
        # decision vector: [w (N), zeta (1), u (T)]
        n_vars = N + 1 + T
        c = np.zeros(n_vars)
        c[N + 1:] = 1.0 / ((1 - self.alpha) * T)
        c[N] = 1.0

        A_ub_rows = []
        b_ub = []

        # u_t + zeta + r_t . w >= 0  =>  -r_t.w - zeta - u_t <= 0
        for t in range(T):
            row = np.zeros(n_vars)
            row[:N] = -self.R[t, :]
            row[N] = -1.0
            row[N + 1 + t] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

        if max_weight_per_asset is not None:
            for i in range(N):
                row = np.zeros(n_vars)
                row[i] = 1.0
                A_ub_rows.append(row)
                b_ub.append(max_weight_per_asset)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        if target_return is not None:
            row = np.zeros((1, n_vars))
            row[0, :N] = -(self.returns.mean().values)  # per-period mean, not annualized
            per_period_target = (1 + target_return) ** (1 / 252) - 1
            A_ub = np.vstack([A_ub, row])
            b_ub = np.append(b_ub, -per_period_target)

        bounds = [self.bounds] * N + [(None, None)] + [(0, None)] * T

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")

        w = res.x[:N] if res.success else np.ones(N) / N
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else w
        zeta = res.x[N] if res.success else np.nan
        cvar = res.fun if res.success else np.nan
        port_returns = self.R @ w
        expected_ret = port_returns.mean() * 252

        return CVaRResult(
            weights=pd.Series(w, index=self.assets, name="weight"),
            expected_return=expected_ret, cvar=cvar, var=zeta, success=bool(res.success),
        )

    def efficient_frontier(self, n_points: int = 20) -> pd.DataFrame:
        lo, hi = self.mu.min(), self.mu.max()
        targets = np.linspace(lo, hi * 0.98, n_points)
        rows = []
        for t in targets:
            r = self.optimize(target_return=t)
            if r.success:
                rows.append({"target_return": t, "return": r.expected_return,
                             "cvar": r.cvar, "var": r.var, "weights": r.weights.to_dict()})
        return pd.DataFrame(rows)


class CDaROptimizer:
    """Conditional Drawdown-at-Risk optimization (Chekhlov, Uryasev & Zabarankin, 2005).
    Controls the average of the worst drawdowns along the cumulative wealth path,
    rather than single-period losses — closer to how investors actually feel pain.
    """

    def __init__(self, returns: pd.DataFrame, alpha: float = 0.95,
                 weight_bounds: tuple = (0.0, 1.0)):
        self.returns = returns
        self.assets = list(returns.columns)
        self.R = returns.values
        self.T, self.N = self.R.shape
        self.alpha = alpha
        self.bounds = weight_bounds

    def optimize(self, target_return: float | None = None) -> CVaRResult:
        N, T = self.N, self.T
        # variables: w(N), zeta(1), u(T) [auxiliary drawdown excess], and
        # cumulative wealth path is a function of w, so we linearize using
        # portfolio log-return path approx by simple cumulative sum (small returns).
        cum = np.cumsum(self.R, axis=0)  # T x N cumulative contribution per asset (linear approx)

        n_vars = N + 1 + T
        c = np.zeros(n_vars)
        c[N + 1:] = 1.0 / ((1 - self.alpha) * T)
        c[N] = 1.0

        A_ub_rows, b_ub = [], []
        for t in range(T):
            # drawdown_t(w) = max_{s<=t} cum_s.w - cum_t.w  (running peak minus current, linear in w)
            # constraint: u_t >= drawdown_t(w) - zeta  for all running peaks s <= t
            for s in range(t + 1):
                row = np.zeros(n_vars)
                row[:N] = -(cum[s] - cum[t])
                row[N] = -1.0
                row[N + 1 + t] = -1.0
                A_ub_rows.append(row)
                b_ub.append(0.0)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub)
        A_eq = np.zeros((1, n_vars)); A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        if target_return is not None:
            row = np.zeros((1, n_vars))
            row[0, :N] = -(self.returns.mean().values)
            per_period_target = (1 + target_return) ** (1 / 252) - 1
            A_ub = np.vstack([A_ub, row]); b_ub = np.append(b_ub, -per_period_target)

        bounds = [self.bounds] * N + [(None, None)] + [(0, None)] * T
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")
        w = res.x[:N] if res.success else np.ones(N) / N
        w = np.clip(w, 0, None); w = w / w.sum() if w.sum() > 0 else w
        port_returns = self.R @ w
        return CVaRResult(
            weights=pd.Series(w, index=self.assets, name="weight"),
            expected_return=port_returns.mean() * 252,
            cvar=res.fun if res.success else np.nan,
            var=res.x[N] if res.success else np.nan,
            success=bool(res.success),
        )
