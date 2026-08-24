"""
Markowitz Mean-Variance Optimization.

Production notes
-----------------
- Uses SLSQP (sequential least squares) rather than a raw closed-form
  quadratic solve, because real portfolios have box constraints (position
  limits), sector/group constraints, and turnover constraints that break
  the closed-form solution.
- Numerically regularizes the covariance matrix (tiny ridge) to avoid
  failures on near-singular matrices, which happen constantly with real
  (correlated) asset universes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str = ""


class MarkowitzOptimizer:
    """Mean-variance optimizer supporting max-Sharpe, min-volatility,
    target-return, and target-risk formulations, plus the full efficient
    frontier, with realistic production constraints.

    Parameters
    ----------
    expected_returns : pd.Series, annualized
    cov_matrix : pd.DataFrame, annualized
    risk_free_rate : float
    weight_bounds : (low, high) applied to every asset by default
    sector_map : optional dict[str, str] asset -> group, for group constraints
    """

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.0, weight_bounds: tuple = (0.0, 1.0),
                 sector_map: dict | None = None, ridge: float = 1e-8):
        self.assets = list(expected_returns.index)
        self.mu = expected_returns.reindex(self.assets).values
        self.cov = cov_matrix.reindex(index=self.assets, columns=self.assets).values
        self.cov = self.cov + np.eye(len(self.assets)) * ridge
        self.rf = risk_free_rate
        self.n = len(self.assets)
        self.default_bounds = weight_bounds
        self.sector_map = sector_map or {}
        self._group_constraints: list[tuple[str, float, float]] = []

        # Feasibility pre-check: with N assets each capped at weight_bounds[1],
        # the maximum achievable total is N * upper_bound, which must be >= 1
        # for weights to sum to 1 at all. Without this check, an infeasible
        # bound (e.g. 3 assets each capped at 20%, max total 60%) fails deep
        # inside SLSQP with a cryptic "Positive directional derivative for
        # linesearch" message instead of an actionable one at construction time.
        lo, hi = weight_bounds
        if self.n * hi < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible weight_bounds: {self.n} assets each capped at "
                f"{hi:.1%} can sum to at most {self.n * hi:.1%}, which is less than "
                f"100%. Raise the upper bound (need >= {1.0 / self.n:.1%}) or add more assets."
            )
        if self.n * lo > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible weight_bounds: {self.n} assets each floored at "
                f"{lo:.1%} must sum to at least {self.n * lo:.1%}, which exceeds "
                f"100%. Lower the minimum bound (need <= {1.0 / self.n:.1%})."
            )

    # -- helpers ---------------------------------------------------------
    def add_group_constraint(self, group: str, min_weight: float, max_weight: float):
        """Constrain total weight allocated to a named group (e.g. 'Equity')."""
        self._group_constraints.append((group, min_weight, max_weight))
        return self

    def _portfolio_stats(self, w):
        ret = w @ self.mu
        vol = np.sqrt(max(w @ self.cov @ w, 1e-16))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _base_constraints(self):
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        for group, lo, hi in self._group_constraints:
            idx = [i for i, a in enumerate(self.assets) if self.sector_map.get(a) == group]
            if not idx:
                continue
            cons.append({"type": "ineq", "fun": (lambda w, idx=idx, hi=hi: hi - w[idx].sum())})
            cons.append({"type": "ineq", "fun": (lambda w, idx=idx, lo=lo: w[idx].sum() - lo)})
        return cons

    def _bounds(self, per_asset_bounds: dict | None = None):
        b = [self.default_bounds] * self.n
        if per_asset_bounds:
            for i, a in enumerate(self.assets):
                if a in per_asset_bounds:
                    b[i] = per_asset_bounds[a]
        return b

    def _solve(self, objective, bounds=None, extra_constraints=None, x0=None):
        bounds = bounds or self._bounds()
        cons = self._base_constraints() + (extra_constraints or [])
        x0 = x0 if x0 is not None else np.ones(self.n) / self.n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
        w = result.x
        w = np.clip(w, self.default_bounds[0], self.default_bounds[1])
        w = w / w.sum()
        ret, vol, sharpe = self._portfolio_stats(w)
        return OptimizationResult(
            weights=pd.Series(w, index=self.assets, name="weight"),
            expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
            success=result.success, message=result.message,
        )

    # -- public API --------------------------------------------------------
    def max_sharpe(self, per_asset_bounds: dict | None = None) -> OptimizationResult:
        def neg_sharpe(w):
            ret, vol, _ = self._portfolio_stats(w)
            return -(ret - self.rf) / vol if vol > 0 else 1e6
        return self._solve(neg_sharpe, bounds=self._bounds(per_asset_bounds))

    def min_volatility(self, per_asset_bounds: dict | None = None) -> OptimizationResult:
        def vol_fn(w):
            return w @ self.cov @ w
        return self._solve(vol_fn, bounds=self._bounds(per_asset_bounds))

    def target_return(self, target: float, per_asset_bounds: dict | None = None) -> OptimizationResult:
        cons = [{"type": "eq", "fun": lambda w: w @ self.mu - target}]
        return self._solve(lambda w: w @ self.cov @ w,
                            bounds=self._bounds(per_asset_bounds), extra_constraints=cons)

    def target_risk(self, target_vol: float, per_asset_bounds: dict | None = None) -> OptimizationResult:
        cons = [{"type": "eq", "fun": lambda w: np.sqrt(w @ self.cov @ w) - target_vol}]
        return self._solve(lambda w: -(w @ self.mu),
                            bounds=self._bounds(per_asset_bounds), extra_constraints=cons)

    def max_quadratic_utility(self, risk_aversion: float = 1.0,
                               per_asset_bounds: dict | None = None) -> OptimizationResult:
        """Classic Markowitz utility: maximize mu'w - (lambda/2) w'Sw."""
        def neg_utility(w):
            ret, _, _ = self._portfolio_stats(w)
            return -(w @ self.mu - 0.5 * risk_aversion * (w @ self.cov @ w))
        return self._solve(neg_utility, bounds=self._bounds(per_asset_bounds))

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Trace the full efficient frontier between min-vol and max-return portfolios."""
        min_vol_res = self.min_volatility()
        max_ret_idx = np.argmax(self.mu)
        max_possible_return = self.mu[max_ret_idx]
        targets = np.linspace(min_vol_res.expected_return, max_possible_return * 0.999, n_points)
        rows = []
        for t in targets:
            try:
                res = self.target_return(t)
                if res.success:
                    rows.append({"target_return": t, "return": res.expected_return,
                                 "volatility": res.volatility, "sharpe": res.sharpe_ratio,
                                 "weights": res.weights.to_dict()})
            except Exception:
                continue
        return pd.DataFrame(rows)
