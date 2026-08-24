"""
NEW #1 — Entropy Pooling (Meucci, 2008).

Black-Litterman only lets you express *linear* views on expected returns
under a joint-normal assumption. Entropy Pooling generalizes this to:
  - views on volatility, correlation, skewness, or any moment
  - views on non-normal, scenario-based (historical or simulated) distributions
  - views that are *inequalities* ("vol will be higher than X"), not just equalities

It works by re-weighting historical/simulated scenarios (each scenario keeps
its outcome, only its *probability* changes) so that the posterior
distribution satisfies your views while staying maximally close — in
relative-entropy (KL-divergence) terms — to your prior. This is the single
most flexible view-blending framework used in modern quant risk management
(literally the technique behind Meucci's "Fully Flexible Views" at his
former shop and now widely used across systematic funds).

Solved via convex duality: minimizing KL-divergence under linear moment
constraints reduces to an unconstrained (small, K-dimensional) dual
optimization, which is fast even with thousands of scenarios.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class MomentView:
    """A view expressed as E[f(X)] {'=', '>=', '<='} value.
    `func` maps a scenario matrix (T x N) to a T-length vector — e.g.
    `lambda R: R[:, i]` for a mean view on asset i, or
    `lambda R: R[:, i] * R[:, j]` for a co-movement / correlation-flavored view.
    """
    func: callable
    value: float
    kind: str = "="   # "=", ">=", "<="


class EntropyPooling:
    def __init__(self, returns: pd.DataFrame, prior_probs: np.ndarray | None = None):
        self.returns = returns
        self.assets = list(returns.columns)
        self.R = returns.values
        self.T = self.R.shape[0]
        self.p0 = prior_probs if prior_probs is not None else np.ones(self.T) / self.T
        self.views: list[MomentView] = []

    def add_view(self, func: callable, value: float, kind: str = "="):
        self.views.append(MomentView(func=func, value=value, kind=kind))
        return self

    def add_mean_view(self, asset: str, value: float, kind: str = "="):
        i = self.assets.index(asset)
        return self.add_view(lambda R, i=i: R[:, i], value, kind)

    def add_volatility_view(self, asset: str, annualized_vol: float, kind: str = "="):
        i = self.assets.index(asset)
        target_var = (annualized_vol ** 2) / 252
        return self.add_view(lambda R, i=i: (R[:, i] - R[:, i].mean()) ** 2, target_var, kind)

    def add_correlation_view(self, asset_a: str, asset_b: str, correlation: float, kind: str = "="):
        i, j = self.assets.index(asset_a), self.assets.index(asset_b)

        def func(R):
            xa, xb = R[:, i], R[:, j]
            xa_c, xb_c = xa - xa.mean(), xb - xb.mean()
            # normalized so E[f] under posterior directly targets correlation
            denom = (xa_c.std() * xb_c.std()) + 1e-12
            return (xa_c * xb_c) / denom
        return self.add_view(func, correlation, kind)

    def solve(self, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
        """Returns posterior scenario probabilities (length T, sums to 1)."""
        if not self.views:
            return self.p0.copy()

        F = np.column_stack([v.func(self.R) for v in self.views])  # T x K
        values = np.array([v.value for v in self.views])
        kinds = [v.kind for v in self.views]

        log_p0 = np.log(np.clip(self.p0, 1e-300, None))

        def posterior_from_lambda(lam):
            log_p = log_p0 + F @ lam
            log_p -= log_p.max()
            p = np.exp(log_p)
            return p / p.sum()

        def dual_objective(lam):
            log_p = log_p0 + F @ lam
            m = log_p.max()
            log_z = m + np.log(np.sum(np.exp(log_p - m)))
            return log_z - lam @ values

        def dual_grad(lam):
            p = posterior_from_lambda(lam)
            return F.T @ p - values

        k = len(self.views)
        lam0 = np.zeros(k)
        bounds = []
        for kind in kinds:
            if kind == "=":
                bounds.append((None, None))
            elif kind == ">=":
                bounds.append((0, None))
            elif kind == "<=":
                bounds.append((None, 0))

        result = minimize(dual_objective, lam0, jac=dual_grad, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": max_iter, "ftol": tol})
        return posterior_from_lambda(result.x)

    def posterior_moments(self) -> tuple[pd.Series, pd.DataFrame]:
        """Convenience: posterior mean vector & covariance matrix implied by
        the reweighted scenario probabilities — plug straight into Markowitz/CVaR.
        """
        p = self.solve()
        mu = pd.Series(self.R.T @ p, index=self.assets) * 252
        centered = self.R - (p @ self.R)
        cov = (centered.T * p) @ centered * 252
        return mu, pd.DataFrame(cov, index=self.assets, columns=self.assets)

    def effective_sample_size(self, p: np.ndarray | None = None) -> float:
        """Diagnostic: how many 'effective' scenarios remain after reweighting.
        Low ESS (relative to T) signals your views are aggressively distorting
        history — a useful production sanity check before trusting the output.
        """
        p = p if p is not None else self.solve()
        return float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-300, None)))))
