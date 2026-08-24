"""
Risk Parity family:
  1. Equal Risk Contribution (ERC) — classic risk parity via convex optimization
  2. Hierarchical Risk Parity (HRP) — Lopez de Prado (2016), no matrix inversion,
     robust to near-singular covariance matrices, no numerical optimizer needed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


class RiskParity:
    """Equal Risk Contribution portfolio: every asset contributes the same
    share of total portfolio variance. Optionally supports *risk budgets*
    (unequal target contributions) rather than strict equality.
    """

    def __init__(self, cov_matrix: pd.DataFrame, risk_budget: pd.Series | None = None,
                 weight_bounds: tuple = (0.0001, 1.0)):
        self.assets = list(cov_matrix.index)
        self.cov = cov_matrix.reindex(index=self.assets, columns=self.assets).values
        self.n = len(self.assets)
        if risk_budget is None:
            self.budget = np.ones(self.n) / self.n
        else:
            b = risk_budget.reindex(self.assets).values
            self.budget = b / b.sum()
        self.bounds = weight_bounds

    def _risk_contributions(self, w):
        port_var = w @ self.cov @ w
        marginal = self.cov @ w
        return w * marginal / port_var

    def solve(self) -> pd.Series:
        def objective(w):
            rc = self._risk_contributions(w)
            # minimize dispersion of (rc_i / budget_i) — pure risk parity when budget uniform
            target = rc.sum() * self.budget
            return np.sum((rc - target) ** 2)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [self.bounds] * self.n
        x0 = np.ones(self.n) / self.n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 2000, "ftol": 1e-16})
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        return pd.Series(w, index=self.assets, name="weight")

    def risk_contribution_report(self, weights: pd.Series) -> pd.DataFrame:
        w = weights.reindex(self.assets).values
        rc = self._risk_contributions(w)
        return pd.DataFrame({
            "weight": w, "risk_contribution": rc,
            "risk_contribution_pct": rc / rc.sum(), "target_budget": self.budget,
        }, index=self.assets)


class HierarchicalRiskParity:
    """Lopez de Prado's HRP (2016): clusters assets by correlation distance,
    quasi-diagonalizes the covariance matrix, then allocates via recursive
    bisection down the cluster tree. Big production advantage over
    mean-variance / ERC: no matrix inversion, so it never blows up on
    ill-conditioned or singular covariance matrices, and weights are far
    more stable out-of-sample.
    """

    def __init__(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame | None = None,
                 linkage_method: str = "single"):
        self.returns = returns
        self.assets = list(returns.columns)
        self.cov = (cov_matrix.reindex(index=self.assets, columns=self.assets)
                    if cov_matrix is not None else returns.cov())
        self.corr = returns.corr()
        self.linkage_method = linkage_method

    def _correlation_distance(self):
        return np.sqrt(0.5 * (1 - self.corr))

    def _tree_clustering(self):
        dist = self._correlation_distance()
        condensed = squareform(dist.values, checks=False)
        return linkage(condensed, method=self.linkage_method)

    @staticmethod
    def _get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _cluster_var(self, cov, items):
        sub_cov = cov.iloc[items, items]
        inv_diag = 1.0 / np.diag(sub_cov.values)
        w = inv_diag / inv_diag.sum()
        return w @ sub_cov.values @ w

    def _recursive_bisection(self, cov, sorted_items):
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]
        while clusters:
            clusters = [c[start:end] for c in clusters
                        for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
                        if len(c) > 1]
            for i in range(0, len(clusters), 2):
                c0 = clusters[i]
                c1 = clusters[i + 1] if i + 1 < len(clusters) else []
                if not c1:
                    continue
                var0 = self._cluster_var(cov, c0)
                var1 = self._cluster_var(cov, c1)
                alpha = 1 - var0 / (var0 + var1)
                weights[c0] *= alpha
                weights[c1] *= (1 - alpha)
        return weights

    def solve(self) -> pd.Series:
        link = self._tree_clustering()
        sorted_ix = self._get_quasi_diag(link)
        sorted_assets = [self.assets[i] for i in sorted_ix]
        cov_df = self.cov.loc[sorted_assets, sorted_assets]
        w = self._recursive_bisection(cov_df, list(range(len(sorted_assets))))
        w.index = sorted_assets
        w = w.reindex(self.assets)
        return (w / w.sum()).rename("weight")

    def linkage_matrix(self):
        """Expose the linkage matrix for dendrogram plotting."""
        return self._tree_clustering()
