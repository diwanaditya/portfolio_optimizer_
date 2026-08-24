"""
Hierarchical Bayesian Portfolio Model.

Extends Bayes-Stein (which shrinks every asset toward one global grand mean)
to a two-level hierarchy: assets are grouped (e.g. by sector/asset-class),
each group has its own group-level mean, and each group mean is itself
shrunk toward a global mean. This is standard hierarchical/partial-pooling
empirical Bayes (Efron-Morris style), solved in closed form via method-of-
moments variance-component estimation — no MCMC/PyMC dependency required,
which matters for a dependency-light production deployment.

The intuition: an EM-equity asset's return estimate should borrow strength
from *other EM-equity assets* (which share genuine common risk drivers)
more than from, say, government bonds. Pure global shrinkage (Bayes-Stein)
throws that structure away; pure no-pooling (raw sample means) ignores it
entirely. Hierarchical shrinkage sits in between, weighted by how much
each level of variance actually explains the data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class HierarchicalBayesResult:
    shrunk_mean: pd.Series
    group_means: pd.Series
    global_mean: float
    within_group_shrinkage: pd.Series   # per-asset shrinkage toward its group mean
    group_shrinkage: pd.Series          # per-group shrinkage toward the global mean


class HierarchicalBayesianPortfolio:
    """Two-level hierarchical empirical Bayes shrinkage of expected returns.

    Level 1 (asset -> group):    asset_mu_i ~ N(group_mu_g(i), tau_within^2)
    Level 2 (group -> global):   group_mu_g ~ N(global_mu, tau_between^2)

    Variance components (tau_within^2, tau_between^2, and the sampling
    variance of each estimate) are estimated via method-of-moments, then
    each level is shrunk using the standard Efron-Morris James-Stein
    formula: shrinkage = sampling_var / (sampling_var + between_var).
    """

    def __init__(self, returns: pd.DataFrame, group_map: dict[str, str],
                 periods_per_year: int = 252):
        self.returns = returns
        self.assets = list(returns.columns)
        self.group_map = group_map
        self.ppy = periods_per_year
        self.T = len(returns)

        missing = [a for a in self.assets if a not in group_map]
        if missing:
            raise ValueError(f"group_map missing entries for: {missing}")

    def solve(self) -> HierarchicalBayesResult:
        mu_hat = self.returns.mean() * self.ppy
        # per-asset sampling variance of the mean estimate: Var(mu_hat_i) = sigma_i^2 / T
        sampling_var = (self.returns.var() * self.ppy ** 2) / self.T  # annualized SE^2 of the mean estimate

        groups = pd.Series(self.group_map)
        unique_groups = groups.unique()

        # --- Level 1: shrink each asset toward its group mean ---
        group_raw_mean = {}
        within_shrinkage = {}
        asset_level1_shrunk = {}
        for g in unique_groups:
            members = groups[groups == g].index.tolist()
            group_mus = mu_hat[members]
            group_vars = sampling_var[members]
            grp_mean = group_mus.mean()
            # between-asset (within-group) variance component, method-of-moments,
            # floored at 0 (can be negative from noise in tiny samples)
            between_var = max(group_mus.var(ddof=1) - group_vars.mean(), 1e-8) if len(members) > 1 else 1e-8
            for a in members:
                shrink = group_vars[a] / (group_vars[a] + between_var)
                shrink = float(np.clip(shrink, 0.0, 1.0))
                asset_level1_shrunk[a] = shrink * grp_mean + (1 - shrink) * group_mus[a]
                within_shrinkage[a] = shrink
            group_raw_mean[g] = grp_mean

        # --- Level 2: shrink each group mean toward the global grand mean ---
        group_means_series = pd.Series(group_raw_mean)
        global_mean = float(group_means_series.mean())
        group_sampling_var = pd.Series(
            {g: sampling_var[groups[groups == g].index].mean() / max(len(groups[groups == g]), 1)
             for g in unique_groups}
        )
        between_group_var = max(group_means_series.var(ddof=1) - group_sampling_var.mean(), 1e-8) \
            if len(unique_groups) > 1 else 1e-8

        group_shrunk = {}
        group_shrinkage_out = {}
        for g in unique_groups:
            shrink = group_sampling_var[g] / (group_sampling_var[g] + between_group_var)
            shrink = float(np.clip(shrink, 0.0, 1.0))
            group_shrunk[g] = shrink * global_mean + (1 - shrink) * group_raw_mean[g]
            group_shrinkage_out[g] = shrink

        # --- Combine: final asset estimate = level-1 shrunk asset mean, but
        # recentered on the level-2-shrunk group mean (chain the two levels) ---
        final = {}
        for a in self.assets:
            g = self.group_map[a]
            group_shift = group_shrunk[g] - group_raw_mean[g]
            final[a] = asset_level1_shrunk[a] + group_shift

        return HierarchicalBayesResult(
            shrunk_mean=pd.Series(final, name="hierarchical_bayes_mean").reindex(self.assets),
            group_means=pd.Series(group_shrunk),
            global_mean=global_mean,
            within_group_shrinkage=pd.Series(within_shrinkage).reindex(self.assets),
            group_shrinkage=pd.Series(group_shrinkage_out),
        )
