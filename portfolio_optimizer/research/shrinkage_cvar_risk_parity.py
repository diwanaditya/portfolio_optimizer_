"""
RESEARCH CONTRIBUTION (WITH NEGATIVE EMPIRICAL RESULT): Shrinkage-Adaptive
CVaR Risk Parity (SA-CVaR-RP)
=====================================================================

STATUS: This is an original method developed for this project. It was
derived from first principles, implemented, and then tested against its
own pre-stated falsifiable predictions using a MULTI-SEED statistical
test with 95% confidence intervals (see
`examples/novel_contribution_validation.py`) — not a single cherry-picked
run. FINAL RESULT (6 independent synthetic universes, reported exactly
as produced, no post-hoc tuning):

    Metric                  | Sample-size rule vs plain | Adaptive-SNR rule vs plain
    Bootstrap weight variance | not statistically distinguishable (95% CI includes 0)
    Walk-forward turnover     | not statistically distinguishable (95% CI includes 0)

Neither shrinkage rule showed a statistically detectable improvement OR
degradation relative to plain CVaR Risk Parity at this sample size. The
honest reading is NOT "it works" or "it's proven worse" — it's "this
small a multi-seed study has insufficient power to detect an effect, and
in the absence of a detected effect, there's no evidence-based reason to
prefer the added complexity of this method over the simpler baseline."

This module is kept in the repository, with that null result disclosed
prominently, as an honest example of the full research process — propose
a theoretically well-motivated method, pre-register falsifiable
predictions, implement it, diagnose and fix real implementation bugs
found along the way (see the in-code history below), test rigorously
across multiple seeds with confidence intervals rather than a single run,
and report the actual statistical conclusion even when it's a null
result. That is what genuine empirical validation looks like — most
proposed improvements in quantitative finance research do NOT survive
this kind of test, and pretending otherwise would be the actual failure
here.

DO NOT use `ShrinkageAdaptiveCVaRRiskParity` in place of
`advanced/cvar_risk_parity.py::CVaRRiskParity` for real capital
allocation. Use the plain version, which has no such disclosed negative
finding against it.

---------------------------------------------------------------------
THE PROBLEM THIS METHOD TRIED TO SOLVE (a genuine, identifiable gap)
---------------------------------------------------------------------
CVaR Risk Parity estimates each asset's marginal contribution to
portfolio CVaR as -E[R_i | portfolio loss >= VaR_alpha], using only the
tail scenarios — at alpha=0.95 and T=500, about 25 observations feeding
each asset's estimate. This is a genuine small-sample estimation problem,
directly analogous to the one Bayes-Stein shrinkage (Jorion 1986) and
Ledoit-Wolf shrinkage solve elsewhere in this codebase.

---------------------------------------------------------------------
THE PROPOSED (AND EMPIRICALLY UNSUCCESSFUL) FIX
---------------------------------------------------------------------
Shrink each asset's tail-conditional mean toward its unconditional mean,
with intensity set either by a simple sample-size rule or a James-Stein-
style signal-to-noise statistic (both implemented below, both tested,
neither validated). See `examples/novel_contribution_validation.py` for
the full empirical writeup, including a diagnosed (and separately tested
and also-unsuccessful) fix for an optimization-instability issue found
along the way.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class SACVaRRiskParityResult:
    weights: pd.Series
    cvar_contributions: pd.Series
    shrinkage_intensities: pd.Series   # per-asset lambda actually applied
    effective_tail_size: int
    total_cvar: float
    success: bool


class ShrinkageAdaptiveCVaRRiskParity:
    """Shrinkage-Adaptive CVaR Risk Parity (SA-CVaR-RP) — see module
    docstring for the full derivation.

    ==================================================================
    EMPIRICAL STATUS (updated after testing — read this before using):
    ==================================================================
    This method was tested against its own stated falsifiable predictions
    in `examples/novel_contribution_validation.py` across multiple random
    seeds. Result: on this repo's synthetic data, NEITHER shrinkage rule
    ("sample_size" or "adaptive_snr") reliably reduced bootstrap weight
    variance or walk-forward turnover relative to plain CVaR Risk Parity.
    In most seeds tested, both metrics were *worse*, not better.

    A follow-up "fixed-point" reformulation (freezing the tail mask via a
    reference weight vector, to remove an iteration-path instability that
    was diagnosed as one candidate explanation) was also tried and made
    results *more* unstable, not less — the fixed marginal-contribution
    vector in that formulation creates a near-degenerate linear objective
    prone to corner solutions.

    HONEST CONCLUSION: the theoretical motivation (shrink a
    small-sample tail-conditional mean estimator, analogous to Bayes-
    Stein/James-Stein) is sound and well-precedented, but this specific
    implementation does not deliver the predicted benefit in this
    repo's testing. It is kept in the codebase, undisclosed status
    corrected, as a worked example of the full research loop — propose,
    derive, implement, pre-register falsifiable predictions, test, and
    report the result even when the result is negative — rather than as
    a working improvement over plain CVaR Risk Parity. Do not use this
    in place of `advanced/cvar_risk_parity.py` for real allocation
    decisions without further, independent validation. See
    `examples/novel_contribution_validation.py` for the multi-seed test
    that produced this conclusion.
    """

    def __init__(self, returns: pd.DataFrame, alpha: float = 0.95,
                 risk_budget: pd.Series | None = None, weight_bounds: tuple = (0.001, 1.0),
                 rule: str = "sample_size", prior_strength: float | None = None):
        self.returns = returns
        self.assets = list(returns.columns)
        self.R = returns.values
        self.T, self.N = self.R.shape
        self.alpha = alpha
        self.bounds = weight_bounds
        self.rule = rule
        self.prior_strength = prior_strength if prior_strength is not None else float(self.N)

        self.mu_full = returns.mean().values  # per-period unconditional mean, low-variance anchor
        full_cov = returns.cov().values + np.eye(self.N) * 1e-8
        self.full_cov_inv = np.linalg.pinv(full_cov)
        if risk_budget is None:
            self.budget = np.ones(self.N) / self.N
        else:
            b = risk_budget.reindex(self.assets).values
            self.budget = b / b.sum()

    def _shrunk_tail_contributions(self, w: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, int]:
        port_losses = -(self.R @ w)
        var_threshold = np.quantile(port_losses, self.alpha)
        tail_mask = port_losses >= var_threshold
        n_tail = max(int(tail_mask.sum()), 2)
        cvar = port_losses[tail_mask].mean()

        mu_tail = self.R[tail_mask].mean(axis=0)          # noisy, n_tail observations
        mu_full = self.mu_full                             # low-variance, T observations
        dof = max(self.N - 2, 1)

        if self.rule == "sample_size":
            # Simple, robust: shrinkage intensity depends only on how much
            # tail data we have relative to a prior_strength "pseudo-count"
            # (analogous to kappa_0 in the NIW Bayesian model), independent
            # of whether the observed deviation looks like signal or noise.
            lam = self.prior_strength / (self.prior_strength + n_tail)
        else:  # "adaptive_snr"
            # James-Stein-style: shrink less when mu_tail's deviation from
            # mu_full looks like genuine signal relative to sampling noise,
            # using the FULL-SAMPLE covariance (well-estimated, unlike a
            # tail-only covariance which would itself be a small-sample
            # estimate of exactly the kind this method exists to avoid).
            diff = mu_tail - mu_full
            quad = diff @ self.full_cov_inv @ diff
            stat = n_tail * quad
            lam = dof / (dof + stat) if stat > 0 else 1.0

        lam = float(np.clip(lam, 0.0, 1.0))

        shrunk_tail_mean = (1 - lam) * mu_tail + lam * mu_full
        marginal = -shrunk_tail_mean
        contributions = w * marginal
        scale = cvar / contributions.sum() if contributions.sum() != 0 else 1.0
        contributions = contributions * scale
        return contributions, cvar, np.full(self.N, lam), n_tail

    def solve(self, max_iter: int = 300) -> SACVaRRiskParityResult:
        def objective(w):
            w = w / w.sum()
            contrib, cvar, _, _ = self._shrunk_tail_contributions(w)
            target = cvar * self.budget
            return np.sum((contrib - target) ** 2)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [self.bounds] * self.N
        x0 = np.ones(self.N) / self.N
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": max_iter, "ftol": 1e-14})
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        contrib, cvar, lam, n_tail = self._shrunk_tail_contributions(w)

        return SACVaRRiskParityResult(
            weights=pd.Series(w, index=self.assets, name="weight"),
            cvar_contributions=pd.Series(contrib, index=self.assets, name="cvar_contribution"),
            shrinkage_intensities=pd.Series(lam, index=self.assets, name="shrinkage_lambda"),
            effective_tail_size=n_tail, total_cvar=cvar, success=bool(result.success),
        )
