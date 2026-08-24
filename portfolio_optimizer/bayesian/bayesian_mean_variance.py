"""
Bayesian Mean-Variance & Bayesian Expected Returns.

Naive Markowitz treats mu_hat and Sigma_hat as if they were known with
certainty. In reality they're estimated from finite, noisy history, and the
optimizer happily overfits to that noise (this is the same disease the
Michaud resampler in `advanced/robust_frontier.py` treats via brute-force
simulation). The Bayesian approach treats this properly and analytically:
put a prior on (mu, Sigma), update it with observed data, and optimize on
the *posterior predictive* distribution — which has fatter, more honest
uncertainty than the naive plug-in estimate.

Two estimators are provided:

1. Bayes-Stein shrinkage (Jorion, 1986) — closed-form empirical Bayes
   estimator that shrinks the sample mean toward a common "grand mean"
   (the minimum-variance portfolio's implied return), with the shrinkage
   intensity chosen analytically from the data. This is the classic,
   fast, no-simulation fix for "sample means are almost pure noise."

2. Full Normal-Inverse-Wishart (NIW) conjugate Bayesian model — puts a
   conjugate NIW prior on (mu, Sigma), updates it with the observed
   sample, and returns the full posterior AND posterior-predictive
   covariance (which is strictly larger than Sigma_hat, correctly
   reflecting parameter uncertainty — the textbook "Bayesian
   mean-variance" input to Markowitz).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BayesStein:
    """Jorion (1986) Bayes-Stein shrinkage estimator for expected returns."""
    shrunk_mean: pd.Series
    grand_mean: float
    shrinkage_intensity: float
    original_mean: pd.Series


def bayes_stein_shrinkage(returns: pd.DataFrame, cov_matrix: pd.DataFrame | None = None,
                           periods_per_year: int = 252) -> BayesStein:
    """
    Shrinks the sample mean toward the grand mean (return of the global
    minimum-variance portfolio), with shrinkage intensity:

        lambda = (N + 2) / ((N + 2) + T * (mu_hat - mu_min)' Sigma^-1 (mu_hat - mu_min))

    where N = number of assets, T = number of observations. This is the
    analytically-optimal shrinkage under a diffuse prior on the grand mean
    and known Sigma (Jorion 1986) — no cross-validation or hyperparameter
    tuning required.
    """
    assets = list(returns.columns)
    T, N = returns.shape
    mu_hat = returns.mean().values * periods_per_year
    cov = (cov_matrix.values if cov_matrix is not None
           else returns.cov().values * periods_per_year)
    cov_inv = np.linalg.pinv(cov)

    ones = np.ones(N)
    # Global minimum-variance portfolio implied return = the "grand mean"
    w_min = cov_inv @ ones / (ones @ cov_inv @ ones)
    grand_mean = float(w_min @ mu_hat)

    diff = mu_hat - grand_mean
    quad = diff @ cov_inv @ diff
    lam = (N + 2) / ((N + 2) + T * quad) if quad > 0 else 1.0
    lam = float(np.clip(lam, 0.0, 1.0))

    shrunk = lam * grand_mean + (1 - lam) * mu_hat
    return BayesStein(
        shrunk_mean=pd.Series(shrunk, index=assets, name="bayes_stein_mean"),
        grand_mean=grand_mean, shrinkage_intensity=lam,
        original_mean=pd.Series(mu_hat, index=assets),
    )


@dataclass
class NIWPosterior:
    """Normal-Inverse-Wishart posterior over (mu, Sigma)."""
    posterior_mean: pd.Series             # E[mu | data]
    posterior_predictive_cov: pd.DataFrame  # Cov of a *new* return draw (wider than Sigma_hat)
    posterior_cov_of_mean: pd.DataFrame   # Cov of the estimate of mu itself (shrinks with more data)
    kappa_n: float
    nu_n: float


class BayesianMeanVariance:
    """Conjugate Normal-Inverse-Wishart Bayesian update for (mu, Sigma), then
    Markowitz optimization on the posterior-predictive moments.

    Prior: mu | Sigma ~ N(mu_0, Sigma / kappa_0),  Sigma ~ Inv-Wishart(Psi_0, nu_0)
    This is the standard conjugate setup so the posterior is available in
    closed form with no MCMC required.

    Reasonable defaults (kappa_0 small, nu_0 = N+2) make this close to an
    uninformative prior that still guarantees a proper (non-degenerate)
    posterior — tune kappa_0/nu_0 up if you have genuine outside conviction
    in `prior_mean`/`prior_cov`.
    """

    def __init__(self, returns: pd.DataFrame, prior_mean: pd.Series | None = None,
                 prior_cov: pd.DataFrame | None = None, kappa_0: float = 1.0,
                 nu_0: float | None = None, periods_per_year: int = 252):
        self.returns = returns
        self.assets = list(returns.columns)
        self.T, self.N = returns.shape
        self.ppy = periods_per_year

        X = returns.values
        self.sample_mean = X.mean(axis=0)
        centered = X - self.sample_mean
        self.sample_scatter = centered.T @ centered  # (T-1) * sample cov, unannualized

        self.mu_0 = (prior_mean.reindex(self.assets).values if prior_mean is not None
                     else self.sample_mean.copy())
        self.psi_0 = (prior_cov.reindex(index=self.assets, columns=self.assets).values
                      if prior_cov is not None else np.cov(X.T) * kappa_0)
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0 if nu_0 is not None else self.N + 2

    def posterior(self) -> NIWPosterior:
        T, N = self.T, self.N
        kappa_n = self.kappa_0 + T
        nu_n = self.nu_0 + T
        mu_n = (self.kappa_0 * self.mu_0 + T * self.sample_mean) / kappa_n

        diff = (self.sample_mean - self.mu_0).reshape(-1, 1)
        psi_n = (self.psi_0 + self.sample_scatter +
                  (self.kappa_0 * T / kappa_n) * (diff @ diff.T))

        # E[Sigma | data] under Inv-Wishart(psi_n, nu_n) = psi_n / (nu_n - N - 1)
        expected_sigma = psi_n / (nu_n - N - 1)
        # Posterior-predictive covariance of a NEW draw integrates out both
        # mu and Sigma uncertainty: predictive Cov = (kappa_n+1)/kappa_n * E[Sigma]
        predictive_cov = (kappa_n + 1) / kappa_n * expected_sigma
        # Covariance of the posterior mean estimate itself (shrinks as T grows)
        cov_of_mean = expected_sigma / kappa_n

        ann = self.ppy
        return NIWPosterior(
            posterior_mean=pd.Series(mu_n * ann, index=self.assets, name="bayesian_mean"),
            posterior_predictive_cov=pd.DataFrame(predictive_cov * ann,
                                                   index=self.assets, columns=self.assets),
            posterior_cov_of_mean=pd.DataFrame(cov_of_mean * ann,
                                                index=self.assets, columns=self.assets),
            kappa_n=kappa_n, nu_n=nu_n,
        )

    def optimize(self, risk_free_rate: float = 0.0, weight_bounds: tuple = (0.0, 1.0)):
        """Convenience: run Markowitz max-Sharpe directly on the posterior-
        predictive moments (properly parameter-uncertainty-widened Sigma).
        """
        from ..optimizers.markowitz import MarkowitzOptimizer
        post = self.posterior()
        opt = MarkowitzOptimizer(post.posterior_mean, post.posterior_predictive_cov,
                                  risk_free_rate=risk_free_rate, weight_bounds=weight_bounds)
        return opt.max_sharpe(), post
