"""
NEW #5 — Factor Risk Model with Exposure Constraints.

Institutional allocators rarely optimize on raw asset covariance alone —
they decompose risk into systematic factor exposures (market, size, value,
momentum, or statistical PCA factors) plus idiosyncratic residual risk, and
constrain factor exposures directly ("keep net market beta under 1.2",
"limit exposure to the momentum factor"). This module provides:

  1. A statistical factor model via PCA on the returns covariance — no
     external factor data required, works on any asset universe out of the box.
  2. Support for a *fundamental* factor model if you supply factor returns
     (e.g. Fama-French, or your own house factors) via cross-sectional
     regression (time-series regression of asset returns on factor returns).
  3. Factor-structured covariance reconstruction: Sigma = B F B' + D, which
     is typically far better-conditioned than the raw sample covariance for
     wide, correlated universes (N assets close to or exceeding T observations).
  4. Portfolio factor exposure reporting and hard exposure-limit constraints
     that plug into the Markowitz optimizer's constraint system.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FactorModelResult:
    exposures: pd.DataFrame       # assets x factors (betas)
    factor_returns: pd.DataFrame  # time x factors
    factor_cov: pd.DataFrame      # factors x factors
    idiosyncratic_var: pd.Series  # per-asset residual variance
    reconstructed_cov: pd.DataFrame
    r_squared: pd.Series


class FactorRiskModel:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.assets = list(returns.columns)

    def fit_statistical(self, n_factors: int = 3, periods_per_year: int = 252) -> FactorModelResult:
        """PCA-based statistical factor model — the principal components of
        the (standardized) return covariance become the 'factors'.
        """
        X = self.returns.values
        X_centered = X - X.mean(axis=0, keepdims=True)
        # SVD for numerical stability instead of eigendecomposition of cov directly
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_factors = min(n_factors, Vt.shape[0])
        components = Vt[:n_factors]                     # n_factors x N (loadings direction)
        factor_scores = X_centered @ components.T        # T x n_factors (the "factor returns")

        # Exposures (betas) via regression of each asset on the factor scores
        F = factor_scores
        FtF_inv = np.linalg.pinv(F.T @ F)
        B = (FtF_inv @ F.T @ X_centered).T                # N x n_factors

        fitted = F @ B.T
        resid = X_centered - fitted
        idio_var = resid.var(axis=0) * periods_per_year

        factor_cov = np.cov(F.T) * periods_per_year
        if n_factors == 1:
            factor_cov = np.array([[factor_cov]])

        recon_cov = B @ factor_cov @ B.T + np.diag(idio_var)

        total_var = X_centered.var(axis=0)
        r_squared = 1 - (resid.var(axis=0) / np.where(total_var == 0, 1, total_var))

        factor_names = [f"PC{i+1}" for i in range(n_factors)]
        return FactorModelResult(
            exposures=pd.DataFrame(B, index=self.assets, columns=factor_names),
            factor_returns=pd.DataFrame(F, index=self.returns.index, columns=factor_names),
            factor_cov=pd.DataFrame(factor_cov, index=factor_names, columns=factor_names),
            idiosyncratic_var=pd.Series(idio_var, index=self.assets),
            reconstructed_cov=pd.DataFrame(recon_cov, index=self.assets, columns=self.assets),
            r_squared=pd.Series(r_squared, index=self.assets),
        )

    def fit_fundamental(self, factor_returns: pd.DataFrame,
                         periods_per_year: int = 252) -> FactorModelResult:
        """Time-series regression of each asset's returns on supplied factor
        returns (e.g. market, SMB, HML, momentum, or ADC's own house factors).
        """
        common_idx = self.returns.index.intersection(factor_returns.index)
        Y = self.returns.loc[common_idx]
        F = factor_returns.loc[common_idx]
        F_mat = np.column_stack([np.ones(len(F)), F.values])  # add intercept

        betas = np.linalg.lstsq(F_mat, Y.values, rcond=None)[0]  # (k+1) x N
        alphas = betas[0]
        B = betas[1:].T  # N x k

        fitted = F_mat @ betas
        resid = Y.values - fitted
        idio_var = resid.var(axis=0) * periods_per_year

        factor_cov = F.cov().values * periods_per_year
        recon_cov = B @ factor_cov @ B.T + np.diag(idio_var)

        total_var = Y.values.var(axis=0)
        r_squared = 1 - (resid.var(axis=0) / np.where(total_var == 0, 1, total_var))

        return FactorModelResult(
            exposures=pd.DataFrame(B, index=self.assets, columns=factor_returns.columns),
            factor_returns=F,
            factor_cov=pd.DataFrame(factor_cov, index=factor_returns.columns,
                                     columns=factor_returns.columns),
            idiosyncratic_var=pd.Series(idio_var, index=self.assets),
            reconstructed_cov=pd.DataFrame(recon_cov, index=self.assets, columns=self.assets),
            r_squared=pd.Series(r_squared, index=self.assets),
        )

    @staticmethod
    def portfolio_factor_exposure(weights: pd.Series, exposures: pd.DataFrame) -> pd.Series:
        """Net portfolio exposure to each factor = w' B."""
        w = weights.reindex(exposures.index).values
        return pd.Series(w @ exposures.values, index=exposures.columns, name="portfolio_exposure")

    @staticmethod
    def exposure_constraint(exposures: pd.DataFrame, factor: str, min_exp: float, max_exp: float):
        """Build a scipy-style constraint dict pair enforcing
        min_exp <= w' B_factor <= max_exp — pass straight into
        MarkowitzOptimizer's extra_constraints (via `_solve`) or compose manually.
        """
        b = exposures[factor].values
        return [
            {"type": "ineq", "fun": lambda w, b=b, hi=max_exp: hi - w @ b},
            {"type": "ineq", "fun": lambda w, b=b, lo=min_exp: w @ b - lo},
        ]
