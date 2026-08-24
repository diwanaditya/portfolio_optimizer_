"""
Black-Litterman Model (He & Litterman, 1999 formulation).

Blends a market-implied equilibrium prior with subjective investor views,
producing a posterior return vector that is far better-behaved for
mean-variance optimization than raw historical means (which are the
single biggest source of "corner solution" garbage weights in naive
Markowitz).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class View:
    """A single investor view.

    Absolute view: assets=['AAPL'], weights=[1.0], value=0.12
        -> "I believe AAPL will return 12% annually"
    Relative view: assets=['AAPL', 'MSFT'], weights=[1.0, -1.0], value=0.03
        -> "I believe AAPL will outperform MSFT by 3%"
    confidence : 0-1, converted internally to view uncertainty (omega).
        Higher confidence = tighter (smaller) omega = view pulls harder.
    """
    assets: list
    weights: list
    value: float
    confidence: float = 0.5


class BlackLitterman:
    def __init__(self, cov_matrix: pd.DataFrame, market_caps: pd.Series | None = None,
                 risk_aversion: float = 2.5, tau: float = 0.05,
                 prior_returns: pd.Series | None = None):
        """
        Parameters
        ----------
        cov_matrix : annualized covariance matrix (assets x assets)
        market_caps : market-cap weights used to derive the equilibrium prior
                      via reverse optimization (pi = delta * Sigma * w_mkt).
                      If omitted, `prior_returns` must be supplied directly.
        risk_aversion : delta, market price of risk (~2.5 typical for equities)
        tau : scalar reflecting uncertainty in the prior (0.01-0.05 typical)
        prior_returns : supply directly instead of deriving from market caps
                        (e.g. from CAPM or an internal view of "neutral" returns)
        """
        self.assets = list(cov_matrix.index)
        self.cov = cov_matrix.reindex(index=self.assets, columns=self.assets).values
        self.tau = tau
        self.delta = risk_aversion

        if prior_returns is not None:
            self.pi = prior_returns.reindex(self.assets).values
        elif market_caps is not None:
            w_mkt = market_caps.reindex(self.assets).values
            w_mkt = w_mkt / w_mkt.sum()
            self.pi = self.delta * self.cov @ w_mkt
        else:
            raise ValueError("Provide either market_caps or prior_returns.")

        self.views: list[View] = []

    def add_view(self, view: View):
        self.views.append(view)
        return self

    def add_absolute_view(self, asset: str, value: float, confidence: float = 0.5):
        return self.add_view(View(assets=[asset], weights=[1.0], value=value, confidence=confidence))

    def add_relative_view(self, outperformer: str, underperformer: str, value: float,
                           confidence: float = 0.5):
        return self.add_view(View(assets=[outperformer, underperformer],
                                   weights=[1.0, -1.0], value=value, confidence=confidence))

    def _build_matrices(self):
        k = len(self.views)
        n = len(self.assets)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        idx = {a: i for i, a in enumerate(self.assets)}
        omega_diag = np.zeros(k)
        for i, v in enumerate(self.views):
            for a, w in zip(v.assets, v.weights):
                P[i, idx[a]] = w
            Q[i] = v.value
            # confidence -> omega: higher confidence => smaller uncertainty.
            # Standard He-Litterman scaling: omega_ii = tau * P Sigma P' / confidence_scale
            view_variance = self.tau * (P[i] @ self.cov @ P[i])
            conf = min(max(v.confidence, 1e-4), 0.9999)
            omega_diag[i] = view_variance * (1 - conf) / conf
        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def posterior(self) -> tuple[pd.Series, pd.DataFrame]:
        """Returns (posterior_expected_returns, posterior_covariance)."""
        if not self.views:
            post_ret = pd.Series(self.pi, index=self.assets)
            post_cov = pd.DataFrame(self.cov, index=self.assets, columns=self.assets)
            return post_ret, post_cov

        P, Q, Omega = self._build_matrices()
        tau_sigma = self.tau * self.cov
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        post_mu = middle @ (tau_sigma_inv @ self.pi + P.T @ omega_inv @ Q)
        post_cov = self.cov + middle  # add view-uncertainty-adjusted tau*Sigma term

        return (pd.Series(post_mu, index=self.assets, name="bl_expected_return"),
                pd.DataFrame(post_cov, index=self.assets, columns=self.assets))

    def implied_prior(self) -> pd.Series:
        """The pre-view equilibrium prior (reverse-optimized market returns)."""
        return pd.Series(self.pi, index=self.assets, name="prior_return")
