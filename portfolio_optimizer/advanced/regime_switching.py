"""
NEW #3 — Regime-Switching Overlay.

A single static covariance/return estimate quietly assumes markets are in
one stationary regime forever. They aren't. This module fits a Gaussian
Hidden Markov Model on a portfolio-level (or market-index) return series
to detect latent regimes (typically: low-vol grind-up, high-vol crisis,
range-bound/neutral), then:

  1. Reports the current regime and its persistence/transition probabilities.
  2. Produces regime-conditional expected returns & covariances (estimated
     only from history classified into that regime) — feed these straight
     into any optimizer above for a regime-aware allocation.
  3. Produces a regime-blended estimate: a probability-weighted mix across
     regimes using the current filtered regime probabilities, which is a
     softer, less discontinuous input than hard regime-switching.

This is the standard "vol regime" overlay used across systematic macro and
multi-strategy funds to avoid feeding a crash-covariance optimizer with
calm-market correlations (which are typically much lower than crisis
correlations — the classic "correlations go to 1 in a crisis" problem).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False


@dataclass
class RegimeReport:
    current_regime: int
    regime_labels: dict
    transition_matrix: pd.DataFrame
    filtered_probabilities: pd.Series
    regime_series: pd.Series


class RegimeSwitchingOverlay:
    def __init__(self, returns: pd.DataFrame, n_regimes: int = 2, random_state: int = 42):
        if not _HAS_HMMLEARN:
            raise ImportError("hmmlearn is required for RegimeSwitchingOverlay "
                               "(pip install hmmlearn).")
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_regimes = n_regimes
        # Fit the HMM on the portfolio-level proxy series: equal-weight return
        # and rolling realized vol, which are enough to separate calm/crisis regimes
        # without overfitting a high-dimensional emission model.
        port_ret = returns.mean(axis=1)
        roll_vol = port_ret.rolling(20, min_periods=5).std().bfill()
        X = np.column_stack([port_ret.values, roll_vol.values])
        self.model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                                  n_iter=500, random_state=random_state)
        self.model.fit(X)
        self._X = X
        self._states = self.model.predict(X)
        self._posteriors = self.model.predict_proba(X)

        # Label regimes by realized volatility level (0 = calmest ... n-1 = most volatile)
        regime_vol = {s: port_ret.values[self._states == s].std() for s in range(n_regimes)}
        order = sorted(regime_vol, key=regime_vol.get)
        self.labels = {old: rank for rank, old in enumerate(order)}
        vol_names = ["low_vol", "medium_vol", "high_vol", "extreme_vol"]
        self.label_names = {rank: vol_names[min(rank, len(vol_names) - 1)]
                             for rank in range(n_regimes)}

    def report(self) -> RegimeReport:
        mapped_states = np.array([self.labels[s] for s in self._states])
        current = mapped_states[-1]
        trans = self.model.transmat_.copy()
        # remap transition matrix to labeled order
        order = [k for k, v in sorted(self.labels.items(), key=lambda kv: kv[1])]
        trans = trans[np.ix_(order, order)]
        trans_df = pd.DataFrame(trans, index=[self.label_names[i] for i in range(self.n_regimes)],
                                 columns=[self.label_names[i] for i in range(self.n_regimes)])
        filt = self._posteriors[-1][order]
        filt_series = pd.Series(filt, index=[self.label_names[i] for i in range(self.n_regimes)])
        regime_series = pd.Series(mapped_states, index=self.returns.index).map(self.label_names)
        return RegimeReport(
            current_regime=int(current), regime_labels=self.label_names,
            transition_matrix=trans_df, filtered_probabilities=filt_series,
            regime_series=regime_series,
        )

    def regime_conditional_moments(self) -> dict:
        """Expected returns & covariance estimated separately within each regime."""
        mapped_states = np.array([self.labels[s] for s in self._states])
        out = {}
        for r in range(self.n_regimes):
            mask = mapped_states == r
            if mask.sum() < 10:
                continue
            sub = self.returns.iloc[mask]
            out[self.label_names[r]] = {
                "mu": sub.mean() * 252,
                "cov": sub.cov() * 252,
                "n_obs": int(mask.sum()),
                "weight_in_history": float(mask.mean()),
            }
        return out

    def blended_moments(self) -> tuple[pd.Series, pd.DataFrame]:
        """Probability-weighted blend of regime-conditional mu/cov using the
        *current* filtered regime probabilities — a smooth, forward-looking
        estimate rather than a hard regime cutover.
        """
        rep = self.report()
        cond = self.regime_conditional_moments()
        mus, covs, weights = [], [], []
        for name, w in rep.filtered_probabilities.items():
            if name not in cond:
                continue
            mus.append(cond[name]["mu"] * w)
            covs.append(cond[name]["cov"] * w)
            weights.append(w)
        mu_blend = sum(mus) / sum(weights)
        cov_blend = sum(covs) / sum(weights)
        return mu_blend, cov_blend
