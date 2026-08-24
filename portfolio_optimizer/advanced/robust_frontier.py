"""
NEW #4 — Robust / Resampled Efficient Frontier (Michaud & Michaud, 1998/2008).

Standard Markowitz treats the estimated mu and Sigma as if they were the
true parameters — they are not, they're noisy estimates, and the optimizer
happily exploits every bit of that noise, producing concentrated, unstable
"error-maximizing" portfolios that whipsaw wildly between rebalances.

This module fixes that via Monte Carlo resampling:
  1. Treat the historical sample (mu_hat, Sigma_hat) as if it were the true
     distribution, and draw B bootstrap resamples of the same length from it
     (via multivariate normal simulation, or block bootstrap on real history
     to preserve autocorrelation/fat tails).
  2. Re-estimate mu, Sigma on each resample and re-run the optimizer.
  3. Average the resulting weights across all B resamples at each point on
     the frontier.

The averaged weights are dramatically more diversified and far more stable
out-of-sample than the single-sample "optimal" portfolio — this is the
technique behind the patented Michaud resampling approach used at New
Frontier Advisors and widely replicated across the industry since.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..optimizers.markowitz import MarkowitzOptimizer


@dataclass
class ResampledResult:
    weights: pd.Series
    weight_std_across_resamples: pd.Series
    n_resamples: int


class ResampledEfficientFrontier:
    def __init__(self, returns: pd.DataFrame, n_resamples: int = 500,
                 block_size: int = 20, risk_free_rate: float = 0.0,
                 weight_bounds: tuple = (0.0, 1.0), random_state: int = 42):
        """
        Parameters
        ----------
        returns : historical periodic returns
        n_resamples : number of Monte Carlo resamples (500 is a reasonable
                      production default; more is smoother but slower)
        block_size : block bootstrap block length (preserves volatility
                     clustering / autocorrelation far better than iid resampling)
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_resamples = n_resamples
        self.block_size = block_size
        self.rf = risk_free_rate
        self.bounds = weight_bounds
        self.rng = np.random.default_rng(random_state)

    def _block_bootstrap_sample(self) -> pd.DataFrame:
        T = len(self.returns)
        n_blocks = int(np.ceil(T / self.block_size))
        starts = self.rng.integers(0, T - self.block_size, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + self.block_size) for s in starts])[:T]
        return self.returns.iloc[idx].reset_index(drop=True)

    def resampled_max_sharpe(self) -> ResampledResult:
        all_weights = []
        for _ in range(self.n_resamples):
            sample = self._block_bootstrap_sample()
            mu = sample.mean() * 252
            cov = sample.cov() * 252
            try:
                opt = MarkowitzOptimizer(mu, cov, risk_free_rate=self.rf,
                                          weight_bounds=self.bounds)
                res = opt.max_sharpe()
                if res.success:
                    all_weights.append(res.weights.values)
            except Exception:
                continue
        W = np.array(all_weights)
        mean_w = W.mean(axis=0)
        mean_w = np.clip(mean_w, 0, None)
        mean_w = mean_w / mean_w.sum()
        return ResampledResult(
            weights=pd.Series(mean_w, index=self.assets, name="weight"),
            weight_std_across_resamples=pd.Series(W.std(axis=0), index=self.assets),
            n_resamples=len(all_weights),
        )

    def resampled_frontier(self, n_points: int = 20) -> pd.DataFrame:
        """Resampled frontier: for each target-return point, average the
        resampled weights across all B simulations. Slower (B x n_points
        optimizations) but this is the full Michaud frontier.
        """
        base_mu = self.returns.mean() * 252
        min_ret, max_ret = base_mu.min(), base_mu.max()
        targets = np.linspace(min_ret, max_ret * 0.95, n_points)
        accum = {t: [] for t in targets}

        for _ in range(self.n_resamples):
            sample = self._block_bootstrap_sample()
            mu = sample.mean() * 252
            cov = sample.cov() * 252
            opt = MarkowitzOptimizer(mu, cov, risk_free_rate=self.rf, weight_bounds=self.bounds)
            for t in targets:
                try:
                    res = opt.target_return(t)
                    if res.success:
                        accum[t].append(res.weights.values)
                except Exception:
                    continue

        rows = []
        for t in targets:
            if not accum[t]:
                continue
            W = np.array(accum[t])
            w_mean = W.mean(axis=0)
            w_mean = np.clip(w_mean, 0, None)
            w_mean = w_mean / w_mean.sum()
            port_ret = w_mean @ base_mu.values
            base_cov = (self.returns.cov() * 252).values
            port_vol = np.sqrt(w_mean @ base_cov @ w_mean)
            rows.append({"target_return": t, "return": port_ret, "volatility": port_vol,
                         "weights": dict(zip(self.assets, w_mean))})
        return pd.DataFrame(rows)
