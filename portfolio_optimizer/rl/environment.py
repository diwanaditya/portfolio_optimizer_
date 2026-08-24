"""
Portfolio Management Environment (Gym-like, no external gym dependency).

State:   a window of the last `lookback` periods of returns (flattened),
         plus current portfolio weights.
Action:  target portfolio weights (continuous, simplex-projected).
Reward:  log-return of the portfolio net of transaction costs, optionally
         penalized for volatility (risk-adjusted reward) — the standard
         setup used in most "Deep RL for portfolio management" papers
         (e.g. Jiang, Xu & Liang 2017; Zhang, Zohren & Roberts 2020).

Kept dependency-free (no `gymnasium` requirement) so it's trivial to drop
into any existing training loop, while still exposing the familiar
`reset()` / `step()` interface.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class PortfolioEnv:
    def __init__(self, returns: pd.DataFrame, lookback: int = 30,
                 transaction_cost_bps: float = 10.0, reward_vol_penalty: float = 0.0,
                 episode_length: int | None = None):
        self.returns = returns.values.astype(np.float32)
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.lookback = lookback
        self.cost = transaction_cost_bps / 10_000.0
        self.vol_penalty = reward_vol_penalty
        self.T = len(returns)
        self.episode_length = episode_length or (self.T - lookback - 1)

        self.state_dim = lookback * self.n_assets + self.n_assets  # returns window + current weights
        self.action_dim = self.n_assets

        self._t = None
        self._start = None
        self._weights = None

    def reset(self, start_index: int | None = None) -> np.ndarray:
        max_start = self.T - self.lookback - self.episode_length - 1
        self._start = start_index if start_index is not None else np.random.randint(0, max(max_start, 1))
        self._t = self._start + self.lookback
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._recent_returns = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        window = self.returns[self._t - self.lookback:self._t].flatten()
        return np.concatenate([window, self._weights]).astype(np.float32)

    @staticmethod
    def _project_to_simplex(action: np.ndarray) -> np.ndarray:
        """Map an unconstrained action vector to valid portfolio weights via
        softmax — guarantees long-only weights summing to 1 regardless of
        the raw network output's scale/sign.
        """
        a = action - action.max()
        e = np.exp(a)
        return e / e.sum()

    def step(self, action: np.ndarray):
        target_w = self._project_to_simplex(np.asarray(action, dtype=np.float32))
        turnover = np.abs(target_w - self._weights).sum()
        txn_cost = turnover * self.cost

        period_return = self.returns[self._t]
        gross_port_return = float(target_w @ period_return)
        net_port_return = gross_port_return - txn_cost

        # log-return reward (standard choice: additive across time, matches
        # compounding growth objective), with optional volatility penalty
        reward = float(np.log1p(np.clip(net_port_return, -0.99, None)))
        self._recent_returns.append(net_port_return)
        if self.vol_penalty > 0 and len(self._recent_returns) >= 5:
            recent_vol = np.std(self._recent_returns[-20:])
            reward -= self.vol_penalty * recent_vol

        # drift weights forward with realized returns, then reset to target
        # (target *is* the post-trade weight, so next state starts from it)
        self._weights = target_w
        self._t += 1

        done = (self._t >= self._start + self.lookback + self.episode_length) or (self._t >= self.T - 1)
        info = {"portfolio_return": net_port_return, "turnover": turnover, "weights": target_w.copy()}
        return self._get_state(), reward, done, info

    def weights_series(self, weights: np.ndarray) -> pd.Series:
        return pd.Series(weights, index=self.assets)
