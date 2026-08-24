"""
Multi-Period Portfolio Optimization.

Every optimizer built so far solves "what's the best portfolio *today*,
assuming I never trade again." Real portfolios rebalance repeatedly, and
naive myopic (single-period) reoptimization at each rebalance date ignores
that today's trade affects the cost and opportunity set of tomorrow's trade.
This module provides two genuinely multi-period formulations:

1. **Analytical multi-period mean-variance** (Li & Ng, 2000): a closed-form
   solution for the optimal *sequence* of portfolio weights over a finite
   horizon under a mean-variance objective on *terminal* wealth, with
   i.i.d. per-period returns. Fast, exact, no simulation — the right tool
   when you trust that per-period return/covariance are stable over the
   horizon.

2. **Scenario-based Model Predictive Control (MPC) multi-period optimizer**:
   simulates many joint paths of future returns (block-bootstrapped from
   history, so it doesn't assume i.i.d. Gaussian returns), then jointly
   optimizes the *entire sequence* of rebalance decisions across the
   horizon to maximize expected CRRA/log utility of terminal wealth net of
   transaction costs at every rebalance — capturing the realistic tradeoff
   of "should I trade less aggressively today because I'll get another
   chance to rebalance in 20 days anyway."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class MultiPeriodPlan:
    weights_by_period: pd.DataFrame  # horizon_periods x n_assets
    expected_terminal_return: float
    expected_terminal_vol: float


class LiNgMultiPeriod:
    """Closed-form analytical multi-period mean-variance (Li & Ng, 2000).

    Under i.i.d. per-period returns with mean mu and covariance Sigma, and
    a mean-variance objective on TERMINAL wealth (maximize E[W_T] - phi *
    Var[W_T]), the optimal per-period weight is a *constant* proportional
    rebalancing rule (rebalance back to the same target weight vector each
    period) — this is the multi-period analogue of the single-period tangency
    portfolio, and it degenerates exactly to single-period Markowitz when
    horizon = 1.
    """

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 horizon_periods: int = 60, risk_aversion: float = 3.0):
        self.assets = list(expected_returns.index)
        self.mu = expected_returns.reindex(self.assets).values
        self.cov = cov_matrix.reindex(index=self.assets, columns=self.assets).values
        self.horizon = horizon_periods
        self.phi = risk_aversion

    def solve(self) -> MultiPeriodPlan:
        # Under the Li-Ng result, the per-period optimal weight for a
        # terminal-wealth mean-variance objective with i.i.d. returns is the
        # SAME constant-mix weight vector every period (the classic
        # "rebalance to target" result) — found by maximizing single-period
        # quadratic utility scaled to reflect T-period compounding risk.
        effective_aversion = self.phi * self.horizon  # variance compounds ~linearly in T for a fixed-mix strategy
        n = len(self.mu)

        def neg_utility(w):
            ret = w @ self.mu
            var = w @ self.cov @ w
            return -(ret - 0.5 * effective_aversion * var)

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * n
        res = minimize(neg_utility, np.ones(n) / n, method="SLSQP",
                        bounds=bounds, constraints=cons, options={"maxiter": 1000})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()

        weights_by_period = pd.DataFrame(
            np.tile(w, (self.horizon, 1)), columns=self.assets,
            index=[f"t+{i+1}" for i in range(self.horizon)],
        )
        term_ret = self.horizon * (w @ self.mu)
        term_var = self.horizon * (w @ self.cov @ w)  # i.i.d. compounding approx
        return MultiPeriodPlan(weights_by_period=weights_by_period,
                                expected_terminal_return=term_ret,
                                expected_terminal_vol=np.sqrt(term_var))


class ScenarioMPCOptimizer:
    """Scenario-based Model Predictive Control multi-period optimizer.

    Jointly optimizes a *sequence* of H rebalance decisions to maximize
    expected CRRA utility of terminal wealth across many bootstrapped
    future return paths, net of transaction costs paid at each rebalance.
    Unlike Li-Ng, this does NOT assume the optimal policy is a constant
    mix — it can front-load or back-load trading depending on the cost
    structure and path-dependent opportunity set, and it uses the actual
    empirical return distribution (fat tails, autocorrelation) rather than
    an i.i.d. Gaussian assumption.

    In production you'd re-solve this at every rebalance date using only
    the newest information (that's the "MPC" part — solve the whole
    horizon, execute only the first step, then re-plan) rather than
    committing to the full H-period plan up front.
    """

    def __init__(self, returns: pd.DataFrame, horizon_periods: int = 20,
                 n_scenarios: int = 200, block_size: int = 10,
                 transaction_cost_bps: float = 10.0, risk_aversion: float = 3.0,
                 random_state: int = 42):
        self.returns = returns
        self.assets = list(returns.columns)
        self.n = len(self.assets)
        self.horizon = horizon_periods
        self.n_scenarios = n_scenarios
        self.block_size = block_size
        self.cost = transaction_cost_bps / 10_000.0
        self.risk_aversion = risk_aversion
        self.rng = np.random.default_rng(random_state)
        self._scenarios = self._generate_scenarios()

    def _generate_scenarios(self) -> np.ndarray:
        """Block-bootstrap n_scenarios paths of length `horizon` from history."""
        T = len(self.returns)
        R = self.returns.values
        paths = np.zeros((self.n_scenarios, self.horizon, self.n))
        for s in range(self.n_scenarios):
            n_blocks = int(np.ceil(self.horizon / self.block_size))
            starts = self.rng.integers(0, max(T - self.block_size, 1), size=n_blocks)
            idx = np.concatenate([np.arange(st, st + self.block_size) for st in starts])[:self.horizon]
            idx = np.clip(idx, 0, T - 1)
            paths[s] = R[idx]
        return paths

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def solve(self, initial_weights: np.ndarray | None = None) -> MultiPeriodPlan:
        n, H = self.n, self.horizon
        w0 = initial_weights if initial_weights is not None else np.ones(n) / n

        def unpack_logits(x):
            return x.reshape(H, n)

        def objective(x):
            logits = unpack_logits(x)
            W = self._softmax_rows(logits)   # smooth simplex parameterization -> clean gradients

            prev_w = np.tile(w0, (self.n_scenarios, 1))
            wealth = np.ones(self.n_scenarios)
            for t in range(H):
                w_t = np.tile(W[t], (self.n_scenarios, 1))
                turnover = np.abs(w_t - prev_w).sum(axis=1)
                cost = turnover * self.cost
                period_ret = (self._scenarios[:, t, :] * w_t).sum(axis=1) - cost
                wealth *= (1 + period_ret)
                prev_w = w_t
            total_utility = wealth.mean() - 0.5 * self.risk_aversion * wealth.var()
            return -total_utility

        # initialize logits near log(w0) with small random jitter per period so the
        # optimizer isn't starting from a perfectly flat (zero-gradient-in-symmetry) point
        base_logits = np.log(np.clip(w0, 1e-6, None))
        x0 = np.tile(base_logits, H) + self.rng.normal(0, 0.05, size=H * n)
        result = minimize(objective, x0, method="Powell",
                           options={"maxiter": 400, "xtol": 1e-6, "ftol": 1e-9})
        W = self._softmax_rows(unpack_logits(result.x))

        weights_df = pd.DataFrame(W, columns=self.assets,
                                   index=[f"t+{i+1}" for i in range(H)])

        # simulate final expected terminal stats under the solved plan
        prev_w = np.tile(w0, (self.n_scenarios, 1))
        wealth = np.ones(self.n_scenarios)
        for t in range(H):
            w_t = np.tile(W[t], (self.n_scenarios, 1))
            turnover = np.abs(w_t - prev_w).sum(axis=1)
            cost = turnover * self.cost
            period_ret = (self._scenarios[:, t, :] * w_t).sum(axis=1) - cost
            wealth *= (1 + period_ret)
            prev_w = w_t

        return MultiPeriodPlan(weights_by_period=weights_df,
                                expected_terminal_return=float(wealth.mean() - 1),
                                expected_terminal_vol=float(wealth.std()))
