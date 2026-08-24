"""
RL vs Classical Optimizer Benchmark.

Runs PPO / SAC / DQN policies against Markowitz, Black-Litterman, and Risk
Parity on the *same* held-out test period, and reports a standard
performance comparison table (annualized return, vol, Sharpe, max
drawdown, turnover). This is the honest way to evaluate whether RL is
actually earning its (very real) extra complexity and training cost over
a classical closed-form/convex-optimization baseline — which, per the
literature (and typically in practice on liquid, low-dimensional asset
universes like this one), it often does not clearly beat.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .environment import PortfolioEnv
from ..estimators.expected_returns import mean_historical_return
from ..estimators.covariance import ledoit_wolf_shrinkage
from ..optimizers.markowitz import MarkowitzOptimizer
from ..optimizers.black_litterman import BlackLitterman
from ..optimizers.risk_parity import RiskParity


@dataclass
class StrategyPerformance:
    name: str
    returns: pd.Series
    weights_history: pd.DataFrame

    def metrics(self, periods_per_year: int = 252) -> dict:
        r = self.returns
        if len(r) == 0 or r.std() == 0:
            return {}
        equity = (1 + r).cumprod()
        ann_ret = equity.iloc[-1] ** (periods_per_year / len(r)) - 1
        ann_vol = r.std() * np.sqrt(periods_per_year)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        dd = (equity / equity.cummax() - 1).min()
        turnover = self.weights_history.diff().abs().sum(axis=1).mean() if len(self.weights_history) > 1 else 0.0
        return {"annualized_return": ann_ret, "annualized_vol": ann_vol,
                "sharpe_ratio": sharpe, "max_drawdown": dd, "avg_turnover": turnover}


def evaluate_rl_agent(agent, test_returns: pd.DataFrame, lookback: int = 30,
                       transaction_cost_bps: float = 10.0, name: str = "RL Agent",
                       action_fn=None) -> StrategyPerformance:
    env = PortfolioEnv(test_returns, lookback=lookback, transaction_cost_bps=transaction_cost_bps)
    env.episode_length = env.T - env.lookback - 1
    state = env.reset(start_index=0)

    def default_fn(s, det):
        result = agent.act(s, deterministic=det)
        return result[0] if isinstance(result, tuple) else result

    fn = action_fn or default_fn
    rewards, weights_hist = [], []
    done = False
    while not done:
        action = fn(state, True)
        state, reward, done, info = env.step(action)
        rewards.append(info["portfolio_return"])
        weights_hist.append(info["weights"])

    returns_series = pd.Series(rewards)
    weights_df = pd.DataFrame(weights_hist)
    returns_series.index = test_returns.index[env.lookback:env.lookback + len(returns_series)]
    weights_df.index = returns_series.index
    weights_df.columns = test_returns.columns
    return StrategyPerformance(name=name, returns=returns_series, weights_history=weights_df)


def evaluate_classical_strategy(strategy_name: str, train_returns: pd.DataFrame,
                                 test_returns: pd.DataFrame, rebalance_every: int = 21,
                                 transaction_cost_bps: float = 10.0,
                                 market_caps: pd.Series | None = None) -> StrategyPerformance:
    """Static-fit classical baseline: fit once on train_returns, hold (with
    periodic rebalance back to the same target) through test_returns —
    fair, comparable protocol to how the RL policy is evaluated out-of-sample.
    """
    assets = list(train_returns.columns)

    if strategy_name == "markowitz":
        mu = mean_historical_return(train_returns)
        cov, _ = ledoit_wolf_shrinkage(train_returns)
        target = MarkowitzOptimizer(mu, cov, risk_free_rate=0.03).max_sharpe().weights
    elif strategy_name == "black_litterman":
        cov, _ = ledoit_wolf_shrinkage(train_returns)
        caps = market_caps if market_caps is not None else pd.Series(1.0, index=assets)
        bl = BlackLitterman(cov, market_caps=caps)
        post_mu, post_cov = bl.posterior()
        target = MarkowitzOptimizer(post_mu, post_cov, risk_free_rate=0.03).max_sharpe().weights
    elif strategy_name == "risk_parity":
        cov, _ = ledoit_wolf_shrinkage(train_returns)
        target = RiskParity(cov).solve()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    weights = target.reindex(assets).values
    weights_hist, port_returns = [], []
    current = weights.copy()
    for t in range(len(test_returns)):
        if t % rebalance_every == 0 and t > 0:
            turnover = np.abs(weights - current).sum()
            cost = turnover * (transaction_cost_bps / 10_000.0)
            current = weights.copy()
        else:
            cost = 0.0
        r = test_returns.iloc[t].values
        port_ret = float(current @ r) - cost
        port_returns.append(port_ret)
        weights_hist.append(current.copy())
        grown = current * (1 + r)
        if grown.sum() > 0:
            current = grown / grown.sum()

    return StrategyPerformance(
        name=strategy_name,
        returns=pd.Series(port_returns, index=test_returns.index),
        weights_history=pd.DataFrame(weights_hist, index=test_returns.index, columns=assets),
    )


def comparison_table(performances: list) -> pd.DataFrame:
    rows = {}
    for p in performances:
        rows[p.name] = p.metrics()
    return pd.DataFrame(rows).T
