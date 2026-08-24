import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_universe
from portfolio_optimizer.rl.environment import PortfolioEnv
from portfolio_optimizer.rl.agents import PPOAgent, SACAgent, DQNAgent
from portfolio_optimizer.rl.benchmark import evaluate_rl_agent, evaluate_classical_strategy, comparison_table


@pytest.fixture(scope="module")
def returns():
    return generate_synthetic_universe(n_days=400)


@pytest.fixture(scope="module")
def env(returns):
    return PortfolioEnv(returns, lookback=15, transaction_cost_bps=10)


class TestPortfolioEnv:
    def test_reset_returns_correct_state_dim(self, env):
        state = env.reset(start_index=0)
        assert state.shape == (env.state_dim,)

    def test_step_returns_valid_tuple(self, env):
        env.reset(start_index=0)
        action = np.zeros(env.action_dim)
        state, reward, done, info = env.step(action)
        assert state.shape == (env.state_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "weights" in info
        assert np.isclose(info["weights"].sum(), 1.0, atol=1e-5)

    def test_simplex_projection_valid(self, env):
        action = np.array([5.0, -3.0, 0.0, 1.0, 2.0, -1.0, 0.5, 4.0])[:env.action_dim]
        w = env._project_to_simplex(action)
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert (w >= 0).all()

    def test_episode_terminates(self, env):
        env.episode_length = 20
        env.reset(start_index=0)
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, done, _ = env.step(np.zeros(env.action_dim))
            steps += 1
        assert done
        assert steps <= 21


class TestPPOAgent:
    def test_ppo_trains_and_acts(self, env):
        agent = PPOAgent(env.state_dim, env.action_dim)
        agent.train(env, total_steps=500, rollout_len=100)
        state = env.reset(start_index=0)
        action, logp, value = agent.act(state, deterministic=True)
        assert action.shape == (env.action_dim,)
        assert np.isfinite(value)


class TestSACAgent:
    def test_sac_trains_and_acts(self, env):
        agent = SACAgent(env.state_dim, env.action_dim)
        agent.train(env, total_steps=300, warmup_steps=100)
        state = env.reset(start_index=0)
        action = agent.act(state, deterministic=True)
        assert action.shape == (env.action_dim,)
        assert np.all(np.abs(action) <= 1.01)  # tanh-squashed


class TestDQNAgent:
    def test_dqn_trains_and_acts(self, env):
        agent = DQNAgent(env.state_dim, env.action_dim)
        agent.train(env, total_steps=300)
        state = env.reset(start_index=0)
        logits, action_idx = agent.act(state, deterministic=True)
        assert logits.shape == (env.action_dim,)
        assert 0 <= action_idx < agent.n_actions

    def test_dqn_action_mapping_all_in(self, env):
        agent = DQNAgent(env.state_dim, env.action_dim)
        logits = agent._action_to_weights_logits(0)
        weights = env._project_to_simplex(logits)
        assert weights[0] > 0.9  # should be nearly all-in on asset 0


class TestBenchmarkComparison:
    def test_classical_strategy_evaluation(self, returns):
        train, test = returns.iloc[:300], returns.iloc[300:]
        perf = evaluate_classical_strategy("risk_parity", train, test)
        assert len(perf.returns) == len(test)
        metrics = perf.metrics()
        assert "sharpe_ratio" in metrics

    def test_rl_agent_evaluation(self, returns):
        train, test = returns.iloc[:300], returns.iloc[300:]
        env = PortfolioEnv(train, lookback=15, transaction_cost_bps=10)
        agent = PPOAgent(env.state_dim, env.action_dim)
        agent.train(env, total_steps=400, rollout_len=100)
        perf = evaluate_rl_agent(agent, test, lookback=15, name="PPO")
        assert len(perf.returns) > 0
        assert perf.name == "PPO"

    def test_comparison_table_structure(self, returns):
        train, test = returns.iloc[:300], returns.iloc[300:]
        perf_mv = evaluate_classical_strategy("markowitz", train, test)
        perf_rp = evaluate_classical_strategy("risk_parity", train, test)
        table = comparison_table([perf_mv, perf_rp])
        assert set(table.index) == {"markowitz", "risk_parity"}
        assert "sharpe_ratio" in table.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
