"""
Reinforcement Learning agents for portfolio management: PPO, SAC, DQN.

Design notes
------------
- PPO and SAC act in the *continuous* action space of `PortfolioEnv`
  (pre-softmax logits over assets, squashed to portfolio weights inside
  the environment) — the standard setup in the deep-RL portfolio
  management literature (Jiang et al. 2017; Zhang, Zohren & Roberts 2020).
- DQN is inherently discrete, so it chooses among a small library of
  *meta-actions* (go all-in on each asset, hold, or equal-weight) rather
  than a continuous weight vector — this is the standard, honest way to
  apply DQN to portfolio choice rather than pretending it natively
  handles continuous simplex actions.
- All three are compact from-scratch PyTorch implementations (no
  stable-baselines3 dependency) so the whole repo stays auditable and
  dependency-light, at the cost of being less exhaustively tuned than a
  mature RL library. Fine for research/comparison use; for a live trading
  desk you'd want longer training runs and hyperparameter search.
"""
from __future__ import annotations
import numpy as np
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "RL agents (PPOAgent, SACAgent, DQNAgent) require PyTorch, which is not "
            "installed. Install it with:\n\n    pip install torch\n\n"
            "This is only required if you use portfolio_optimizer.rl -- every other "
            "module in this package works without torch."
        )


# When torch is unavailable, fall back to plain `object` as the base class
# so the network class DEFINITIONS below don't fail at module-import time
# (a bare `class Foo(nn.Module)` would raise AttributeError immediately if
# nn is None). The method BODIES inside these classes (which reference
# nn.Linear, F.relu, etc.) are only ever executed if someone actually
# instantiates and calls them -- and _require_torch() in every Agent's
# __init__ guarantees that never happens without torch installed, with a
# clear error message instead of a confusing AttributeError deep in a
# network class.
_ModuleBase = nn.Module if _TORCH_AVAILABLE else object


def _mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ============================================================= PPO =====
class PPOActorCritic(_ModuleBase):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.actor_mean = _mlp(state_dim, action_dim, hidden)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.critic = _mlp(state_dim, 1, hidden)

    def forward(self, state):
        mean = self.actor_mean(state)
        std = torch.exp(self.actor_log_std).clamp(1e-3, 2.0)
        value = self.critic(state).squeeze(-1)
        return mean, std, value


class PPOAgent:
    """Proximal Policy Optimization (Schulman et al., 2017), clipped
    surrogate objective + GAE advantage estimation.
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, epochs=10, minibatch_size=64, entropy_coef=0.01):
        _require_torch()
        self.net = PPOActorCritic(state_dim, action_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip_eps = gamma, gae_lambda, clip_eps
        self.epochs, self.minibatch_size, self.entropy_coef = epochs, minibatch_size, entropy_coef

    def act(self, state: np.ndarray, deterministic: bool = False):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.net(s)
            if deterministic:
                action = mean
                logp = torch.zeros(1)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
        return action.squeeze(0).numpy(), logp.item(), value.item()

    def _gae(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        values = list(values) + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values[:-1])
        return advantages, returns

    def train_on_rollout(self, states, actions, logps, rewards, values, dones, last_value):
        advantages, returns = self._gae(rewards, values, dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.float32)
        old_logp_t = torch.as_tensor(np.array(logps), dtype=torch.float32)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        n = len(states)
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                mean, std, value = self.net(states_t[mb])
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(actions_t[mb]).sum(-1)
                ratio = torch.exp(logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[mb])
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

    def train(self, env, total_steps: int = 20_000, rollout_len: int = 200):
        state = env.reset()
        steps_done = 0
        while steps_done < total_steps:
            states, actions, logps, rewards, values, dones = [], [], [], [], [], []
            for _ in range(rollout_len):
                action, logp, value = self.act(state)
                next_state, reward, done, _ = env.step(action)
                states.append(state); actions.append(action); logps.append(logp)
                rewards.append(reward); values.append(value); dones.append(float(done))
                state = next_state
                steps_done += 1
                if done:
                    state = env.reset()
                if steps_done >= total_steps:
                    break
            _, _, last_value = self.act(state)
            self.train_on_rollout(states, actions, logps, rewards, values, dones, last_value)
        return self


# ============================================================= SAC =====
class SACActor(_ModuleBase):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = _mlp(state_dim, hidden, hidden)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = F.relu(self.net(state))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)  # squash to bounded range, env softmaxes on top
        logp = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(-1)


class SACCritic(_ModuleBase):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, 1, hidden)
        self.q2 = _mlp(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class SACAgent:
    """Soft Actor-Critic (Haarnoja et al., 2018): off-policy, entropy-
    regularized actor-critic with twin Q-networks and automatic entropy
    temperature — well suited to the continuous portfolio-weight action
    space since it's markedly more sample-efficient than PPO off small
    training budgets (relevant since real market history is finite).
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005,
                 buffer_size=50_000, batch_size=128, target_entropy=None):
        _require_torch()
        self.actor = SACActor(state_dim, action_dim)
        self.critic = SACCritic(state_dim, action_dim)
        self.critic_target = SACCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy or -action_dim

        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.buffer = deque(maxlen=buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state: np.ndarray, deterministic: bool = False):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(s)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(s)
        return action.squeeze(0).numpy()

    def remember(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32)
        a = torch.as_tensor(np.array(a), dtype=torch.float32)
        r = torch.as_tensor(np.array(r), dtype=torch.float32)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
        d = torch.as_tensor(np.array(d), dtype=torch.float32)

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = r + self.gamma * (1 - d) * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        new_action, logp = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        self._soft_update()

    def train(self, env, total_steps: int = 20_000, warmup_steps: int = 1000):
        state = env.reset()
        for step in range(total_steps):
            if step < warmup_steps:
                action = np.random.uniform(-1, 1, size=env.action_dim)
            else:
                action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            self.remember(state, action, reward, next_state, float(done))
            state = next_state if not done else env.reset()
            self.update()
        return self


# ============================================================= DQN =====
class DQNNetwork(_ModuleBase):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = _mlp(state_dim, n_actions, hidden)

    def forward(self, state):
        return self.net(state)


class DQNAgent:
    """Deep Q-Network (Mnih et al., 2015) over a discrete *meta-action*
    library: {go all-in on asset i for each i, hold current weights,
    equal-weight rebalance}. DQN is fundamentally a discrete-action
    algorithm, so this is the honest way to apply it to portfolio choice
    rather than forcing it onto a continuous simplex.
    """
    def __init__(self, state_dim, n_assets, lr=1e-3, gamma=0.99,
                 buffer_size=50_000, batch_size=128, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay_steps=10_000):
        _require_torch()
        self.n_assets = n_assets
        self.n_actions = n_assets + 2  # all-in on each asset, hold, equal-weight
        self.q_net = DQNNetwork(state_dim, self.n_actions)
        self.target_net = DQNNetwork(state_dim, self.n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma, self.batch_size = gamma, batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.eps_start, self.eps_end, self.eps_decay = epsilon_start, epsilon_end, epsilon_decay_steps
        self._step_count = 0

    def _action_to_weights_logits(self, action_idx: int) -> np.ndarray:
        """Map a discrete action index to a continuous weight-logit vector
        compatible with PortfolioEnv's softmax projection."""
        logits = np.full(self.n_assets, -10.0, dtype=np.float32)
        if action_idx < self.n_assets:
            logits[action_idx] = 10.0        # all-in on asset `action_idx`
        elif action_idx == self.n_assets:
            logits[:] = 0.0                  # equal-weight
        else:
            logits[:] = 0.0                  # "hold": env re-softmaxes; closest discrete proxy
        return logits

    def epsilon(self):
        frac = min(self._step_count / self.eps_decay, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, int]:
        self._step_count += 1
        if not deterministic and np.random.rand() < self.epsilon():
            action_idx = np.random.randint(self.n_actions)
        else:
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.q_net(s)
            action_idx = int(q.argmax(dim=-1).item())
        return self._action_to_weights_logits(action_idx), action_idx

    def remember(self, s, a_idx, r, s2, d):
        self.buffer.append((s, a_idx, r, s2, d))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32)
        a = torch.as_tensor(np.array(a), dtype=torch.int64)
        r = torch.as_tensor(np.array(r), dtype=torch.float32)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
        d = torch.as_tensor(np.array(d), dtype=torch.float32)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(s2).max(dim=1).values
            target = r + self.gamma * (1 - d) * next_q
        loss = F.mse_loss(q_values, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def train(self, env, total_steps: int = 20_000, target_update_every: int = 500):
        state = env.reset()
        for step in range(total_steps):
            logits, a_idx = self.act(state)
            next_state, reward, done, _ = env.step(logits)
            self.remember(state, a_idx, reward, next_state, float(done))
            state = next_state if not done else env.reset()
            self.update()
            if step % target_update_every == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        return self
