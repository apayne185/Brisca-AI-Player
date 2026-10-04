"""Proximal Policy Optimization with masked actions and a self-play league.

Each episode's opponent is drawn from a league: fixed baselines (random,
greedy, heuristic) and frozen snapshots of the learner itself, taken
periodically during training. Mixing in baselines keeps the policy from
over-fitting to its own quirks; snapshots keep raising the bar.
"""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from brisca.agents import GreedyAgent, HeuristicAgent, RandomAgent
from brisca.agents.base import Agent
from brisca.arena import play_match
from brisca.encoding import NUM_ACTIONS, OBS_SIZE
from brisca.env import BriscaEnv, VectorEnv
from brisca.rl.agent import PolicyAgent
from brisca.rl.model import ActorCritic

Metrics = dict[str, float]


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 1_000_000
    num_envs: int = 32
    rollout_len: int = 64
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 1e-3
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    shaping: float = 0.5
    """Weight of the per-move point-difference reward relative to the win/loss reward."""
    self_play_prob: float = 0.5
    """Chance an episode's opponent is a past snapshot rather than a baseline."""
    snapshot_every: int = 20
    """Updates between adding the current policy to the league."""
    max_snapshots: int = 10
    eval_every: int = 50
    eval_deals: int = 100
    seed: int = 0


class League:
    """Opponent pool: fixed baselines plus a rolling window of frozen snapshots."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.baselines: list[Agent] = [
            RandomAgent(seed=config.seed),
            GreedyAgent(),
            HeuristicAgent(),
        ]
        self.snapshots: list[Agent] = []

    def add_snapshot(self, model: ActorCritic, seed: int) -> None:
        frozen = copy.deepcopy(model)
        self.snapshots.append(PolicyAgent(frozen, greedy=False, seed=seed))
        self.snapshots = self.snapshots[-self.config.max_snapshots :]

    def sample(self, rng: random.Random) -> Agent:
        if self.snapshots and rng.random() < self.config.self_play_prob:
            return rng.choice(self.snapshots)
        return rng.choice(self.baselines)


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over a ``(T, N)`` rollout."""
    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_adv = delta + gamma * lam * not_done * next_adv
        advantages[t] = next_adv
        next_value = values[t]
    return advantages, advantages + values


def train(
    config: PPOConfig,
    on_metrics: Callable[[int, Metrics], None] | None = None,
) -> ActorCritic:
    """Train a policy and return it. ``on_metrics(step, metrics)`` receives progress."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = ActorCritic(hidden=config.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    league = League(config)
    envs = VectorEnv(
        [
            BriscaEnv(league.sample, shaping=config.shaping, seed=config.seed * 10_000 + i)
            for i in range(config.num_envs)
        ]
    )

    horizon, n_envs = config.rollout_len, config.num_envs
    batch_size = horizon * n_envs
    num_updates = max(1, config.total_steps // batch_size)

    obs_buf = torch.zeros(horizon, n_envs, OBS_SIZE)
    mask_buf = torch.zeros(horizon, n_envs, NUM_ACTIONS, dtype=torch.bool)
    act_buf = torch.zeros(horizon, n_envs, dtype=torch.long)
    logp_buf = torch.zeros(horizon, n_envs)
    val_buf = torch.zeros(horizon, n_envs)
    rew_buf = torch.zeros(horizon, n_envs)
    done_buf = torch.zeros(horizon, n_envs)

    np_obs, np_mask = envs.reset()
    obs, mask = torch.from_numpy(np_obs), torch.from_numpy(np_mask)
    episode_returns = np.zeros(n_envs, dtype=np.float32)
    finished: list[float] = []
    start = time.perf_counter()

    for update in range(1, num_updates + 1):
        model.eval()
        for t in range(horizon):
            with torch.no_grad():
                logits, value = model(obs, mask)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            obs_buf[t], mask_buf[t], act_buf[t] = obs, mask, action
            logp_buf[t], val_buf[t] = dist.log_prob(action), value

            np_obs, np_mask, reward, done = envs.step(action.numpy())
            rew_buf[t] = torch.from_numpy(reward)
            done_buf[t] = torch.from_numpy(done.astype(np.float32))
            episode_returns += reward
            finished.extend(episode_returns[done].tolist())
            episode_returns[done] = 0.0
            obs, mask = torch.from_numpy(np_obs), torch.from_numpy(np_mask)

        with torch.no_grad():
            _, last_value = model(obs, mask)
        advantages, targets = _gae(
            rew_buf, val_buf, done_buf, last_value, config.gamma, config.gae_lambda
        )

        metrics = _update(
            model,
            optimizer,
            config,
            obs_buf.reshape(batch_size, -1),
            mask_buf.reshape(batch_size, -1),
            act_buf.reshape(-1),
            logp_buf.reshape(-1),
            advantages.reshape(-1),
            targets.reshape(-1),
        )

        if update % config.snapshot_every == 0:
            league.add_snapshot(model, seed=config.seed + update)

        steps = update * batch_size
        metrics["steps_per_second"] = steps / (time.perf_counter() - start)
        if finished:
            metrics["episode_return"] = float(np.mean(finished))
            finished.clear()
        if update % config.eval_every == 0 or update == num_updates:
            metrics["eval_vs_heuristic"] = evaluate(model, HeuristicAgent(), config.eval_deals)
        if on_metrics:
            on_metrics(steps, metrics)

    return model


def _update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    obs: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    targets: torch.Tensor,
) -> Metrics:
    model.train()
    totals: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}
    batch_size = obs.shape[0]
    for _ in range(config.epochs):
        for idx in torch.randperm(batch_size).split(config.minibatch_size):
            logits, value = model(obs[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            log_ratio = logp - old_logp[idx]
            ratio = log_ratio.exp()

            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -torch.min(
                ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv
            ).mean()
            value_loss = 0.5 * (value - targets[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            totals["policy_loss"].append(policy_loss.item())
            totals["value_loss"].append(value_loss.item())
            totals["entropy"].append(entropy.item())
            # Low-variance KL estimator (Schulman), for monitoring update size.
            totals["kl"].append(((ratio - 1) - log_ratio).mean().item())
    return {key: float(np.mean(values)) for key, values in totals.items()}


def evaluate(model: ActorCritic, opponent: Agent, deals: int, seed: int = 12345) -> float:
    """Greedy-policy duplicate-match score against ``opponent``."""
    return play_match(PolicyAgent(model, greedy=True), opponent, deals=deals, seed=seed).score
