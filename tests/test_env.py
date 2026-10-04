import random

import numpy as np
import pytest

from brisca.agents import Agent, GreedyAgent, RandomAgent
from brisca.encoding import OBS_SIZE
from brisca.env import BriscaEnv, VectorEnv, fixed


def run_episode(env: BriscaEnv, rng: random.Random) -> list[float]:
    _, mask = env.reset()
    rewards, done = [], False
    while not done:
        result = env.step(rng.choice(np.flatnonzero(mask).tolist()))
        rewards.append(result.reward)
        mask, done = result.mask, result.done
    return rewards


def test_episode_has_one_decision_per_learner_card() -> None:
    env = BriscaEnv(fixed(GreedyAgent()), seed=0)
    assert len(run_episode(env, random.Random(0))) == 20


def test_unshaped_reward_is_terminal_outcome_only() -> None:
    env = BriscaEnv(fixed(GreedyAgent()), seed=1)
    rewards = run_episode(env, random.Random(1))
    assert rewards[:-1] == [0.0] * 19
    assert rewards[-1] in (-1.0, 0.0, 1.0)


def test_shaped_rewards_sum_to_outcome_plus_scaled_margin() -> None:
    env = BriscaEnv(fixed(GreedyAgent()), shaping=0.5, seed=2)
    total = sum(run_episode(env, random.Random(2)))
    assert env.state is not None
    me, them = env.state.scores[env.seat], env.state.scores[1 - env.seat]
    outcome = float(np.sign(me - them))
    assert total == pytest.approx(outcome + 0.5 * (me - them) / 120)


def test_learner_plays_both_seats() -> None:
    env = BriscaEnv(fixed(RandomAgent(seed=0)), seed=3)
    seats = set()
    for _ in range(20):
        env.reset()
        seats.add(env.seat)
    assert seats == {0, 1}


def test_step_before_reset_fails() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        BriscaEnv(fixed(GreedyAgent())).step(0)


def test_opponent_is_sampled_each_episode() -> None:
    pool: list[Agent] = [RandomAgent(seed=0), GreedyAgent()]
    env = BriscaEnv(lambda rng: rng.choice(pool), seed=4)
    seen = set()
    for _ in range(20):
        env.reset()
        seen.add(env.opponent)
    assert seen == set(pool)


def test_vector_env_auto_resets_finished_games() -> None:
    envs = VectorEnv([BriscaEnv(fixed(GreedyAgent()), seed=i) for i in range(4)])
    obs, masks = envs.reset()
    assert obs.shape == (4, OBS_SIZE)
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(45):
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        obs, masks, _rewards, dones = envs.step(actions)
        finished += int(dones.sum())
        assert masks.any(axis=1).all(), "every env always has a legal move"
    assert finished >= len(envs) * 2
