"""Reinforcement-learning environments: one learner seat against an opponent agent.

The interface mirrors Gymnasium (``reset``/``step``) but stays dependency-free
and adds the action mask that card games need. ``VectorEnv`` steps many games
in lockstep and auto-resets finished ones, which is what batched policy
inference in PPO expects.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from brisca.agents.base import Agent
from brisca.cards import DECK, TOTAL_POINTS
from brisca.encoding import BoolArray, FloatArray, action_mask, encode_observation
from brisca.engine import NUM_PLAYERS, GameState, Seed, make_rng, new_game, returns, step
from brisca.observation import observe

OpponentSampler = Callable[[random.Random], Agent]
"""Picks the opponent for each new episode, e.g. from a league of past policies."""


def fixed(agent: Agent) -> OpponentSampler:
    """An opponent sampler that always returns ``agent``."""
    return lambda _rng: agent


@dataclass(frozen=True, slots=True)
class StepResult:
    obs: FloatArray
    mask: BoolArray
    reward: float
    done: bool


class BriscaEnv:
    """The learner plays a random seat against an opponent drawn per episode.

    Reward is the terminal outcome (+1/0/-1). ``shaping`` optionally adds the
    change in point difference after each of the learner's moves, scaled to
    the game total, which densifies the signal early in training without
    changing the optimal policy's preference for winning.
    """

    def __init__(self, opponent: OpponentSampler, shaping: float = 0.0, seed: Seed = None) -> None:
        self._sample_opponent = opponent
        self.shaping = shaping
        self._rng = make_rng(seed)
        self.state: GameState | None = None
        self.seat = 0
        self.opponent: Agent | None = None

    def reset(self) -> tuple[FloatArray, BoolArray]:
        self.seat = self._rng.randrange(NUM_PLAYERS)
        self.opponent = self._sample_opponent(self._rng)
        first_player = self._rng.randrange(NUM_PLAYERS)
        self.state = self._advance_opponent(new_game(self._rng, first_player=first_player))
        return self._observe()

    def step(self, action: int) -> StepResult:
        state = self.state
        if state is None or state.is_terminal:
            raise RuntimeError("call reset() before step()")
        before = self._margin(state)
        state = self._advance_opponent(step(state, DECK[action]))
        self.state = state

        reward = self.shaping * (self._margin(state) - before) / TOTAL_POINTS
        if state.is_terminal:
            reward += returns(state)[self.seat]
        obs, mask = self._observe()
        return StepResult(obs, mask, reward, state.is_terminal)

    def _advance_opponent(self, state: GameState) -> GameState:
        assert self.opponent is not None
        while not state.is_terminal and state.to_play != self.seat:
            state = step(state, self.opponent.act(observe(state, state.to_play)))
        return state

    def _margin(self, state: GameState) -> int:
        return 2 * state.scores[self.seat] - sum(state.scores)

    def _observe(self) -> tuple[FloatArray, BoolArray]:
        assert self.state is not None
        obs = observe(self.state, self.seat)
        return encode_observation(obs), action_mask(obs)


class VectorEnv:
    """Steps ``len(envs)`` games in lockstep, resetting each one when it ends."""

    def __init__(self, envs: list[BriscaEnv]) -> None:
        self.envs = envs

    def __len__(self) -> int:
        return len(self.envs)

    def reset(self) -> tuple[FloatArray, BoolArray]:
        obs, masks = zip(*(env.reset() for env in self.envs), strict=True)
        return np.stack(obs), np.stack(masks)

    def step(self, actions: np.ndarray) -> tuple[FloatArray, BoolArray, np.ndarray, np.ndarray]:
        """Returns next observations, masks, rewards and done flags.

        For finished games the returned observation is the first one of the
        next game, as in Gymnasium's autoreset vector environments.
        """
        obs, masks, rewards, dones = [], [], [], []
        for env, action in zip(self.envs, actions, strict=True):
            result = env.step(int(action))
            o, m = env.reset() if result.done else (result.obs, result.mask)
            obs.append(o)
            masks.append(m)
            rewards.append(result.reward)
            dones.append(result.done)
        return (
            np.stack(obs),
            np.stack(masks),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
        )
