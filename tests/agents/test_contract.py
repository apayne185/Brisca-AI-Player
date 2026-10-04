"""Behaviour every agent must satisfy, whatever its strategy."""

import dataclasses
import random
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brisca.agents import REGISTRY, Agent, make_agent
from brisca.observation import Observation, observe
from tests.helpers import random_playout

# Small budgets keep contract tests fast; strength is tested separately.
FAST_KWARGS: dict[str, dict[str, Any]] = {
    "random": {"seed": 0},
    "greedy": {},
    "heuristic": {},
    "ismcts": {"iterations": 20, "seed": 0},
    "alphabeta": {"samples": 2, "depth": 2, "seed": 0},
}


def mid_game_observations(deal_seed: int, policy: random.Random) -> list[Observation]:
    return [
        observe(state, state.to_play)
        for state in random_playout(deal_seed, policy)
        if not state.is_terminal
    ]


def test_every_registered_agent_has_fast_kwargs() -> None:
    assert set(FAST_KWARGS) == set(REGISTRY)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_agents_satisfy_protocol(name: str) -> None:
    agent = make_agent(name, **FAST_KWARGS[name])
    assert isinstance(agent, Agent)
    assert agent.name == name


@pytest.mark.parametrize("name", sorted(REGISTRY))
@settings(max_examples=10, deadline=None)
@given(deal_seed=st.integers(0, 2**32 - 1), policy=st.randoms(use_true_random=False))
def test_agents_play_a_card_from_their_hand(
    name: str, deal_seed: int, policy: random.Random
) -> None:
    agent = make_agent(name, **FAST_KWARGS[name])
    for obs in mid_game_observations(deal_seed, policy):
        snapshot = dataclasses.replace(obs)
        assert agent.act(obs) in obs.hand
        assert obs == snapshot


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_agents_are_reproducible_from_seed(name: str) -> None:
    observations = mid_game_observations(3, random.Random(3))
    first = make_agent(name, **FAST_KWARGS[name])
    second = make_agent(name, **FAST_KWARGS[name])
    assert [first.act(o) for o in observations] == [second.act(o) for o in observations]


def test_make_agent_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown agent 'nope'"):
        make_agent("nope")
