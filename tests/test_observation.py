import dataclasses
import random
from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brisca import GameState, determinize, legal_actions, observe, step
from tests.helpers import random_playout

deal_seeds = st.integers(min_value=0, max_value=2**32 - 1)
policies = st.randoms(use_true_random=False)
players = st.sampled_from([0, 1])


def states_of(deal_seed: int, policy: random.Random) -> list[GameState]:
    return [s for s in random_playout(deal_seed, policy) if not s.is_terminal]


@given(deal_seed=deal_seeds, policy=policies, player=players, sample_seed=st.integers())
def test_determinized_state_is_consistent_with_observation(
    deal_seed: int, policy: random.Random, player: int, sample_seed: int
) -> None:
    for state in states_of(deal_seed, policy):
        obs = observe(state, player)
        sample = determinize(obs, sample_seed)
        assert observe(sample, player) == obs
        assert obs.trump == state.trump
        assert sample.hands[player] == state.hands[player]
        assert sorted(sample.hands[obs.opponent] + sample.stock) == sorted(
            state.hands[obs.opponent] + state.stock
        ), "hidden cards are redistributed, never invented"


@given(deal_seed=deal_seeds, policy=policies, player=players)
def test_observation_does_not_leak_hidden_information(
    deal_seed: int, policy: random.Random, player: int
) -> None:
    for state in states_of(deal_seed, policy):
        obs = observe(state, player)
        resampled = determinize(obs, seed=0)
        assert observe(resampled, player) == obs
        # Every card the player cannot see is either unseen or provably the opponent's.
        hidden = set(state.hands[obs.opponent]) | set(state.stock) - {state.trump_card}
        assert set(obs.unseen_cards()) | set(obs.known_opponent_cards()) == hidden


@given(deal_seed=deal_seeds, policy=policies, player=players)
def test_drawn_trump_card_is_tracked(deal_seed: int, policy: random.Random, player: int) -> None:
    for state in states_of(deal_seed, policy):
        obs = observe(state, player)
        for card in obs.known_opponent_cards():
            assert card in state.hands[obs.opponent]
            assert card in determinize(obs, seed=1).hands[obs.opponent]


@given(deal_seed=deal_seeds, policy=policies, sample_seed=st.integers())
def test_determinized_games_can_be_played_out(
    deal_seed: int, policy: random.Random, sample_seed: int
) -> None:
    states = states_of(deal_seed, policy)
    state = determinize(observe(states[len(states) // 2], 0), sample_seed)
    rng = random.Random(sample_seed)
    while not state.is_terminal:
        state = step(state, rng.choice(legal_actions(state)))
    assert sum(state.scores) == 120


def test_determinize_samples_opponent_hand_uniformly() -> None:
    obs = observe(next(random_playout(deal_seed=5, policy_rng=random.Random(5))), player=0)
    rng = random.Random(0)
    n_samples = 20_000
    counts: Counter[object] = Counter()
    for _ in range(n_samples):
        counts.update(determinize(obs, rng).hands[1])

    unseen = obs.unseen_cards()
    expected = n_samples * obs.opponent_hand_size / len(unseen)
    assert set(counts) == set(unseen)
    # Binomial std is ~sqrt(20000 * 3/33) ≈ 42; allow a generous 5-sigma band.
    assert all(abs(count - expected) < 5 * expected**0.5 for count in counts.values())


def test_determinize_rejects_inconsistent_observation() -> None:
    obs = observe(next(random_playout(deal_seed=0, policy_rng=random.Random(0))), player=0)
    broken = dataclasses.replace(obs, stock_size=5)
    with pytest.raises(ValueError, match="inconsistent"):
        determinize(broken)
