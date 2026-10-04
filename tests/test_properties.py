"""Invariants that must hold in every reachable state of every game."""

import random
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from brisca import DECK, TOTAL_POINTS, Card, GameState, returns
from tests.helpers import random_playout

deal_seeds = st.integers(min_value=0, max_value=2**32 - 1)
policies = st.randoms(use_true_random=False)


def all_cards(state: GameState) -> list[Card]:
    return [
        *(card for hand in state.hands for card in hand),
        *state.stock,
        *state.current_trick,
        *(card for trick in state.history for card in trick.cards),
    ]


@settings(max_examples=200)
@given(deal_seed=deal_seeds, policy=policies)
def test_invariants_hold_throughout_the_game(deal_seed: int, policy: random.Random) -> None:
    states = list(random_playout(deal_seed, policy))

    assert len(states) == len(DECK) + 1, "every card is played exactly once"
    for state in states:
        assert Counter(all_cards(state)) == Counter(DECK), "cards are conserved"
        assert sum(state.scores) == sum(t.points for t in state.history)
        assert all(len(hand) <= 3 for hand in state.hands)
        if state.stock:
            assert state.stock[-1] == state.trump_card, "face-up trump is drawn last"
            if not state.current_trick:
                assert [len(hand) for hand in state.hands] == [3, 3]
        if state.history and not state.current_trick:
            assert state.to_play == state.history[-1].winner, "trick winner leads"

    final = states[-1]
    assert final.is_terminal
    assert sum(final.scores) == TOTAL_POINTS
    assert sum(returns(final)) == 0, "zero-sum"


@given(deal_seed=deal_seeds, policy=policies)
def test_hands_sizes_only_shrink_in_the_last_three_tricks(
    deal_seed: int, policy: random.Random
) -> None:
    for state in random_playout(deal_seed, policy):
        if not state.current_trick:
            expected = min(3, 20 - len(state.history))
            assert [len(hand) for hand in state.hands] == [expected, expected]
