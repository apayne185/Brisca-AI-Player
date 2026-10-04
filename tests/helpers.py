"""Shared builders for engine tests."""

from __future__ import annotations

import random
from collections.abc import Iterator

from brisca import Card, GameState, legal_actions, new_game, step


def cards(text: str) -> tuple[Card, ...]:
    """``cards("1O 3C")`` -> ``(Card(1, OROS), Card(3, COPAS))``."""
    return tuple(Card.parse(token) for token in text.split())


def make_state(
    hand0: str,
    hand1: str,
    trump: str,
    stock: str = "",
    trick: str = "",
    to_play: int = 0,
    scores: tuple[int, int] = (0, 0),
) -> GameState:
    return GameState(
        hands=(cards(hand0), cards(hand1)),
        stock=cards(stock),
        trump_card=Card.parse(trump),
        current_trick=cards(trick),
        to_play=to_play,
        scores=scores,
        history=(),
    )


def random_playout(deal_seed: int, policy_rng: random.Random) -> Iterator[GameState]:
    """Yield every state of a game where both players move uniformly at random."""
    state = new_game(deal_seed)
    yield state
    while not state.is_terminal:
        state = step(state, policy_rng.choice(legal_actions(state)))
        yield state
