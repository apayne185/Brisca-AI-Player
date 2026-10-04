"""Fixed-size numeric encodings of observations and actions for learned models.

Actions are card ordinals in ``range(40)``; a mask marks which are legal (the
cards in hand). Observations become a flat float32 vector of one-hot card
planes plus a few normalised scalars, so that a plain MLP can consume them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from brisca.cards import DECK, TOTAL_POINTS, Card, Suit
from brisca.engine import HAND_SIZE, NUM_PLAYERS, beats
from brisca.observation import Observation

NUM_ACTIONS = len(DECK)
_FULL_STOCK = len(DECK) - NUM_PLAYERS * HAND_SIZE

CARD_PLANES = (
    "hand",
    "trick",
    "played",
    "trump_card",
    "known_opponent",
    "trump_suit",
    "beats_trick",
)
SCALARS = ("my_score", "opponent_score", "stock_size", "is_leading", "opponent_hand_size")
OBS_SIZE = len(CARD_PLANES) * NUM_ACTIONS + len(Suit) + len(SCALARS)

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def _plane(cards: tuple[Card, ...] | frozenset[Card]) -> FloatArray:
    plane = np.zeros(NUM_ACTIONS, dtype=np.float32)
    plane[[card.ordinal for card in cards]] = 1.0
    return plane


def encode_observation(obs: Observation) -> FloatArray:
    trump_suit = np.zeros(len(Suit), dtype=np.float32)
    trump_suit[obs.trump] = 1.0
    scalars = np.array(
        [
            obs.scores[obs.player] / TOTAL_POINTS,
            obs.scores[obs.opponent] / TOTAL_POINTS,
            obs.stock_size / _FULL_STOCK,
            float(not obs.current_trick),
            obs.opponent_hand_size / HAND_SIZE,
        ],
        dtype=np.float32,
    )
    played = frozenset(card for trick in obs.history for card in trick.cards)
    # Relational planes: the network should not have to rediscover the rules of
    # trick-taking from card ids alone.
    trumps = tuple(card for card in DECK if card.suit == obs.trump)
    winners = (
        tuple(card for card in DECK if beats(card, obs.current_trick[0], obs.trump))
        if obs.current_trick
        else ()
    )
    return np.concatenate(
        [
            _plane(obs.hand),
            _plane(obs.current_trick),
            _plane(played),
            _plane((obs.trump_card,)),
            _plane(obs.known_opponent_cards()),
            _plane(trumps),
            _plane(winners),
            trump_suit,
            scalars,
        ]
    )


def action_mask(obs: Observation) -> BoolArray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[[card.ordinal for card in obs.hand]] = True
    return mask
