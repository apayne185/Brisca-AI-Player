"""Brisca: an imperfect-information game AI platform."""

from importlib.metadata import version

from brisca.cards import DECK, POINTS, RANKS, TOTAL_POINTS, Card, Suit
from brisca.engine import (
    GameState,
    IllegalActionError,
    Trick,
    legal_actions,
    new_game,
    returns,
    step,
    trick_winner,
    winner,
)
from brisca.observation import Observation, determinize, observe

__version__ = version("brisca")

__all__ = [
    "DECK",
    "POINTS",
    "RANKS",
    "TOTAL_POINTS",
    "Card",
    "GameState",
    "IllegalActionError",
    "Observation",
    "Suit",
    "Trick",
    "__version__",
    "determinize",
    "legal_actions",
    "new_game",
    "observe",
    "returns",
    "step",
    "trick_winner",
    "winner",
]
