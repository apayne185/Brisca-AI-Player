"""The 40-card Spanish deck and Brisca card values."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class Suit(IntEnum):
    OROS = 0
    COPAS = 1
    ESPADAS = 2
    BASTOS = 3

    @property
    def symbol(self) -> str:
        return self.name[0]

    @classmethod
    def from_symbol(cls, symbol: str) -> Suit:
        for suit in cls:
            if suit.symbol == symbol.upper():
                return suit
        raise ValueError(f"unknown suit symbol {symbol!r}")


RANKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

# Ranks in ascending trick-taking strength: 2 < 4 < 5 < 6 < 7 < 10 < 11 < 12 < 3 < 1.
_STRENGTH_ORDER: tuple[int, ...] = (2, 4, 5, 6, 7, 10, 11, 12, 3, 1)
STRENGTH: dict[int, int] = {rank: i for i, rank in enumerate(_STRENGTH_ORDER)}

POINTS: dict[int, int] = dict.fromkeys(RANKS, 0) | {1: 11, 3: 10, 12: 4, 11: 3, 10: 2}

TOTAL_POINTS = 120
_RANK_INDEX: dict[int, int] = {rank: i for i, rank in enumerate(RANKS)}


class Card(NamedTuple):
    rank: int
    suit: Suit

    @property
    def points(self) -> int:
        return POINTS[self.rank]

    @property
    def strength(self) -> int:
        return STRENGTH[self.rank]

    @property
    def ordinal(self) -> int:
        """Stable id in ``range(40)``, used for vector encodings and action masks."""
        return int(self.suit) * len(RANKS) + _RANK_INDEX[self.rank]

    @classmethod
    def from_ordinal(cls, ordinal: int) -> Card:
        return DECK[ordinal]

    @classmethod
    def parse(cls, text: str) -> Card:
        """Parse the compact form produced by ``str()``, e.g. ``"1O"`` or ``"12E"``."""
        rank, suit = int(text[:-1]), Suit.from_symbol(text[-1])
        if rank not in POINTS:
            raise ValueError(f"invalid rank {rank} in {text!r}")
        return cls(rank, suit)

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.symbol}"


DECK: tuple[Card, ...] = tuple(Card(rank, suit) for suit in Suit for rank in RANKS)
