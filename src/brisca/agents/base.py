"""The interface every Brisca agent implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from brisca.cards import Card
from brisca.observation import Observation


@runtime_checkable
class Agent(Protocol):
    """Chooses a card to play from what one player can see.

    Agents only ever receive an ``Observation``, never the full ``GameState``,
    so they cannot cheat by looking at hidden cards. Stochastic agents take a
    seed at construction so that games are reproducible.
    """

    name: str

    def act(self, obs: Observation) -> Card:
        """Return a card from ``obs.hand``."""
        ...
