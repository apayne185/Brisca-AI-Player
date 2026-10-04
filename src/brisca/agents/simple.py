"""Baseline agents: uniform random and one-ply greedy."""

from __future__ import annotations

from brisca.agents.tactics import card_cost, cheapest
from brisca.cards import Card
from brisca.engine import Seed, beats, make_rng
from brisca.observation import Observation


class RandomAgent:
    """Plays a uniformly random card. The floor every other agent must beat."""

    name = "random"

    def __init__(self, seed: Seed = None) -> None:
        self._rng = make_rng(seed)

    def act(self, obs: Observation) -> Card:
        return self._rng.choice(obs.hand)


class GreedyAgent:
    """Maximises the points of the current trick, ignoring the future.

    Following, it plays the card with the best immediate point swing, spending
    as little as possible. Leading, it gives away its cheapest card.
    """

    name = "greedy"

    def act(self, obs: Observation) -> Card:
        trump = obs.trump
        if not obs.current_trick:
            return cheapest(obs.hand, trump)

        lead = obs.current_trick[0]

        def swing(card: Card) -> tuple[float, float]:
            at_stake = lead.points + card.points
            return (at_stake if beats(card, lead, trump) else -at_stake, -card_cost(card, trump))

        return max(obs.hand, key=swing)
