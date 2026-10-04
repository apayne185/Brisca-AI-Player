"""A parameterised rule-based agent encoding common Brisca tactics."""

from __future__ import annotations

from dataclasses import dataclass

from brisca.agents.tactics import cheapest, winners_against
from brisca.cards import Card
from brisca.observation import Observation


@dataclass(frozen=True, slots=True)
class HeuristicParams:
    """Tunable knobs. Defaults are sensible; Phase 4 tunes them with Optuna."""

    capture_threshold: int = 10
    """Minimum points on the table worth spending a trump to capture."""
    trump_cost: float = 5.0
    """Reluctance to give away a trump when leading or discarding."""
    secure_points: bool = True
    """When winning in the led suit, win with the highest-point card to bank it."""
    endgame_trumps: bool = True
    """Once the stock is empty, spend trumps on any trick carrying points."""


class HeuristicAgent:
    """Wins valuable tricks cheaply, saves trumps, and dumps low cards otherwise."""

    name = "heuristic"

    def __init__(self, params: HeuristicParams | None = None) -> None:
        self.params = params or HeuristicParams()

    def act(self, obs: Observation) -> Card:
        p, trump, hand = self.params, obs.trump, obs.hand
        if not obs.current_trick:
            return cheapest(hand, trump, p.trump_cost)

        lead = obs.current_trick[0]
        winners = winners_against(hand, lead, trump)
        in_suit = [c for c in winners if c.suit == lead.suit and c.suit != trump]
        if in_suit:
            if p.secure_points:
                return max(in_suit, key=lambda c: (c.points, c.strength))
            return cheapest(in_suit, trump)

        trumps = [c for c in winners if c.suit == trump]
        threshold = 1 if p.endgame_trumps and obs.stock_size == 0 else p.capture_threshold
        if trumps and lead.points >= threshold:
            return min(trumps, key=lambda c: (c.points, c.strength))

        return cheapest(hand, trump, p.trump_cost)
