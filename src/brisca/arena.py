"""Play agents against each other."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from brisca.agents.base import Agent
from brisca.engine import NUM_PLAYERS, GameState, IllegalActionError, new_game, step, winner
from brisca.observation import observe


def play_game(agents: Sequence[Agent], seed: int, first_player: int = 0) -> GameState:
    """Play one game with ``agents[i]`` in seat ``i`` and return the final state."""
    if len(agents) != NUM_PLAYERS:
        raise ValueError(f"need exactly {NUM_PLAYERS} agents")
    state = new_game(seed, first_player=first_player)
    while not state.is_terminal:
        agent = agents[state.to_play]
        card = agent.act(observe(state, state.to_play))
        if card not in state.hands[state.to_play]:
            raise IllegalActionError(f"agent {agent.name!r} played {card}, not in hand")
        state = step(state, card)
    return state


@dataclass(frozen=True, slots=True)
class MatchResult:
    wins: int
    draws: int
    losses: int
    points_for: int
    points_against: int
    deal_scores: tuple[float, ...] = ()
    """Mean score over each duplicate pair of games, in deal order."""

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Win rate counting draws as half a win."""
        return (self.wins + self.draws / 2) / self.games


def play_match(agent: Agent, opponent: Agent, deals: int, seed: int = 0) -> MatchResult:
    """Play ``deals`` duplicate pairs: each deal twice, with seats swapped.

    Duplicate play cancels most of the luck of the deal, so far fewer games are
    needed to tell agents apart. Results are from ``agent``'s point of view.
    """
    deal_seeds = random.Random(seed).sample(range(2**31), deals)
    wins = draws = losses = points_for = points_against = 0
    deal_scores = []
    for deal in deal_seeds:
        deal_score = 0.0
        for seat in range(NUM_PLAYERS):
            seats = [opponent] * NUM_PLAYERS
            seats[seat] = agent
            final = play_game(seats, deal)
            won = winner(final)
            wins += won == seat
            draws += won is None
            losses += won is not None and won != seat
            points_for += final.scores[seat]
            points_against += sum(final.scores) - final.scores[seat]
            deal_score += 1.0 if won == seat else 0.5 if won is None else 0.0
        deal_scores.append(deal_score / NUM_PLAYERS)
    return MatchResult(wins, draws, losses, points_for, points_against, tuple(deal_scores))
