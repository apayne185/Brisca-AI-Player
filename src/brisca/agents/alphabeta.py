"""Determinized alpha-beta search (Perfect Information Monte Carlo).

Sample several hidden states consistent with the observation, solve each as a
perfect-information game with depth-limited alpha-beta, and play the card with
the best average value. PIMC is a strong practical baseline for trick-taking
games, though it is known to overestimate how much it can exploit hidden
information ("strategy fusion").

Once the stock is empty Brisca becomes a perfect-information game (every unseen
card is in the opponent's hand), so the search runs to the end and is exact.
"""

from __future__ import annotations

import math

from brisca.cards import Card
from brisca.engine import NUM_PLAYERS, GameState, Seed, make_rng, step
from brisca.observation import Observation, determinize


def evaluate(state: GameState, player: int) -> float:
    """Point difference from ``player``'s perspective."""
    return float(state.scores[player] - state.scores[(player + 1) % NUM_PLAYERS])


def alphabeta(state: GameState, depth: int, alpha: float, beta: float, player: int) -> float:
    """Minimax value of ``state`` for ``player``, searching ``depth`` plies.

    Turns do not strictly alternate (the trick winner leads), so each node
    maximises or minimises according to who is to play.
    """
    if depth == 0 or state.is_terminal:
        return evaluate(state, player)

    moves = sorted(state.hands[state.to_play], key=lambda c: -c.points)
    if state.to_play == player:
        value = -math.inf
        for move in moves:
            value = max(value, alphabeta(step(state, move), depth - 1, alpha, beta, player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = math.inf
    for move in moves:
        value = min(value, alphabeta(step(state, move), depth - 1, alpha, beta, player))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


class DeterminizedAlphaBetaAgent:
    name = "alphabeta"

    def __init__(self, samples: int = 20, depth: int = 6, seed: Seed = None) -> None:
        if samples < 1 or depth < 1:
            raise ValueError("samples and depth must be positive")
        self.samples = samples
        self.depth = depth
        self._rng = make_rng(seed)

    def act(self, obs: Observation) -> Card:
        if len(obs.hand) == 1:
            return obs.hand[0]

        remaining_plies = len(obs.hand) + obs.opponent_hand_size
        exact = obs.stock_size == 0
        depth = remaining_plies if exact else self.depth
        samples = 1 if exact else self.samples

        totals = dict.fromkeys(obs.hand, 0.0)
        for _ in range(samples):
            state = determinize(obs, self._rng)
            for card in obs.hand:
                totals[card] += alphabeta(
                    step(state, card), depth - 1, -math.inf, math.inf, obs.player
                )
        return max(obs.hand, key=totals.__getitem__)
