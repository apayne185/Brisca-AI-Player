"""Ratings and uncertainty from head-to-head game results.

Ratings come from a Bradley-Terry model fitted by maximum likelihood, reported
on the Elo scale. Unlike incremental Elo updates, the fit does not depend on
the order games were played. Confidence intervals come from a bootstrap that
resamples whole deals, keeping each duplicate pair together.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ELO_SCALE = 400 / math.log(10)


@dataclass(frozen=True, slots=True)
class Outcome:
    """One game: ``score_a`` is 1 if ``a`` won, 0.5 for a draw, 0 if ``b`` won."""

    a: str
    b: str
    score_a: float
    deal: int


def bradley_terry(
    outcomes: Sequence[Outcome],
    players: Sequence[str],
    anchor: str | None = None,
    prior_games: float = 1.0,
    iterations: int = 1000,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Elo-scale ratings by minorization-maximization (Hunter, 2004).

    ``prior_games`` adds that many virtual draws between every pair, which keeps
    ratings finite when an agent wins or loses every game. Ratings are shifted
    so ``anchor`` sits at 0 (or the mean at 0 if no anchor is given).
    """
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = np.full((n, n), prior_games / 2)
    np.fill_diagonal(wins, 0.0)
    for o in outcomes:
        i, j = index[o.a], index[o.b]
        wins[i, j] += o.score_a
        wins[j, i] += 1 - o.score_a
    games = wins + wins.T
    total_wins = wins.sum(axis=1)

    strength = np.ones(n)
    for _ in range(iterations):
        denom = (games / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = total_wins / denom
        updated /= np.exp(np.log(updated).mean())
        if np.abs(updated - strength).max() < tol:
            strength = updated
            break
        strength = updated

    elo = ELO_SCALE * np.log(strength)
    elo -= elo[index[anchor]] if anchor is not None else elo.mean()
    return {p: float(elo[index[p]]) for p in players}


def bootstrap_intervals(
    outcomes: Sequence[Outcome],
    players: Sequence[str],
    anchor: str | None = None,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals for each rating, resampling deals within each pairing."""
    by_pair_deal: dict[tuple[str, str], dict[int, list[Outcome]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for o in outcomes:
        pair = (o.a, o.b) if o.a < o.b else (o.b, o.a)
        by_pair_deal[pair][o.deal].append(o)

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {p: [] for p in players}
    for _ in range(samples):
        resampled: list[Outcome] = []
        for deals in by_pair_deal.values():
            keys = list(deals)
            for key in rng.choices(keys, k=len(keys)):
                resampled.extend(deals[key])
        for player, rating in bradley_terry(resampled, players, anchor).items():
            draws[player].append(rating)

    tail = (1 - confidence) / 2 * 100
    return {
        p: (float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail)))
        for p, v in draws.items()
    }


def wilson_interval(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win rate (draws counted as half a win)."""
    if n == 0:
        return (0.0, 1.0)
    centre = (score + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(score * (1 - score) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - half, centre + half)
