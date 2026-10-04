import math
import random

import pytest

from brisca.ratings import (
    ELO_SCALE,
    Outcome,
    bootstrap_intervals,
    bradley_terry,
    wilson_interval,
)


def simulate(true_elo: dict[str, float], games_per_pair: int, seed: int = 0) -> list[Outcome]:
    rng = random.Random(seed)
    players = list(true_elo)
    outcomes = []
    deal = 0
    for i, a in enumerate(players):
        for b in players[i + 1 :]:
            p_a = 1 / (1 + math.exp(-(true_elo[a] - true_elo[b]) / ELO_SCALE))
            for _ in range(games_per_pair):
                outcomes.append(Outcome(a, b, float(rng.random() < p_a), deal))
                deal += 1
    return outcomes


def test_recovers_known_ratings() -> None:
    truth = {"weak": 0.0, "mid": 150.0, "strong": 300.0}
    ratings = bradley_terry(simulate(truth, 4000), list(truth), anchor="weak", prior_games=0)
    for player, elo in truth.items():
        assert ratings[player] == pytest.approx(elo, abs=20)


def test_even_results_give_equal_ratings() -> None:
    outcomes = [Outcome("a", "b", 0.5, d) for d in range(10)]
    ratings = bradley_terry(outcomes, ["a", "b"])
    assert ratings["a"] == pytest.approx(0.0, abs=1e-6)
    assert ratings["b"] == pytest.approx(0.0, abs=1e-6)


def test_prior_keeps_a_clean_sweep_finite() -> None:
    outcomes = [Outcome("a", "b", 1.0, d) for d in range(20)]
    ratings = bradley_terry(outcomes, ["a", "b"], anchor="b")
    assert ratings["b"] == 0.0
    assert 0 < ratings["a"] < 1000


def test_bootstrap_interval_brackets_estimate_and_shrinks_with_data() -> None:
    truth = {"a": 0.0, "b": 100.0}

    def width(games: int) -> float:
        outcomes = simulate(truth, games)
        lo, hi = bootstrap_intervals(outcomes, ["a", "b"], anchor="a", samples=200)["b"]
        estimate = bradley_terry(outcomes, ["a", "b"], anchor="a")["b"]
        assert lo <= estimate <= hi
        return hi - lo

    assert width(2000) < width(100) / 2


@pytest.mark.parametrize(
    ("score", "n", "expected"),
    [(0.5, 100, (0.404, 0.596)), (1.0, 10, (0.722, 1.0)), (0.0, 0, (0.0, 1.0))],
)
def test_wilson_interval(score: float, n: int, expected: tuple[float, float]) -> None:
    lo, hi = wilson_interval(score, n)
    assert lo == pytest.approx(expected[0], abs=1e-3)
    assert hi == pytest.approx(expected[1], abs=1e-3)
