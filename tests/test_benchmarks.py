"""Engine throughput benchmarks. Run with ``make bench``."""

import random

from pytest_benchmark.fixture import BenchmarkFixture

from brisca import determinize, legal_actions, new_game, observe, step


def play_random_game(seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    state = new_game(rng)
    while not state.is_terminal:
        state = step(state, rng.choice(legal_actions(state)))
    return state.scores


def test_random_game(benchmark: BenchmarkFixture) -> None:
    scores = benchmark(play_random_game, 0)
    assert sum(scores) == 120


def test_determinize(benchmark: BenchmarkFixture) -> None:
    obs = observe(new_game(0), player=0)
    rng = random.Random(0)
    state = benchmark(determinize, obs, rng)
    assert state.hands[0] == obs.hand
