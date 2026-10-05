import math
import random
from pathlib import Path

import duckdb
import pytest

from brisca import Card, step
from brisca.agents.alphabeta import alphabeta
from brisca.detection.simulate import (
    BOT_STYLES,
    Profile,
    _endgame_optimal,
    sample_profile,
    simulate_player,
    stakes,
    think_time,
)
from brisca.observation import observe
from tests.helpers import make_state, random_playout


def profile(style: str | None, tempo: float = 1.5) -> Profile:
    return Profile(0, style is not None, style, "heuristic", 0.5, tempo)


def test_stakes_are_bounded_and_track_points_at_risk() -> None:
    cheap = observe(make_state("2C 4E 5B", "", trump="7O", stock="10B 7O", trick="6C"), 0)
    rich = observe(make_state("1C 2O 5B", "", trump="7O", stock="10B 7O", trick="3C"), 0)
    assert 0.0 <= stakes(cheap) < stakes(rich) <= 1.0


def test_bot_rate_controls_the_population() -> None:
    rng = random.Random(0)
    assert not any(sample_profile(i, 0.0, rng).is_bot for i in range(50))
    bots = [sample_profile(i, 1.0, rng) for i in range(200)]
    assert all(p.is_bot for p in bots)
    assert {p.bot_style for p in bots} == set(BOT_STYLES)


def test_timing_styles() -> None:
    obs = observe(make_state("1C 2O 5B", "", trump="7O", stock="10B 7O", trick="3C"), 0)
    forced = observe(make_state("1C", "", trump="7O", trick="3C"), 0)
    rng = random.Random(0)
    assert all(think_time(profile("naive"), obs, rng) < 0.2 for _ in range(50))
    assert think_time(profile(None), forced, random.Random(1)) < 2.0

    # The mimic style uses exactly the human timing model.
    for seed in range(20):
        assert think_time(profile("mimic"), obs, random.Random(seed)) == think_time(
            profile(None), obs, random.Random(seed)
        )


def test_endgame_optimality_matches_exhaustive_search() -> None:
    for seed in range(5):
        for state in random_playout(seed, random.Random(seed)):
            if state.stock or state.is_terminal or len(state.hands[state.to_play]) < 2:
                continue
            player, depth = state.to_play, sum(len(h) for h in state.hands)
            values: dict[Card, float] = {
                c: alphabeta(step(state, c), depth - 1, -math.inf, math.inf, player)
                for c in state.hands[player]
            }
            for card, value in values.items():
                assert _endgame_optimal(state, card) == (value == max(values.values()))


@pytest.mark.parametrize("style", [None, "naive", "mimic"])
def test_simulated_player_produces_consistent_events(style: str | None) -> None:
    moves, games = simulate_player(profile(style), seed=3)
    assert len(moves) == 20 * len(games)
    assert all(m.think_s > 0 for m in moves)
    assert all((m.endgame_optimal is not None) == m.endgame_decision for m in moves)
    assert all(m.endgame_decision == (m.stock_size == 0 and m.n_options > 1) for m in moves)
    timestamps = [m.ts for m in moves]
    assert timestamps == sorted(timestamps)


def test_population_tables(telemetry_db: Path) -> None:
    with duckdb.connect(str(telemetry_db), read_only=True) as con:
        players = con.sql("SELECT count(*), sum(is_bot::INT) FROM players").fetchone()
        assert players is not None
        assert players[0] == 50
        assert 0 < players[1] < 50
        orphans = con.sql(
            "SELECT count(*) FROM moves m ANTI JOIN games g USING (player_id, game_id)"
        ).fetchone()
        assert orphans == (0,)
