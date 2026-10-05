import math
from pathlib import Path

import duckdb
import pytest

from brisca.detection.features import SESSION_FEATURES_SQL
from brisca.detection.online import ONLINE_FEATURES, SessionTracker
from brisca.streaming import events_from_db


def batch_features(db: Path) -> dict[tuple[int, int], dict[str, float | None]]:
    with duckdb.connect(str(db), read_only=True) as con:
        cursor = con.execute(SESSION_FEATURES_SQL)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, r, strict=True)) for r in cursor.fetchall()]
    return {(r["player_id"], r["session_id"]): {f: r[f] for f in ONLINE_FEATURES} for r in rows}


def test_streaming_features_match_training_features(telemetry_db: Path) -> None:
    """No training/serving skew: online aggregates equal the SQL used for training."""
    tracker = SessionTracker()
    online: dict[tuple[int, int], dict[str, float | None]] = {}
    for event in events_from_db(telemetry_db):
        online.update(tracker.on_event(event))

    batch = batch_features(telemetry_db)
    assert online.keys() == batch.keys()
    for key, expected in batch.items():
        for feature, value in expected.items():
            got = online[key][feature]
            if value is None:
                assert got is None, (key, feature)
            else:
                assert got is not None
                assert math.isclose(got, value, rel_tol=1e-12), (key, feature, got, value)


def move(player: int = 1, session: int = 0, ts: float = 0.0, **overrides: object) -> dict:  # type: ignore[type-arg]
    return {
        "type": "move",
        "player_id": player,
        "session_id": session,
        "ts": ts,
        "agrees_heuristic": True,
        "agrees_greedy": False,
        "endgame_decision": False,
        "endgame_optimal": None,
        **overrides,
    }


def test_session_end_closes_and_late_events_are_dropped() -> None:
    tracker = SessionTracker()
    assert tracker.on_event(move()) == []
    assert tracker.on_event({"type": "game_end", "player_id": 1, "session_id": 0,
                             "end_ts": 5.0, "points": 70, "won": True}) == []  # fmt: skip
    [(key, features)] = tracker.on_event({"type": "session_end", "player_id": 1, "session_id": 0})
    assert key == (1, 0)
    assert features == {
        "agree_heuristic": 1.0,
        "agree_greedy": 0.0,
        "endgame_accuracy": None,
        "avg_points": 70.0,
        "win_rate": 1.0,
    }
    assert tracker.on_event(move(ts=6.0)) == []
    assert tracker.dropped == 1
    assert not tracker.open


def test_idle_sessions_time_out_in_event_time() -> None:
    tracker = SessionTracker(idle_timeout=60)
    tracker.on_event(move(player=1, ts=0))
    closed = tracker.on_event(move(player=2, ts=100))
    assert [key for key, _ in closed] == [(1, 0)]
    assert (2, 0) in tracker.open


def test_closed_session_memory_is_bounded() -> None:
    tracker = SessionTracker(remember_closed=2)
    for player in range(5):
        tracker.on_event(move(player=player))
        tracker.on_event({"type": "session_end", "player_id": player, "session_id": 0})
    assert len(tracker._closed) == 2


def test_unknown_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown event type"):
        SessionTracker().on_event({"type": "chat", "player_id": 1, "session_id": 0})
