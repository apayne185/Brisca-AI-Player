"""Incremental session features for real-time scoring.

Training computes features in SQL over complete sessions
(``brisca.detection.features``). In production, events arrive one at a time,
so the same features are maintained as running aggregates. Any difference
between the two definitions is training/serving skew; a test checks that both
produce identical values on the same telemetry.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from brisca.detection.features import DECISION_FEATURES

SessionKey = tuple[int, int]
"""(player_id, session_id)"""
Features = dict[str, float | None]
Closed = tuple[SessionKey, Features]

ONLINE_FEATURES = DECISION_FEATURES
"""Features that can be maintained incrementally (timing features are batch-only for now)."""


@dataclass
class SessionAggregate:
    moves: int = 0
    agree_heuristic: int = 0
    agree_greedy: int = 0
    endgame_decisions: int = 0
    endgame_optimal: int = 0
    games: int = 0
    points: int = 0
    wins: int = 0
    last_ts: float = 0.0

    def add_move(self, event: Mapping[str, Any]) -> None:
        self.moves += 1
        self.agree_heuristic += bool(event["agrees_heuristic"])
        self.agree_greedy += bool(event["agrees_greedy"])
        if event["endgame_decision"]:
            self.endgame_decisions += 1
            self.endgame_optimal += bool(event["endgame_optimal"])
        self.last_ts = max(self.last_ts, float(event["ts"]))

    def add_game(self, event: Mapping[str, Any]) -> None:
        self.games += 1
        self.points += int(event["points"])
        self.wins += bool(event["won"])
        self.last_ts = max(self.last_ts, float(event["end_ts"]))

    def features(self) -> Features:
        """Same definitions as SESSION_FEATURES_SQL; ``None`` where SQL yields NULL."""

        def ratio(num: int, den: int) -> float | None:
            return num / den if den else None

        return {
            "agree_heuristic": ratio(self.agree_heuristic, self.moves),
            "agree_greedy": ratio(self.agree_greedy, self.moves),
            "endgame_accuracy": ratio(self.endgame_optimal, self.endgame_decisions),
            "avg_points": ratio(self.points, self.games),
            "win_rate": ratio(self.wins, self.games),
        }


class SessionTracker:
    """Routes events to per-session aggregates and emits features when a session closes.

    A session closes on an explicit ``session_end`` event, or when no event has
    been seen for ``idle_timeout`` seconds of event time (a player who simply
    disappears). Out-of-order events for an already-closed session are dropped.
    """

    def __init__(self, idle_timeout: float = 1800.0, remember_closed: int = 100_000) -> None:
        self.idle_timeout = idle_timeout
        self.remember_closed = remember_closed
        self.open: dict[SessionKey, SessionAggregate] = {}
        self._closed: OrderedDict[SessionKey, None] = OrderedDict()  # bounded, oldest first
        self.dropped = 0

    def on_event(self, event: Mapping[str, Any]) -> list[Closed]:
        key = (int(event["player_id"]), int(event["session_id"]))
        if key in self._closed:
            self.dropped += 1
            return []
        kind = event["type"]
        if kind == "session_end":
            return [self._close(key)] if key in self.open else []
        aggregate = self.open.setdefault(key, SessionAggregate())
        if kind == "move":
            aggregate.add_move(event)
        elif kind == "game_end":
            aggregate.add_game(event)
        else:
            raise ValueError(f"unknown event type {kind!r}")
        return self.expire(now=aggregate.last_ts)

    def expire(self, now: float) -> list[Closed]:
        """Close sessions idle for longer than ``idle_timeout`` (in event time)."""
        stale = [k for k, a in self.open.items() if now - a.last_ts > self.idle_timeout]
        return [self._close(k) for k in stale]

    def _close(self, key: SessionKey) -> Closed:
        self._closed[key] = None
        if len(self._closed) > self.remember_closed:
            self._closed.popitem(last=False)
        return key, self.open.pop(key).features()
