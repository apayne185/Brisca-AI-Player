"""Real-time bot scoring over Kafka.

    producer --(brisca.events)--> scorer --(brisca.bot-scores)--> downstream

The producer publishes gameplay events keyed by player, so each player's
events stay ordered within one partition. The scorer keeps running per-session
aggregates (``brisca.detection.online``), scores each session as it closes and
publishes the result. Offsets are committed only after results are produced,
giving at-least-once delivery; a replayed session yields the same score, so
downstream consumers can de-duplicate on (player_id, session_id).

Requires the ``stream`` extra (``confluent-kafka``).
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import duckdb
import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from brisca.detection.model import Detector
from brisca.detection.online import ONLINE_FEATURES, SessionKey, SessionTracker

log = logging.getLogger("brisca.streaming")

EVENTS_TOPIC = "brisca.events"
SCORES_TOPIC = "brisca.bot-scores"
Event = dict[str, Any]


class ProducerLike(Protocol):
    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None: ...
    def poll(self, timeout: float) -> int: ...
    def flush(self, timeout: float) -> int: ...


class MessageLike(Protocol):
    def value(self) -> bytes | None: ...
    def error(self) -> Any: ...


class ConsumerLike(Protocol):
    def poll(self, timeout: float) -> MessageLike | None: ...
    def commit(self, asynchronous: bool = True) -> Any: ...


# --- Event sources -----------------------------------------------------------


def _player_events(moves: Iterable[dict[str, Any]], games: Iterable[dict[str, Any]]) -> list[Event]:
    """One player's events in time order, with a ``session_end`` after each session."""
    events: list[Event] = [{"type": "move", **m} for m in moves]
    events += [{"type": "game_end", **g} for g in games]
    events.sort(key=lambda e: e["ts"] if e["type"] == "move" else e["end_ts"])
    ends: dict[int, Event] = {}
    for e in events:
        ends[e["session_id"]] = e
    out: list[Event] = []
    for e in events:
        out.append(e)
        if ends[e["session_id"]] is e:
            out.append(
                {"type": "session_end", "player_id": e["player_id"], "session_id": e["session_id"]}
            )
    return out


def events_from_db(db: str | Path) -> Iterator[Event]:
    """Replay simulated telemetry from DuckDB, player by player."""
    with duckdb.connect(str(db), read_only=True) as con:
        moves = con.execute("SELECT * FROM moves ORDER BY player_id, ts").fetchall()
        move_cols = [d[0] for d in con.description]
        games = con.execute("SELECT * FROM games ORDER BY player_id, end_ts").fetchall()
        game_cols = [d[0] for d in con.description]
    move_rows = (dict(zip(move_cols, r, strict=True)) for r in moves)
    by_player_moves = itertools.groupby(move_rows, lambda r: r["player_id"])
    games_by_player: dict[int, list[dict[str, Any]]] = {}
    for row in games:
        g = dict(zip(game_cols, row, strict=True))
        games_by_player.setdefault(g["player_id"], []).append(g)
    for player_id, player_moves in by_player_moves:
        yield from _player_events(player_moves, games_by_player.get(player_id, []))


def live_events(bot_rate: float = 0.15, seed: int = 0) -> Iterator[Event]:
    """Simulate players indefinitely, as a stand-in for a live platform."""
    from brisca.detection.simulate import sample_profile, simulate_player

    rng = random.Random(seed)
    for player_id in itertools.count():
        profile = sample_profile(player_id, bot_rate, rng)
        moves, games = simulate_player(profile, seed * 1_000_003 + player_id)
        yield from _player_events((asdict(m) for m in moves), (asdict(g) for g in games))


def produce(
    events: Iterable[Event],
    producer: ProducerLike,
    topic: str = EVENTS_TOPIC,
    rate: float | None = None,
    limit: int | None = None,
) -> int:
    """Publish events keyed by player, optionally paced to ``rate`` events per second."""
    sent = 0
    start = time.monotonic()
    for event in itertools.islice(events, limit):
        producer.produce(
            topic, key=str(event["player_id"]).encode(), value=json.dumps(event).encode()
        )
        producer.poll(0)
        sent += 1
        if rate:
            ahead = sent / rate - (time.monotonic() - start)
            if ahead > 0:
                time.sleep(ahead)
    producer.flush(30)
    return sent


# --- Scoring -----------------------------------------------------------------


class Scorer:
    """Turns a stream of gameplay events into a stream of session scores."""

    def __init__(
        self,
        detector: Detector,
        producer: ProducerLike,
        out_topic: str = SCORES_TOPIC,
        tracker: SessionTracker | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        unsupported = set(detector.features) - set(ONLINE_FEATURES)
        if unsupported:
            raise ValueError(f"cannot compute these features online: {sorted(unsupported)}")
        self.detector = detector
        self.producer = producer
        self.out_topic = out_topic
        self.tracker = tracker or SessionTracker()
        registry = registry or CollectorRegistry()
        self.registry = registry
        self.events = Counter(
            "brisca_stream_events", "Events consumed", ["type"], registry=registry
        )
        self.malformed = Counter("brisca_stream_malformed", "Undecodable events", registry=registry)
        self.scored = Counter("brisca_stream_sessions_scored", "Sessions scored", registry=registry)
        self.flagged = Counter(
            "brisca_stream_sessions_flagged", "Sessions flagged", registry=registry
        )
        self.scores = Histogram(
            "brisca_stream_bot_score",
            "Calibrated bot probability of scored sessions",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
            registry=registry,
        )
        self.open_sessions = Gauge(
            "brisca_stream_open_sessions", "Sessions being aggregated", registry=registry
        )

    def handle(self, event: Event) -> list[Event]:
        self.events.labels(event["type"]).inc()
        results = [self._score(key, features) for key, features in self.tracker.on_event(event)]
        self.open_sessions.set(len(self.tracker.open))
        return results

    def _score(self, key: SessionKey, features: dict[str, float | None]) -> Event:
        values = [features[f] for f in self.detector.features]
        row = np.array([[np.nan if v is None else v for v in values]], dtype=np.float64)
        probability = float(self.detector.predict_proba(row)[0])
        flagged = bool(self.detector.flag(row)[0])
        result = {
            "player_id": key[0],
            "session_id": key[1],
            "probability": probability,
            "flagged": flagged,
            "features": features,
        }
        self.producer.produce(
            self.out_topic, key=str(key[0]).encode(), value=json.dumps(result).encode()
        )
        self.scored.inc()
        self.scores.observe(probability)
        if flagged:
            self.flagged.inc()
        return result

    def run(
        self,
        consumer: ConsumerLike,
        max_messages: int | None = None,
        commit_every: int = 100,
        idle_polls: int | None = None,
    ) -> int:
        """Consume until ``max_messages`` (or ``idle_polls`` empty polls); returns messages read."""
        processed = since_commit = empty = 0
        while max_messages is None or processed < max_messages:
            message = consumer.poll(1.0)
            if message is None:
                empty += 1
                if idle_polls is not None and empty >= idle_polls:
                    break
                continue
            empty = 0
            if message.error():
                log.warning("consumer error: %s", message.error())
                continue
            processed += 1
            since_commit += 1
            try:
                event = json.loads(message.value() or b"")
                self.handle(event)
            except (ValueError, KeyError, TypeError) as exc:  # poison message: skip, count
                self.malformed.inc()
                log.warning("skipping malformed event: %s", exc)
            if since_commit >= commit_every:
                self._commit(consumer)
                since_commit = 0
        self._commit(consumer)
        return processed

    def _commit(self, consumer: ConsumerLike) -> None:
        # Results must be delivered before the input offsets are committed.
        self.producer.flush(30)
        try:
            consumer.commit(asynchronous=False)
        except Exception as exc:  # e.g. nothing to commit yet
            log.debug("commit skipped: %s", exc)


# --- Kafka wiring --------------------------------------------------------------


def kafka_producer(bootstrap: str) -> ProducerLike:
    from confluent_kafka import Producer

    return Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True, "linger.ms": 20})


def kafka_consumer(bootstrap: str, topic: str = EVENTS_TOPIC, group: str = "bot-scorer") -> Any:
    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    return consumer


def ensure_topics(bootstrap: str, topics: Iterable[str], partitions: int = 6) -> None:
    from confluent_kafka.admin import AdminClient
    from confluent_kafka.cimpl import NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    missing = [NewTopic(t, num_partitions=partitions) for t in topics if t not in existing]
    for topic, future in admin.create_topics(missing).items() if missing else []:
        try:
            future.result()
            log.info("created topic %s", topic)
        except Exception as exc:  # created concurrently by another process
            log.info("topic %s: %s", topic, exc)
