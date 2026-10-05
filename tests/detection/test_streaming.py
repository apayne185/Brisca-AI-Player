import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from brisca.detection.features import load_dataset
from brisca.detection.model import Detector, fit_detector
from brisca.detection.online import ONLINE_FEATURES
from brisca.streaming import (
    SCORES_TOPIC,
    Scorer,
    events_from_db,
    live_events,
    produce,
)


class FakeProducer:
    def __init__(self, log: list[str] | None = None) -> None:
        self.messages: list[tuple[str, bytes | None, bytes]] = []
        self.log = log if log is not None else []

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self.messages.append((topic, key, value))

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float) -> int:
        self.log.append("flush")
        return 0


class FakeMessage:
    def __init__(self, value: bytes | None, error: Any = None) -> None:
        self._value, self._error = value, error

    def value(self) -> bytes | None:
        return self._value

    def error(self) -> Any:
        return self._error


class FakeConsumer:
    def __init__(self, messages: list[FakeMessage], log: list[str]) -> None:
        self.messages = list(messages)
        self.log = log

    def poll(self, timeout: float) -> FakeMessage | None:
        return self.messages.pop(0) if self.messages else None

    def commit(self, asynchronous: bool = True) -> None:
        self.log.append("commit")


@pytest.fixture(scope="module")
def detector(telemetry_db: Path) -> Detector:
    data = load_dataset(telemetry_db)
    detector = fit_detector(data.select(ONLINE_FEATURES), data.y, data.groups, ONLINE_FEATURES)
    detector.threshold = 0.5
    return detector


def test_produce_keys_events_by_player(telemetry_db: Path) -> None:
    producer = FakeProducer()
    sent = produce(events_from_db(telemetry_db), producer, limit=50, rate=1e6)
    assert sent == 50
    for _, key, value in producer.messages:
        assert key == str(json.loads(value)["player_id"]).encode()


def test_scorer_scores_every_session_once(telemetry_db: Path, detector: Detector) -> None:
    log: list[str] = []
    events = list(events_from_db(telemetry_db))
    messages = [FakeMessage(json.dumps(e).encode()) for e in events]
    messages.insert(10, FakeMessage(b"not json"))
    messages.insert(20, FakeMessage(None, error="broker hiccup"))

    producer = FakeProducer(log)
    scorer = Scorer(detector, producer)
    read = scorer.run(FakeConsumer(messages, log), commit_every=500, idle_polls=1)

    sessions = {(e["player_id"], e["session_id"]) for e in events}
    results = [json.loads(v) for topic, _, v in producer.messages if topic == SCORES_TOPIC]
    assert read == len(events) + 1  # the malformed message counts as read
    assert {(r["player_id"], r["session_id"]) for r in results} == sessions
    assert len(results) == len(sessions)
    assert all(0 <= r["probability"] <= 1 for r in results)

    # Same decision as scoring the batch features directly.
    data = load_dataset(telemetry_db)
    X = data.select(ONLINE_FEATURES)
    assert sum(r["flagged"] for r in results) == int(detector.flag(X).sum())

    assert scorer.malformed._value.get() == 1
    assert log[-2:] == ["flush", "commit"], "results are flushed before offsets are committed"


def test_scorer_rejects_batch_only_features(telemetry_db: Path) -> None:
    data = load_dataset(telemetry_db)
    detector = fit_detector(data.X, data.y, data.groups, data.features)
    with pytest.raises(ValueError, match="cannot compute these features online"):
        Scorer(detector, FakeProducer())


def test_live_events_close_every_session() -> None:
    events = []
    for event in live_events(seed=3):
        events.append(event)
        if event["player_id"] == 2:
            break
    first = [e for e in events if e["player_id"] == 0]
    sessions = {e["session_id"] for e in first}
    assert {e["session_id"] for e in first if e["type"] == "session_end"} == sessions
    assert first[-1]["type"] == "session_end"


@pytest.mark.skipif(
    "BRISCA_KAFKA_BOOTSTRAP" not in os.environ, reason="needs a Kafka broker (CI provides one)"
)
def test_round_trip_through_a_real_broker(telemetry_db: Path, detector: Detector) -> None:
    from brisca.streaming import ensure_topics, kafka_consumer, kafka_producer

    bootstrap = os.environ["BRISCA_KAFKA_BOOTSTRAP"]
    run = uuid.uuid4().hex[:8]
    events_topic, scores_topic = f"test.events.{run}", f"test.scores.{run}"
    ensure_topics(bootstrap, [events_topic, scores_topic], partitions=3)

    events = [e for e in events_from_db(telemetry_db) if e["player_id"] < 5]
    produce(events, kafka_producer(bootstrap), topic=events_topic)

    scorer = Scorer(detector, kafka_producer(bootstrap), out_topic=scores_topic)
    consumer = kafka_consumer(bootstrap, topic=events_topic, group=f"test-{run}")
    try:
        scorer.run(consumer, max_messages=len(events))
    finally:
        consumer.close()

    sessions = {(e["player_id"], e["session_id"]) for e in events}
    results = kafka_consumer(bootstrap, topic=scores_topic, group=f"reader-{run}")
    seen = set()
    try:
        for _ in range(60):
            message = results.poll(1.0)
            if message is not None and not message.error():
                r = json.loads(message.value())
                seen.add((r["player_id"], r["session_id"]))
            if seen == sessions:
                break
    finally:
        results.close()
    assert seen == sessions
