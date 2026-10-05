"""Prometheus metrics, on a registry per app so tests stay isolated."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


@dataclass(frozen=True)
class Metrics:
    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    decision_latency: Histogram
    games: Counter
    bot_scores: Histogram
    flags: Counter
    feature_psi: Gauge

    @classmethod
    def create(cls) -> Metrics:
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests=Counter(
                "brisca_http_requests",
                "HTTP requests",
                ["route", "method", "status"],
                registry=registry,
            ),
            latency=Histogram(
                "brisca_http_request_seconds",
                "HTTP request latency",
                ["route"],
                registry=registry,
            ),
            decision_latency=Histogram(
                "brisca_agent_decision_seconds",
                "Time for an agent to choose a card",
                ["agent"],
                buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
                registry=registry,
            ),
            games=Counter(
                "brisca_demo_games", "Demo games by event", ["event", "agent"], registry=registry
            ),
            bot_scores=Histogram(
                "brisca_bot_score",
                "Calibrated bot probabilities served",
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
                registry=registry,
            ),
            flags=Counter("brisca_bot_flags", "Sessions flagged as bots", registry=registry),
            feature_psi=Gauge(
                "brisca_bot_feature_psi",
                "PSI of recently scored sessions vs training, per feature",
                ["feature"],
                registry=registry,
            ),
        )
