"""Round-robin tournaments between agents, run in parallel with duplicate deals."""

from __future__ import annotations

import itertools
import multiprocessing
import random
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brisca.agents import make_agent
from brisca.agents.base import Agent
from brisca.arena import play_game
from brisca.cards import Card
from brisca.engine import winner
from brisca.observation import Observation


@dataclass(frozen=True)
class AgentSpec:
    """A named, reproducible agent configuration, e.g. ``ismcts`` with 1000 iterations."""

    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TournamentConfig:
    agents: tuple[AgentSpec, ...]
    deals: int = 100
    """Duplicate deals per pairing; each is played twice with seats swapped."""
    seed: int = 0

    @classmethod
    def from_toml(cls, path: str | Path) -> TournamentConfig:
        raw = tomllib.loads(Path(path).read_text())
        agents = tuple(AgentSpec(**spec) for spec in raw.pop("agents"))
        if len({a.id for a in agents}) != len(agents):
            raise ValueError("agent ids must be unique")
        return cls(agents=agents, **raw)


@dataclass(frozen=True, slots=True)
class GameRecord:
    agent0: str
    agent1: str
    deal_seed: int
    score0: int
    score1: int
    winner: int | None
    seconds0: float
    seconds1: float


def build_agent(spec: AgentSpec) -> Agent:
    if spec.type == "ppo":  # optional dependencies, imported only when needed
        if "model_uri" in spec.params:
            from brisca.mlops.tracking import configure, load_policy_agent

            configure("tournament")
            return load_policy_agent(**spec.params)
        from brisca.rl import PolicyAgent

        return PolicyAgent.from_checkpoint(**spec.params)
    if spec.type == "onnx":
        from brisca.onnx_policy import OnnxPolicyAgent

        return OnnxPolicyAgent(**spec.params)
    if spec.type == "llm":
        from brisca.llm.agent import LLMAgent

        return LLMAgent(**spec.params)
    return make_agent(spec.type, **spec.params)


class _Timed:
    """Wraps an agent to accumulate its thinking time."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.name = agent.name
        self.seconds = 0.0

    def act(self, obs: Observation) -> Card:
        start = time.perf_counter()
        card = self.agent.act(obs)
        self.seconds += time.perf_counter() - start
        return card


def _play_pairing(a: AgentSpec, b: AgentSpec, deal_seeds: list[int]) -> list[GameRecord]:
    agents = {a.id: build_agent(a), b.id: build_agent(b)}
    records = []
    for deal in deal_seeds:
        for first, second in ((a.id, b.id), (b.id, a.id)):
            seats = [_Timed(agents[first]), _Timed(agents[second])]
            final = play_game(seats, deal)
            records.append(
                GameRecord(
                    agent0=first,
                    agent1=second,
                    deal_seed=deal,
                    score0=final.scores[0],
                    score1=final.scores[1],
                    winner=winner(final),
                    seconds0=seats[0].seconds,
                    seconds1=seats[1].seconds,
                )
            )
    return records


def _chunks(items: list[int], size: int) -> Iterator[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_tournament(
    config: TournamentConfig, workers: int | None = None, chunk_size: int = 10
) -> list[GameRecord]:
    """Play every pair of agents on the same ``config.deals`` deals, from both seats.

    Every pairing uses the same deal seeds, so differences between agents are
    not down to some of them being dealt better cards.
    """
    deal_seeds = random.Random(config.seed).sample(range(2**31), config.deals)
    jobs = [
        (a, b, chunk)
        for a, b in itertools.combinations(config.agents, 2)
        for chunk in _chunks(deal_seeds, chunk_size)
    ]
    if workers == 1:
        return [r for job in jobs for r in _play_pairing(*job)]
    # spawn, not fork: forking a process that holds torch/DuckDB threads is unsafe.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_play_pairing, *job) for job in jobs]
        return [r for future in futures for r in future.result()]
