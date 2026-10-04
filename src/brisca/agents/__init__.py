"""Brisca agents, all behind the ``Agent`` protocol."""

from collections.abc import Callable
from typing import Any

from brisca.agents.alphabeta import DeterminizedAlphaBetaAgent
from brisca.agents.base import Agent
from brisca.agents.heuristic import HeuristicAgent, HeuristicParams
from brisca.agents.ismcts import ISMCTSAgent
from brisca.agents.simple import GreedyAgent, RandomAgent


def _heuristic(**params: Any) -> HeuristicAgent:
    return HeuristicAgent(HeuristicParams(**params))


REGISTRY: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": _heuristic,
    "ismcts": ISMCTSAgent,
    "alphabeta": DeterminizedAlphaBetaAgent,
}


def make_agent(name: str, **kwargs: Any) -> Agent:
    """Build an agent by name, e.g. ``make_agent("ismcts", iterations=500, seed=0)``."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(REGISTRY)}") from None
    return factory(**kwargs)


__all__ = [
    "REGISTRY",
    "Agent",
    "DeterminizedAlphaBetaAgent",
    "GreedyAgent",
    "HeuristicAgent",
    "HeuristicParams",
    "ISMCTSAgent",
    "RandomAgent",
    "make_agent",
]
