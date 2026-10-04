"""Every agent must clearly beat random play. Seeded, so results are deterministic.

Run with ``make test-slow``; CI runs these in a separate job.
"""

from typing import Any

import pytest

from brisca.agents import RandomAgent, make_agent
from brisca.arena import play_match

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("name", "kwargs", "deals", "min_score"),
    [
        ("greedy", {}, 300, 0.75),
        ("heuristic", {}, 300, 0.75),
        ("alphabeta", {"samples": 5, "depth": 4, "seed": 0}, 20, 0.7),
        ("ismcts", {"iterations": 200, "seed": 0}, 10, 0.7),
    ],
)
def test_agent_beats_random(
    name: str, kwargs: dict[str, Any], deals: int, min_score: float
) -> None:
    result = play_match(make_agent(name, **kwargs), RandomAgent(seed=1), deals=deals, seed=0)
    assert result.score >= min_score, result
