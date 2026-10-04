import itertools
from pathlib import Path

import pytest

from brisca.agents import GreedyAgent
from brisca.observation import observe
from brisca.tournament import (
    AgentSpec,
    TournamentConfig,
    _Timed,
    build_agent,
    run_tournament,
)
from tests.helpers import make_state

CONFIG = TournamentConfig(
    agents=(
        AgentSpec("random", "random", {"seed": 0}),
        AgentSpec("greedy", "greedy"),
        AgentSpec("heuristic", "heuristic"),
    ),
    deals=4,
    seed=0,
)


def test_config_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "t.toml"
    path.write_text(
        'deals = 7\n[[agents]]\nid = "r"\ntype = "random"\nparams = { seed = 1 }\n'
        '[[agents]]\nid = "g"\ntype = "greedy"\n'
    )
    config = TournamentConfig.from_toml(path)
    assert config.deals == 7
    assert config.agents == (AgentSpec("r", "random", {"seed": 1}), AgentSpec("g", "greedy"))


def test_config_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "t.toml"
    path.write_text(
        '[[agents]]\nid = "x"\ntype = "random"\n[[agents]]\nid = "x"\ntype = "greedy"\n'
    )
    with pytest.raises(ValueError, match="unique"):
        TournamentConfig.from_toml(path)


@pytest.mark.parametrize("workers", [1, 2])
def test_every_pairing_plays_the_same_deals_from_both_seats(workers: int) -> None:
    records = run_tournament(CONFIG, workers=workers, chunk_size=3)
    assert len(records) == 3 * CONFIG.deals * 2

    deals_by_pair: dict[frozenset[str], set[int]] = {}
    for r in records:
        deals_by_pair.setdefault(frozenset((r.agent0, r.agent1)), set()).add(r.deal_seed)
        assert r.score0 + r.score1 == 120
        assert r.seconds0 >= 0
        assert r.seconds1 >= 0
    assert len({frozenset(d) for d in deals_by_pair.values()}) == 1, "same deals everywhere"

    for a, b in itertools.permutations(["random", "greedy", "heuristic"], 2):
        seated = [r for r in records if (r.agent0, r.agent1) == (a, b)]
        assert len(seated) == CONFIG.deals


def test_timed_agent_accumulates_thinking_time() -> None:
    timed = _Timed(GreedyAgent())
    obs = observe(make_state("1O 2C", "3E 4B", trump="7O", stock="10B 7O"), 0)
    timed.act(obs)
    timed.act(obs)
    assert timed.name == "greedy"
    assert timed.seconds > 0


def test_build_agent_loads_ppo_checkpoints(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from brisca.rl import ActorCritic, PolicyAgent, save_checkpoint

    path = tmp_path / "ppo.pt"
    save_checkpoint(ActorCritic(hidden=8), path, {})
    agent = build_agent(AgentSpec("p", "ppo", {"path": str(path)}))
    assert isinstance(agent, PolicyAgent)
