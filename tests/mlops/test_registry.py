import random
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from brisca.encoding import action_mask, encode_observation
from brisca.mlops import promotion
from brisca.mlops.promotion import promote, sign_test
from brisca.mlops.tracking import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL,
    configure,
    git_tags,
    load_policy_agent,
    log_policy,
)
from brisca.observation import observe
from brisca.rl import ActorCritic
from brisca.tournament import AgentSpec, build_agent
from tests.helpers import random_playout


def logged_policy(seed: int = 0) -> str:
    configure("test")
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    with mlflow.start_run():
        return log_policy(ActorCritic(hidden=16), {"seed": seed})


def test_paired_sign_test_counts_differences_around_zero() -> None:
    assert sign_test([0.25, 0.5, -0.25, 0.0, 0.75], centre=0.0) == pytest.approx(5 / 16)


def test_panel_scores_cover_every_opponent() -> None:
    from brisca.agents import GreedyAgent

    scores = promotion.panel_scores(GreedyAgent(), ["random", "greedy"], deals=3, seed=0)
    assert len(scores) == 6
    assert scores[3:] == [0.5, 0.5, 0.5], "a deterministic agent ties itself on every deal"


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([1.0] * 5, 1 / 32),
        ([1.0] * 8 + [0.0] * 2, 56 / 1024),
        ([0.5] * 10, 1.0),
        ([0.0] * 5, 1.0),
        ([0.75, 0.25], 0.75),
    ],
)
def test_sign_test(scores: list[float], expected: float) -> None:
    assert sign_test(scores) == pytest.approx(expected)


def test_git_tags_record_the_commit() -> None:
    tags = git_tags()
    assert set(tags) <= {"git_sha", "git_dirty"}


def test_logged_policy_serves_through_pyfunc() -> None:
    uri = logged_policy()
    model = mlflow.pyfunc.load_model(uri)
    states = [s for s in random_playout(0, random.Random(0)) if not s.is_terminal][:5]
    observations = [observe(s, s.to_play) for s in states]
    actions = model.predict(
        {
            "obs": np.stack([encode_observation(o) for o in observations]),
            "mask": np.stack([action_mask(o) for o in observations]),
        }
    )
    for obs, action in zip(observations, actions, strict=True):
        assert action in [card.ordinal for card in obs.hand]


def test_logged_policy_loads_as_agent() -> None:
    agent = load_policy_agent(logged_policy())
    state = next(iter(random_playout(1, random.Random(1))))
    assert agent.act(observe(state, 0)) in state.hands[0]


def test_tournament_can_load_registered_models() -> None:
    promote(logged_policy())
    agent = build_agent(
        AgentSpec("champ", "ppo", {"model_uri": f"models:/{REGISTERED_MODEL}@champion"})
    )
    assert agent.name == "ppo"


def fake_panel(candidate: list[float], champion: list[float]) -> object:
    """Stand-in for panel_scores: the first call is the candidate, the second the champion."""
    results = iter([candidate, champion])

    def panel_scores(*args: object, **kwargs: object) -> list[float]:
        return next(results)

    return panel_scores


def test_first_model_becomes_champion() -> None:
    decision = promote(logged_policy())
    assert decision.promoted
    assert decision.reason == "no champion yet"
    champion = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    assert str(champion.version) == decision.version == "1"


def test_challenger_needs_a_significant_win(monkeypatch: pytest.MonkeyPatch) -> None:
    promote(logged_policy(0))

    # Better on 3 deals, worse on 2: not significant.
    monkeypatch.setattr(promotion, "panel_scores", fake_panel([1.0] * 3 + [0.0] * 2, [0.5] * 5))
    rejected = promote(logged_policy(1), deals=5)
    assert not rejected.promoted
    assert "does not significantly beat v1 against greedy/heuristic" in rejected.reason

    # Better on all 10 deals: p = 2^-10.
    monkeypatch.setattr(promotion, "panel_scores", fake_panel([0.75] * 10, [0.25] * 10))
    accepted = promote(logged_policy(2), deals=10)
    assert accepted.promoted
    assert accepted.p_value == pytest.approx(1 / 1024)
    assert (accepted.candidate_score, accepted.champion_score) == (0.75, 0.25)

    client = MlflowClient()
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version) == "3"
    assert client.get_model_version(REGISTERED_MODEL, "2").tags["promoted"] == "False"


def test_promotion_is_logged_as_a_run(isolated_mlflow: Path) -> None:
    promote(logged_policy())
    runs = mlflow.search_runs(experiment_names=["promotion"])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.reason"] == "no champion yet"
