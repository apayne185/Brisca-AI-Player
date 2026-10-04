import random
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from brisca.arena import MatchResult
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


def fake_match(deal_scores: list[float]) -> object:
    """Stand-in for play_match that returns fixed per-deal results."""

    def play_match(*args: object, **kwargs: object) -> MatchResult:
        wins = 2 * sum(s > 0.5 for s in deal_scores)
        losses = 2 * sum(s < 0.5 for s in deal_scores)
        draws = 2 * len(deal_scores) - wins - losses
        return MatchResult(wins, draws, losses, 0, 0, tuple(deal_scores))

    return play_match


def test_first_model_becomes_champion() -> None:
    decision = promote(logged_policy())
    assert decision.promoted
    assert decision.reason == "no champion yet"
    champion = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    assert str(champion.version) == decision.version == "1"


def test_challenger_needs_a_significant_win(monkeypatch: pytest.MonkeyPatch) -> None:
    promote(logged_policy(0))

    monkeypatch.setattr(promotion, "play_match", fake_match([1.0] * 3 + [0.0] * 2))
    rejected = promote(logged_policy(1), deals=5)
    assert not rejected.promoted
    assert "does not significantly beat v1" in rejected.reason

    monkeypatch.setattr(promotion, "play_match", fake_match([1.0] * 10))
    accepted = promote(logged_policy(2), deals=10)
    assert accepted.promoted
    assert accepted.p_value == pytest.approx(1 / 1024)

    client = MlflowClient()
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version) == "3"
    assert client.get_model_version(REGISTERED_MODEL, "2").tags["promoted"] == "False"


def test_promotion_is_logged_as_a_run(isolated_mlflow: Path) -> None:
    promote(logged_policy())
    runs = mlflow.search_runs(experiment_names=["promotion"])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.reason"] == "no champion yet"
