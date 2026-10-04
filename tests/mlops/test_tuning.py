import tomllib
from pathlib import Path

import mlflow
import optuna
import pytest

from brisca.agents import HeuristicParams
from brisca.cli import main
from brisca.mlops.tuning import (
    _tracked,
    best_heuristic_params,
    best_ppo_overrides,
    tune,
)
from brisca.rl.ppo import PPOConfig


@pytest.fixture
def storage(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'optuna.db'}"


def test_heuristic_search_finds_valid_params(storage: str) -> None:
    study = tune("heuristic", trials=4, storage=storage, deals=3)
    assert len(study.trials) == 4
    assert 0.0 <= study.best_value <= 1.0
    assert isinstance(best_heuristic_params(study), HeuristicParams)
    assert len(mlflow.search_runs(experiment_names=["tune-heuristic"])) == 4


def test_studies_resume_from_storage(storage: str) -> None:
    tune("heuristic", trials=2, storage=storage, deals=2)
    assert len(tune("heuristic", trials=2, storage=storage, deals=2).trials) == 4


def test_parallel_workers_share_one_study(tmp_path: Path) -> None:
    study = tune(
        "heuristic", trials=4, storage=f"sqlite:///{tmp_path / 'o.db'}", workers=2, deals=2
    )
    assert len(study.trials) == 4


def test_ppo_search_returns_real_config_fields(tmp_path: Path) -> None:
    study = tune(
        "ppo", trials=1, storage=f"sqlite:///{tmp_path / 'o.db'}", total_steps=512, eval_deals=1
    )
    overrides = best_ppo_overrides(study)
    PPOConfig(**overrides)  # every suggested name is a real field
    assert {"learning_rate", "card_head"} <= set(overrides)


def test_pruned_trials_are_tagged_in_mlflow() -> None:
    mlflow.set_experiment("prune-test")

    def objective(trial: optuna.Trial) -> float:
        raise optuna.TrialPruned

    study = optuna.create_study()
    study.optimize(_tracked(objective, "s"), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    runs = mlflow.search_runs(experiment_names=["prune-test"])
    assert runs.iloc[0]["tags.state"] == "pruned"


def test_cli_writes_best_settings_as_toml(tmp_path: Path) -> None:
    out = tmp_path / "best.toml"
    storage = f"sqlite:///{tmp_path / 'o.db'}"
    main(
        [
            "tune",
            "heuristic",
            "--trials",
            "2",
            "--deals",
            "2",
            "--storage",
            storage,
            "--out",
            str(out),
        ]
    )
    assert out.read_text().startswith("# Best of 2 Optuna trials")
    assert "# Held-out score on 10 unseen deals" in out.read_text()
    HeuristicParams(**tomllib.loads(out.read_text()))


def test_cli_promote(capsys: pytest.CaptureFixture[str]) -> None:
    from brisca.mlops.tracking import configure, log_policy
    from brisca.rl import ActorCritic

    configure("test")
    with mlflow.start_run():
        uri = log_policy(ActorCritic(hidden=8), {})
    main(["promote", uri])
    assert "v1 promoted=True" in capsys.readouterr().out
