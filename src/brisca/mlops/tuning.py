"""Hyperparameter search with Optuna, with every trial tracked in MLflow.

Studies persist to SQLite, so a search can be resumed or spread across
processes (``workers``) that share the same storage.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import mlflow
import optuna
import torch

from brisca.agents import GreedyAgent, HeuristicAgent, HeuristicParams
from brisca.agents.base import Agent
from brisca.arena import play_match
from brisca.mlops.tracking import configure, git_tags
from brisca.rl.ppo import Metrics, PPOConfig, evaluate, train

log = logging.getLogger("brisca.tune")
Objective = Callable[[optuna.Trial], float]


HOLDOUT_SEED = 987_654
"""Deals for held-out validation; never used by any objective."""


def heuristic_score(params: HeuristicParams, deals: int, seed: int) -> float:
    """Mean duplicate score against greedy and the default heuristic."""
    opponents: list[Agent] = [GreedyAgent(), HeuristicAgent()]
    agent = HeuristicAgent(params)
    scores = [play_match(agent, o, deals=deals, seed=seed).score for o in opponents]
    return sum(scores) / len(scores)


def heuristic_objective(deals: int, seed: int = 0) -> Objective:
    def objective(trial: optuna.Trial) -> float:
        params = HeuristicParams(
            capture_threshold=trial.suggest_int("capture_threshold", 0, 11),
            trump_cost=trial.suggest_float("trump_cost", 0.0, 12.0),
            secure_points=trial.suggest_categorical("secure_points", [True, False]),
            endgame_trumps=trial.suggest_categorical("endgame_trumps", [True, False]),
        )
        return heuristic_score(params, deals, seed)

    return objective


def heuristic_holdout(study: optuna.Study, deals: int) -> float:
    """Re-score the best parameters on fresh deals.

    Every trial is scored on the same deals, so the best trial's value is
    biased upwards by selection (the winner's curse). This is the number to
    trust.
    """
    return heuristic_score(best_heuristic_params(study), deals, HOLDOUT_SEED)


def ppo_objective(total_steps: int, eval_deals: int, seed: int = 0) -> Objective:
    """Greedy-policy score against the heuristic after a short training budget.

    Intermediate evaluations feed Optuna's pruner, so hopeless configurations
    stop early and the budget goes to promising ones.
    """

    def objective(trial: optuna.Trial) -> float:
        config = PPOConfig(
            total_steps=total_steps,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            entropy_coef=trial.suggest_float("entropy_coef", 1e-4, 5e-2, log=True),
            shaping=trial.suggest_float("shaping", 0.0, 1.0),
            hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
            card_head=trial.suggest_categorical("card_head", [False, True]),
            self_play_prob=trial.suggest_float("self_play_prob", 0.0, 0.8),
            rollout_len=trial.suggest_categorical("rollout_len", [32, 64, 128]),
            epochs=trial.suggest_int("epochs", 2, 8),
            clip=trial.suggest_float("clip", 0.1, 0.3),
            eval_every=max(1, total_steps // (32 * 64 * 4)),
            eval_deals=eval_deals,
            seed=seed + trial.number,
        )

        def report(step: int, metrics: Metrics) -> None:
            mlflow.log_metrics(metrics, step=step)
            if "eval_vs_heuristic" in metrics:
                trial.report(metrics["eval_vs_heuristic"], step)
                if trial.should_prune():
                    raise optuna.TrialPruned

        torch.set_num_threads(1)
        model = train(config, on_metrics=report)
        return evaluate(model, HeuristicAgent(), deals=eval_deals * 2)

    return objective


OBJECTIVES: dict[str, Callable[..., Objective]] = {
    "heuristic": heuristic_objective,
    "ppo": ppo_objective,
}


def _tracked(objective: Objective, study_name: str) -> Objective:
    """Run each trial inside its own MLflow run, tagged with the study."""

    def wrapped(trial: optuna.Trial) -> float:
        tags = {**git_tags(), "study": study_name, "trial": str(trial.number)}
        with mlflow.start_run(run_name=f"{study_name}-{trial.number}", tags=tags):
            try:
                value = objective(trial)
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                raise
            mlflow.log_params(trial.params)
            mlflow.log_metric("objective", value)
            return value

    return wrapped


def _optimize(
    target: str, kwargs: dict[str, Any], study_name: str, storage: str, trials: int
) -> None:
    configure(f"tune-{target}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    objective = _tracked(OBJECTIVES[target](**kwargs), study_name)
    study.optimize(objective, n_trials=trials)


def tune(
    target: str,
    trials: int,
    storage: str,
    study_name: str | None = None,
    workers: int = 1,
    seed: int = 0,
    **kwargs: Any,
) -> optuna.Study:
    """Create (or resume) a study and run ``trials`` trials across ``workers`` processes."""
    study_name = study_name or f"{target}-search"
    # Create the MLflow store and experiment once, here, so worker processes
    # don't race to initialise the same SQLite database.
    configure(f"tune-{target}")
    optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    per_worker = [trials // workers + (i < trials % workers) for i in range(workers)]
    if workers == 1:
        _optimize(target, kwargs, study_name, storage, trials)
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [
                pool.submit(_optimize, target, kwargs, study_name, storage, n)
                for n in per_worker
                if n
            ]
            for future in futures:
                future.result()

    study = optuna.load_study(study_name=study_name, storage=storage)
    log.info("best value %.4f with %s", study.best_value, study.best_params)
    return study


def best_heuristic_params(study: optuna.Study) -> HeuristicParams:
    return HeuristicParams(**study.best_params)


def best_ppo_overrides(study: optuna.Study) -> dict[str, Any]:
    """Best PPO settings, restricted to real ``PPOConfig`` fields."""
    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    return {k: v for k, v in study.best_params.items() if k in fields}
