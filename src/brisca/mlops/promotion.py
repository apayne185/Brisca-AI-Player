"""Champion/challenger gate for the model registry.

Candidate and champion each play the same duplicate deals against a fixed
reference panel of agents. The candidate is promoted only if it does better
than the champion on significantly more of those deals than it does worse,
judged by an exact one-sided paired sign test; ties carry no information and
are dropped.

Why a panel rather than a head-to-head match: head-to-head results are not
transitive. ppo-v2 was rated clearly above ppo-v1 across the tournament field
yet drew with it head to head, so a head-to-head gate wrongly kept v1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from brisca.agents import make_agent
from brisca.agents.base import Agent
from brisca.arena import play_match
from brisca.mlops.tracking import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL,
    git_tags,
    load_policy_agent,
)

REFERENCE_PANEL = ("greedy", "heuristic")


def sign_test(values: Sequence[float], centre: float = 0.5) -> float:
    """One-sided p-value that values above ``centre`` outnumber those below it."""
    wins = sum(v > centre for v in values)
    losses = sum(v < centre for v in values)
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n)


@dataclass(frozen=True)
class Decision:
    promoted: bool
    version: str
    candidate_score: float | None
    champion_score: float | None
    p_value: float | None
    reason: str


def panel_scores(agent: Agent, panel: Sequence[str], deals: int, seed: int) -> list[float]:
    """Per-deal duplicate scores of ``agent`` against each panel opponent, in a fixed order."""
    scores: list[float] = []
    for name in panel:
        scores.extend(play_match(agent, make_agent(name), deals=deals, seed=seed).deal_scores)
    return scores


def promote(
    candidate_uri: str,
    deals: int = 300,
    alpha: float = 0.05,
    seed: int = 2024,
    panel: Sequence[str] = REFERENCE_PANEL,
    model_name: str = REGISTERED_MODEL,
) -> Decision:
    """Register ``candidate_uri`` and make it champion if it beats the incumbent.

    Every candidate becomes a registry version, tagged with the evaluation, so
    rejected models stay auditable. The first model registered becomes
    champion by default.
    """
    client = MlflowClient()
    try:
        incumbent = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        incumbent = None

    version = str(mlflow.register_model(candidate_uri, model_name).version)
    if incumbent is None:
        decision = Decision(True, version, None, None, None, "no champion yet")
    else:
        candidate = panel_scores(load_policy_agent(candidate_uri), panel, deals, seed)
        champion = panel_scores(
            load_policy_agent(f"models:/{model_name}@{CHAMPION_ALIAS}"), panel, deals, seed
        )
        differences = [c - h for c, h in zip(candidate, champion, strict=True)]
        p_value = sign_test(differences, centre=0.0)
        promoted = p_value < alpha
        cand_mean, champ_mean = sum(candidate) / len(candidate), sum(champion) / len(champion)
        verdict = "beats" if promoted else "does not significantly beat"
        reason = (
            f"{verdict} v{incumbent.version} against {'/'.join(panel)}: "
            f"{cand_mean:.3f} vs {champ_mean:.3f}, p={p_value:.4f} "
            f"(alpha={alpha}, {deals} deals per opponent)"
        )
        decision = Decision(promoted, version, cand_mean, champ_mean, p_value, reason)

    experiment = mlflow.set_experiment("promotion")
    with mlflow.start_run(
        run_name=f"promote-v{version}", experiment_id=experiment.experiment_id, tags=git_tags()
    ):
        mlflow.log_params(
            {"candidate_uri": candidate_uri, "deals": deals, "alpha": alpha, "panel": panel}
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in asdict(decision).items() if isinstance(v, (int, float))}
        )
        mlflow.set_tag("reason", decision.reason)

    for key, value in asdict(decision).items():
        client.set_model_version_tag(model_name, version, key, str(value))
    if decision.promoted:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    return decision
