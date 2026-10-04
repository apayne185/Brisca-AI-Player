"""Champion/challenger gate for the model registry.

A candidate replaces the current champion only if it wins significantly more
duplicate deals than it loses, judged by an exact one-sided sign test. Deals
where both games split evenly carry no information and are dropped, as usual
for a sign test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from brisca.arena import play_match
from brisca.mlops.tracking import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL,
    git_tags,
    load_policy_agent,
)


def sign_test(deal_scores: Sequence[float]) -> float:
    """One-sided p-value that the candidate wins more deals than it loses."""
    wins = sum(s > 0.5 for s in deal_scores)
    losses = sum(s < 0.5 for s in deal_scores)
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n)


@dataclass(frozen=True)
class Decision:
    promoted: bool
    version: str
    score: float | None
    p_value: float | None
    reason: str


def promote(
    candidate_uri: str,
    deals: int = 300,
    alpha: float = 0.05,
    seed: int = 2024,
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
        decision = Decision(True, version, None, None, "no champion yet")
    else:
        candidate = load_policy_agent(candidate_uri)
        champion = load_policy_agent(f"models:/{model_name}@{CHAMPION_ALIAS}")
        result = play_match(candidate, champion, deals=deals, seed=seed)
        p_value = sign_test(result.deal_scores)
        promoted = p_value < alpha
        verdict = "beats" if promoted else "does not significantly beat"
        reason = (
            f"{verdict} v{incumbent.version}: score {result.score:.3f}, "
            f"p={p_value:.4f} (alpha={alpha}, {deals} deals)"
        )
        decision = Decision(promoted, version, result.score, p_value, reason)

    experiment = mlflow.set_experiment("promotion")
    with mlflow.start_run(
        run_name=f"promote-v{version}", experiment_id=experiment.experiment_id, tags=git_tags()
    ):
        mlflow.log_params({"candidate_uri": candidate_uri, "deals": deals, "alpha": alpha})
        mlflow.log_metrics(
            {k: float(v) for k, v in asdict(decision).items() if isinstance(v, (int, float))}
        )
        mlflow.set_tag("reason", decision.reason)

    for key, value in asdict(decision).items():
        client.set_model_version_tag(model_name, version, key, str(value))
    if decision.promoted:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    return decision
