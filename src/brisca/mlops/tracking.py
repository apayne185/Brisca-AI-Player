"""MLflow conventions shared by training, tuning and promotion."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import torch
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel
from mlflow.types import Schema, TensorSpec

from brisca.encoding import NUM_ACTIONS, OBS_SIZE, action_mask, encode_observation
from brisca.engine import new_game
from brisca.observation import observe
from brisca.rl.agent import PolicyAgent, load_checkpoint, save_checkpoint
from brisca.rl.model import ActorCritic

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL = "brisca-ppo"
CHAMPION_ALIAS = "champion"
_CHECKPOINT = "checkpoint"


def configure(experiment: str) -> None:
    """Use ``$MLFLOW_TRACKING_URI`` if set, else a local SQLite store, and pick the experiment."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)


def git_tags() -> dict[str, str]:
    """Code version for lineage: the commit and whether the tree had local changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"git_sha": sha, "git_dirty": str(bool(dirty)).lower()}


class PolicyModel(PythonModel):
    """Serves a policy through MLflow's generic ``predict`` interface.

    Input is a dict of ``obs`` (n, OBS_SIZE) float32 and ``mask`` (n, 40) bool
    arrays from ``brisca.encoding``; output is the chosen card ordinal per row.
    """

    def load_context(self, context: Any) -> None:
        self.model, _ = load_checkpoint(context.artifacts[_CHECKPOINT])
        self.model.eval()

    # Left unannotated on purpose: MLflow inspects predict()'s type hints to build
    # a schema, and the explicit signature passed to log_model is more precise.
    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        obs = torch.as_tensor(np.asarray(model_input["obs"], dtype=np.float32))
        mask = torch.as_tensor(np.asarray(model_input["mask"], dtype=np.bool_))
        with torch.inference_mode():
            logits, _ = self.model(obs, mask)
        actions: npt.NDArray[np.int64] = logits.argmax(dim=-1).numpy()
        return actions


def _signature() -> ModelSignature:
    return ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.float32), (-1, OBS_SIZE), "obs"),
                TensorSpec(np.dtype(np.bool_), (-1, NUM_ACTIONS), "mask"),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype(np.int64), (-1,), "card")]),
    )


def _input_example() -> dict[str, npt.NDArray[Any]]:
    obs = observe(new_game(seed=0), player=0)
    return {"obs": encode_observation(obs)[None], "mask": action_mask(obs)[None]}


def log_policy(model: ActorCritic, metadata: dict[str, Any]) -> str:
    """Log ``model`` to the active run as a pyfunc model and return its URI."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.pt"
        save_checkpoint(model, path, metadata)
        info = mlflow.pyfunc.log_model(
            name="policy",
            python_model=PolicyModel(),
            artifacts={_CHECKPOINT: str(path)},
            signature=_signature(),
            input_example=_input_example(),
            pip_requirements=["brisca[rl]"],
        )
    return str(info.model_uri)


def load_policy_agent(model_uri: str, **kwargs: Any) -> PolicyAgent:
    """Load a logged or registered policy (``models:/brisca-ppo@champion``) as an Agent."""
    local = Path(mlflow.artifacts.download_artifacts(model_uri))
    checkpoint = next((local / "artifacts").glob("*.pt"))
    return PolicyAgent.from_checkpoint(checkpoint, **kwargs)
