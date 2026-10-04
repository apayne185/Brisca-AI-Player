"""Command-line PPO training.

    brisca-train --config configs/ppo.toml --total-steps 500000 --register

Settings come from ``PPOConfig`` defaults, then the optional TOML ``--config``,
then individual flags. With the ``mlops`` extra installed, every run is tracked
in MLflow (parameters, learning curves, git commit, and the trained policy);
``--register`` sends the result through the champion gate.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from brisca.rl.agent import save_checkpoint
from brisca.rl.model import ActorCritic
from brisca.rl.ppo import Metrics, PPOConfig, train

log = logging.getLogger("brisca.train")
EXPERIMENT = "ppo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    overrides: dict[str, Any] = tomllib.loads(known.config.read_text()) if known.config else {}
    unknown = set(overrides) - {f.name for f in dataclasses.fields(PPOConfig)}
    if unknown:
        raise SystemExit(f"unknown settings in {known.config}: {sorted(unknown)}")

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    defaults = dataclasses.replace(PPOConfig(), **overrides)
    for field in dataclasses.fields(PPOConfig):
        flag, default = f"--{field.name.replace('_', '-')}", getattr(defaults, field.name)
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("models/ppo.pt"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--mlflow", action=argparse.BooleanOptionalAction, default=True, help="track in MLflow"
    )
    parser.add_argument(
        "--register", action="store_true", help="register and promote if it beats the champion"
    )
    return parser.parse_args(argv)


def config_from(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PPOConfig)})


@contextlib.contextmanager
def _tracking(args: argparse.Namespace, config: PPOConfig) -> Iterator[Any]:
    """Yield the MLflow module inside an active run, or ``None`` if tracking is off."""
    if not args.mlflow:
        yield None
        return
    try:
        import mlflow

        from brisca.mlops.tracking import configure, git_tags
    except ImportError:
        log.warning("mlflow not installed; install the 'mlops' extra to track runs")
        yield None
        return
    configure(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name, tags=git_tags()):
        mlflow.log_params(dataclasses.asdict(config))
        yield mlflow


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    config = config_from(args)
    torch.set_num_threads(1)  # the network is tiny; threads only add overhead

    with _tracking(args, config) as mlflow:

        def report(step: int, metrics: Metrics) -> None:
            log.info(
                "step=%d %s", step, " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()))
            )
            if mlflow is not None:
                mlflow.log_metrics(metrics, step=step)

        model = train(config, on_metrics=report)
        metadata = {"config": dataclasses.asdict(config)}
        save_checkpoint(model, args.out, metadata)
        log.info("saved %s", args.out)
        model_uri = _log_model(model, metadata) if mlflow is not None else None

    if args.register:
        if model_uri is None:
            raise SystemExit("--register needs MLflow tracking (the 'mlops' extra)")
        from brisca.mlops.promotion import promote

        decision = promote(model_uri)
        log.info(
            "registry v%s promoted=%s: %s", decision.version, decision.promoted, decision.reason
        )


def _log_model(model: ActorCritic, metadata: dict[str, Any]) -> str:
    from brisca.mlops.tracking import log_policy

    model_uri = log_policy(model, metadata)
    log.info("logged %s", model_uri)
    return model_uri


if __name__ == "__main__":
    main()
