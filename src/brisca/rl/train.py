"""Command-line PPO training: ``brisca-train --total-steps 500000 --out models/ppo.pt``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch

from brisca.rl.agent import save_checkpoint
from brisca.rl.ppo import Metrics, PPOConfig, train

log = logging.getLogger("brisca.train")


def parse_args(argv: list[str] | None = None) -> tuple[PPOConfig, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = PPOConfig()
    for field in dataclasses.fields(PPOConfig):
        default = getattr(defaults, field.name)
        parser.add_argument(
            f"--{field.name.replace('_', '-')}", type=type(default), default=default
        )
    parser.add_argument("--out", type=Path, default=Path("models/ppo.pt"))
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    return PPOConfig(**args), out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config, out = parse_args(argv)
    torch.set_num_threads(1)  # the network is tiny; threads only add overhead

    def report(step: int, metrics: Metrics) -> None:
        log.info("step=%d %s", step, " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items())))

    model = train(config, on_metrics=report)
    save_checkpoint(model, out, {"config": dataclasses.asdict(config)})
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
