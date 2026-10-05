"""Export a trained policy to ONNX so it can be served without PyTorch.

brisca-export-onnx models/ppo-v2.pt models/ppo-v2.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from brisca.encoding import NUM_ACTIONS, OBS_SIZE
from brisca.rl.agent import load_checkpoint


def export_onnx(checkpoint: str | Path, out: str | Path, check: bool = True) -> Path:
    model, _ = load_checkpoint(checkpoint)
    model.eval()
    # Batch of 2: torch.export specialises size-1 dimensions, which would pin the batch.
    obs = torch.zeros(2, OBS_SIZE)
    mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch")
    torch.onnx.export(
        model,
        (obs, mask),
        out,
        input_names=["obs", "mask"],
        output_names=["logits", "value"],
        dynamic_shapes={"obs": {0: batch}, "mask": {0: batch}},
        external_data=False,
    )
    if check:
        _check_parity(model, out)
    return out


def _check_parity(model: torch.nn.Module, path: Path, samples: int = 256) -> None:
    """The ONNX graph must reproduce the PyTorch logits on random inputs."""
    rng = np.random.default_rng(0)
    obs = rng.random((samples, OBS_SIZE), dtype=np.float32)
    mask = rng.random((samples, NUM_ACTIONS)) < 0.1
    mask[np.arange(samples), rng.integers(0, NUM_ACTIONS, samples)] = True
    with torch.inference_mode():
        expected, _ = model(torch.from_numpy(obs), torch.from_numpy(mask))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"obs": obs, "mask": mask})[0]
    legal = mask
    if not np.allclose(logits[legal], expected.numpy()[legal], atol=1e-4):
        raise AssertionError("ONNX logits differ from PyTorch")
    if not (logits.argmax(1) == expected.numpy().argmax(1)).all():
        raise AssertionError("ONNX policy picks different cards from PyTorch")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    print(f"exported {export_onnx(args.checkpoint, args.out)} (parity check passed)")


if __name__ == "__main__":
    main()
