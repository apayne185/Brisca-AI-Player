"""Serve an exported policy with ONNX Runtime, no PyTorch required.

Requires the ``onnx`` extra. Produce the model with ``brisca-export-onnx``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from brisca.cards import DECK, Card
from brisca.encoding import action_mask, encode_observation
from brisca.observation import Observation


class OnnxPolicyAgent:
    """Plays the policy's most likely card, like ``PolicyAgent(greedy=True)``."""

    name = "ppo"

    def __init__(self, path: str | Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # tiny network: threads only add overhead
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def act(self, obs: Observation) -> Card:
        feeds = {
            "obs": encode_observation(obs)[None],
            "mask": action_mask(obs)[None],
        }
        logits = self.session.run(["logits"], feeds)[0]
        return DECK[int(np.argmax(logits[0]))]
