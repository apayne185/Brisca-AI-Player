"""Wrap a trained network as an arena ``Agent`` and persist it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from brisca.cards import DECK, Card
from brisca.encoding import action_mask, encode_observation
from brisca.observation import Observation
from brisca.rl.model import ActorCritic


class PolicyAgent:
    """Plays the policy's most likely card, or samples from it if ``greedy=False``."""

    name = "ppo"

    def __init__(self, model: ActorCritic, greedy: bool = True, seed: int | None = None) -> None:
        self.model = model.eval()
        self.greedy = greedy
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    @torch.inference_mode()
    def act(self, obs: Observation) -> Card:
        x = torch.from_numpy(encode_observation(obs)).unsqueeze(0)
        mask = torch.from_numpy(action_mask(obs)).unsqueeze(0)
        logits, _ = self.model(x, mask)
        if self.greedy:
            action = int(logits.argmax(dim=-1))
        else:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1, generator=self._generator))
        return DECK[action]

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> PolicyAgent:
        return cls(load_checkpoint(path)[0], **kwargs)


def save_checkpoint(model: ActorCritic, path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": model.hidden, "state_dict": model.state_dict(), **metadata}, path)


def load_checkpoint(path: str | Path) -> tuple[ActorCritic, dict[str, Any]]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model = ActorCritic(hidden=payload.pop("hidden"))
    model.load_state_dict(payload.pop("state_dict"))
    return model, payload
