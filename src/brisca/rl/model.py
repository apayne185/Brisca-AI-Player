"""Actor-critic network with action masking."""

from __future__ import annotations

import torch
from torch import nn

from brisca.encoding import NUM_ACTIONS, OBS_SIZE


class ActorCritic(nn.Module):
    """A shared MLP trunk with a policy head over the 40 cards and a value head.

    Illegal cards (not in hand) get a logit of -inf-ish so they are never
    sampled and receive no gradient.
    """

    def __init__(self, hidden: int = 256, obs_size: int = OBS_SIZE) -> None:
        super().__init__()
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, NUM_ACTIONS)
        self.value = nn.Linear(hidden, 1)
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        # Near-uniform initial policy and small initial values, standard for PPO.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns masked logits ``(batch, 40)`` and state values ``(batch,)``."""
        h = self.trunk(obs)
        logits = self.policy(h).masked_fill(~mask, torch.finfo(torch.float32).min)
        return logits, self.value(h).squeeze(-1)
