"""Actor-critic network with action masking."""

from __future__ import annotations

import torch
from torch import nn

from brisca.cards import DECK
from brisca.encoding import CARD_PLANES, NUM_ACTIONS, OBS_SIZE

_CARD_FEATURES = len(CARD_PLANES) + 2  # per-card plane values plus points and strength
_CARD_DIM = 64  # kept small: the card network runs once per card, 40 times per state


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """A shared MLP trunk with a policy over the 40 cards and a value head.

    Two policy heads are available:

    - ``card_head=False``: a linear layer from the trunk to 40 logits, so each
      card's logit is learned separately.
    - ``card_head=True``: one small network, shared by all cards, scores each
      card from the game context plus that card's own features (its column of
      the card planes, its points and strength). What is learned about one card
      transfers to every other, which makes learning far more sample-efficient.

    Illegal cards (not in hand) get a logit of -inf-ish so they are never
    sampled and receive no gradient.
    """

    card_static: torch.Tensor

    def __init__(self, hidden: int = 256, card_head: bool = False) -> None:
        super().__init__()
        self.hidden = hidden
        self.card_head = card_head
        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(OBS_SIZE, hidden), 2**0.5),
            nn.ReLU(),
            _orthogonal(nn.Linear(hidden, hidden), 2**0.5),
            nn.ReLU(),
        )
        self.value = _orthogonal(nn.Linear(hidden, 1), 1.0)
        # Small final gains give a near-uniform initial policy, standard for PPO.
        if card_head:
            static = torch.tensor([[c.points / 11, c.strength / 9] for c in DECK])
            self.register_buffer("card_static", static)
            self.context = _orthogonal(nn.Linear(hidden, _CARD_DIM), 2**0.5)
            self.card_proj = _orthogonal(nn.Linear(_CARD_FEATURES, _CARD_DIM), 2**0.5)
            self.card_score = nn.Sequential(
                nn.ReLU(),
                _orthogonal(nn.Linear(_CARD_DIM, _CARD_DIM), 2**0.5),
                nn.ReLU(),
                _orthogonal(nn.Linear(_CARD_DIM, 1), 0.01),
            )
        else:
            self.policy = _orthogonal(nn.Linear(hidden, NUM_ACTIONS), 0.01)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns masked logits ``(batch, 40)`` and state values ``(batch,)``."""
        h = self.trunk(obs)
        if self.card_head:
            planes = obs[:, : len(CARD_PLANES) * NUM_ACTIONS]
            per_card = planes.reshape(-1, len(CARD_PLANES), NUM_ACTIONS).transpose(1, 2)
            static = self.card_static.expand(obs.shape[0], -1, -1)
            features = torch.cat([per_card, static], dim=-1)
            context = self.context(h).unsqueeze(1)
            logits = self.card_score(context + self.card_proj(features)).squeeze(-1)
        else:
            logits = self.policy(h)
        logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min)
        return logits, self.value(h).squeeze(-1)
