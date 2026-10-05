"""A Brisca agent that asks Claude which card to play.

Structured outputs constrain the answer to the cards in hand, so the model
cannot make an illegal move. Refusals, truncated answers and API errors fall
back to a deterministic agent and are counted, so a benchmark can report how
often the model actually decided.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Literal

import anthropic
from anthropic.types.beta import (
    BetaJSONOutputFormatParam,
    BetaOutputConfigParam,
    BetaTextBlock,
)

from brisca.agents import HeuristicAgent
from brisca.agents.base import Agent
from brisca.cards import Card
from brisca.llm.render import RULES, describe
from brisca.observation import Observation

log = logging.getLogger("brisca.llm")

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL = "claude-opus-5-5"
# Re-run a declined request on a suitable model server-side instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens (input, output), for cost reporting.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}

SYSTEM = f"""You are an expert Brisca player. Choose the card that maximises your
chance of winning the game (not just the current trick).

{RULES}

Think about which cards are still unseen, whether the current trick is worth
winning, and whether to keep trumps and high cards for later. Answer with a
one- or two-sentence reason and the card code from your hand."""


class LLMAgent:
    name = "llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: Effort = "low",
        client: Any = None,
        fallback: Agent | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.client = client if client is not None else anthropic.Anthropic()
        self.fallback = fallback or HeuristicAgent()
        self.stats: Counter[str] = Counter()

    def act(self, obs: Observation) -> Card:
        if len(obs.hand) == 1:
            return obs.hand[0]
        self.stats["decisions"] += 1
        output_config: BetaOutputConfigParam = {"effort": self.effort, "format": _schema(obs.hand)}
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                system=SYSTEM,
                messages=[{"role": "user", "content": describe(obs)}],
                output_config=output_config,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            log.warning("Claude request failed, using fallback: %s", exc)
            return self._fall_back(obs, "api_error")

        self.stats["input_tokens"] += response.usage.input_tokens
        self.stats["output_tokens"] += response.usage.output_tokens
        if response.stop_reason in ("refusal", "max_tokens"):
            return self._fall_back(obs, str(response.stop_reason))

        text = next(b.text for b in response.content if isinstance(b, BetaTextBlock))
        card = Card.parse(json.loads(text)["card"])
        if card not in obs.hand:  # guaranteed by the schema; kept as a guard
            return self._fall_back(obs, "illegal")
        return card

    def _fall_back(self, obs: Observation, reason: str) -> Card:
        self.stats[f"fallback_{reason}"] += 1
        return self.fallback.act(obs)

    def cost_usd(self) -> float:
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        return (
            self.stats["input_tokens"] * price_in + self.stats["output_tokens"] * price_out
        ) / 1e6

    @property
    def fallback_rate(self) -> float:
        fallbacks = sum(v for k, v in self.stats.items() if k.startswith("fallback_"))
        return fallbacks / max(1, self.stats["decisions"])


def _schema(hand: tuple[Card, ...]) -> BetaJSONOutputFormatParam:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "card": {"type": "string", "enum": [str(c) for c in hand]},
            },
            "required": ["reason", "card"],
            "additionalProperties": False,
        },
    }
