"""Plain-language explanations of a move, grounded in ISMCTS search statistics.

The search does the playing; the language model only explains. Feeding it the
search's numbers keeps the explanation honest about what the engine actually
found, instead of letting the model invent its own analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types.beta import BetaTextBlock

from brisca.agents.ismcts import ISMCTSAgent, MoveStats
from brisca.llm.agent import DEFAULT_MODEL, FALLBACK_BETA
from brisca.llm.render import RULES, card_name, describe
from brisca.observation import Observation

SYSTEM = f"""You explain Brisca moves to a casual player in two or three short
sentences. A search engine has analysed the position; base your explanation on
its numbers and on the rules, and do not contradict its recommendation. Avoid
jargon such as "MCTS" or "visits"; talk about chances of winning.

{RULES}"""


@dataclass(frozen=True)
class Analysis:
    recommended: MoveStats
    moves: list[MoveStats]
    explanation: str | None


def analyse(obs: Observation, iterations: int = 2000, seed: int = 0) -> list[MoveStats]:
    return ISMCTSAgent(iterations=iterations, seed=seed).search(obs)


def explain(
    obs: Observation,
    moves: list[MoveStats],
    client: Any = None,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Ask Claude to explain the top move; ``None`` if it declines or the call fails."""
    client = client if client is not None else anthropic.Anthropic()
    table = "\n".join(
        f"- {card_name(m.card)}: estimated chance of winning the game {m.value:.0%}" for m in moves
    )
    prompt = (
        f"{describe(obs)}\n\nSearch results, best first:\n{table}\n\n"
        f"Explain why {card_name(moves[0].card)} is the recommended play."
    )
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.APIError:
        return None
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if isinstance(b, BetaTextBlock)).strip()
    return text or None


def hint(obs: Observation, client: Any = None, iterations: int = 2000) -> Analysis:
    moves = analyse(obs, iterations=iterations)
    return Analysis(moves[0], moves, explain(obs, moves, client=client))
