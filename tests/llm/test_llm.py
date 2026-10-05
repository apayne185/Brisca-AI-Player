"""LLM features against a fake client: no network calls, no cost."""

import json
import random
from collections.abc import Callable
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage  # noqa: E402

from brisca.agents import HeuristicAgent, RandomAgent  # noqa: E402
from brisca.arena import play_game  # noqa: E402
from brisca.cards import Card  # noqa: E402
from brisca.llm.agent import FALLBACK_BETA, LLMAgent  # noqa: E402
from brisca.llm.explain import hint  # noqa: E402
from brisca.llm.render import card_name, describe  # noqa: E402
from brisca.observation import Observation, observe  # noqa: E402
from tests.helpers import make_state, random_playout  # noqa: E402

Responder = Callable[[dict[str, Any]], BetaMessage]


def message(text: str, stop_reason: str = "end_turn") -> BetaMessage:
    return BetaMessage.model_construct(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5-5",
        content=[BetaTextBlock(type="text", text=text, citations=None)],
        stop_reason=stop_reason,
        usage=BetaUsage.model_construct(input_tokens=1000, output_tokens=200),
    )


class FakeClient:
    """Mimics ``client.beta.messages.create`` and records every request."""

    def __init__(self, respond: Responder) -> None:
        self.requests: list[dict[str, Any]] = []
        self.respond = respond
        self.beta = self
        self.messages = self

    def create(self, **kwargs: Any) -> BetaMessage:
        self.requests.append(kwargs)
        return self.respond(kwargs)


def pick_first_allowed(request: dict[str, Any]) -> BetaMessage:
    allowed = request["output_config"]["format"]["schema"]["properties"]["card"]["enum"]
    return message(json.dumps({"reason": "test", "card": allowed[0]}))


def raise_connection_error(request: dict[str, Any]) -> BetaMessage:
    raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example"))


def position(lead: str = "") -> Observation:
    state = make_state("1C 2O 5B", "4B 6E 7C", trump="7O", stock="10B 7O", trick=lead)
    return observe(state, 0)


def reachable_position(seed: int = 5, moves: int = 9) -> Observation:
    """A position from a real game, as ISMCTS needs a consistent observation."""
    states = list(random_playout(seed, random.Random(seed)))
    return observe(states[moves], states[moves].to_play)


def test_card_names_are_readable() -> None:
    assert card_name(Card.parse("1O")) == "Ace of oros (1O, 11 pts)"
    assert card_name(Card.parse("5B")) == "5 of bastos (5B, 0 pts)"


def test_description_covers_what_the_player_can_see() -> None:
    text = describe(position())
    assert "Trump suit: oros" in text
    assert "Ace of copas (1C, 11 pts)" in text
    assert "You lead this trick." in text
    assert "The opponent led: 3 of espadas" not in text
    assert "The opponent led: Three of espadas (3E, 10 pts)" in describe(position(lead="3E"))


def test_description_never_leaks_the_opponents_hand() -> None:
    for state in random_playout(4, random.Random(4)):
        if state.is_terminal or not state.stock:
            continue
        text = describe(observe(state, 0))
        for card in state.hands[1]:
            assert str(card) not in text.split("Your hand:")[1].split("\n")[0]


def test_agent_plays_the_card_claude_chooses() -> None:
    client = FakeClient(lambda r: message(json.dumps({"reason": "save trumps", "card": "5B"})))
    agent = LLMAgent(client=client, effort="medium")
    assert agent.act(position()) == Card.parse("5B")

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["schema"]["properties"]["card"]["enum"] == [
        "1C",
        "2O",
        "5B",
    ]
    assert request["fallbacks"] == "default"
    assert request["betas"] == [FALLBACK_BETA]
    assert agent.stats["input_tokens"] == 1000
    assert agent.cost_usd() == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    assert agent.fallback_rate == 0.0


def test_forced_moves_skip_the_api() -> None:
    client = FakeClient(pick_first_allowed)
    obs = observe(make_state("1C", "4B", trump="7O"), 0)
    assert LLMAgent(client=client).act(obs) == Card.parse("1C")
    assert client.requests == []


@pytest.mark.parametrize(
    ("respond", "reason"),
    [
        (lambda r: message("", stop_reason="refusal"), "fallback_refusal"),
        (lambda r: message('{"reason": "', stop_reason="max_tokens"), "fallback_max_tokens"),
        (raise_connection_error, "fallback_api_error"),
    ],
)
def test_failures_fall_back_and_are_counted(respond: Responder, reason: str) -> None:
    agent = LLMAgent(client=FakeClient(respond))
    obs = position()
    assert agent.act(obs) == HeuristicAgent().act(obs)
    assert agent.stats[reason] == 1
    assert agent.fallback_rate == 1.0


def test_llm_agent_completes_a_game_in_the_arena() -> None:
    agent = LLMAgent(client=FakeClient(pick_first_allowed))
    final = play_game([agent, RandomAgent(seed=0)], seed=3)
    assert sum(final.scores) == 120
    assert agent.stats["decisions"] > 10


def test_explanation_is_grounded_in_the_search() -> None:
    client = FakeClient(lambda r: message("Keep your trump; this trick is worthless."))
    obs = reachable_position()
    analysis = hint(obs, client=client, iterations=200)
    assert analysis.explanation == "Keep your trump; this trick is worthless."
    assert analysis.recommended == analysis.moves[0]
    assert {m.card for m in analysis.moves} == set(obs.hand)
    assert all(0.0 <= m.value <= 1.0 for m in analysis.moves)

    prompt = client.requests[0]["messages"][0]["content"]
    assert f"Explain why {card_name(analysis.recommended.card)}" in prompt
    assert "chance of winning the game" in prompt


@pytest.mark.parametrize(
    "respond", [lambda r: message("", stop_reason="refusal"), raise_connection_error]
)
def test_explanation_failures_return_none(respond: Responder) -> None:
    analysis = hint(reachable_position(), client=FakeClient(respond), iterations=50)
    assert analysis.explanation is None
    assert analysis.moves
