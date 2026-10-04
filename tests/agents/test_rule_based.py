import pytest

from brisca import Card
from brisca.agents import GreedyAgent, HeuristicAgent, HeuristicParams
from brisca.observation import Observation, observe
from tests.helpers import make_state

TRUMP = "7O"  # oros is trump in every position below


def obs_for(hand: str, lead: str = "", stock: str = "10B 7O") -> Observation:
    """Player 0's view, optionally after player 1 has led ``lead``."""
    to_play = 0
    state = make_state(hand, "4B 5B 6B", trump=TRUMP, stock=stock, trick=lead, to_play=to_play)
    return observe(state, 0)


@pytest.mark.parametrize("agent", [GreedyAgent(), HeuristicAgent()])
def test_leads_cheapest_non_trump(agent: GreedyAgent | HeuristicAgent) -> None:
    assert agent.act(obs_for("1O 4E 12C")) == Card.parse("4E")


@pytest.mark.parametrize("agent", [GreedyAgent(), HeuristicAgent()])
def test_trumps_a_led_ace(agent: GreedyAgent | HeuristicAgent) -> None:
    assert agent.act(obs_for("2O 5E 6B", lead="1C")) == Card.parse("2O")


@pytest.mark.parametrize("agent", [GreedyAgent(), HeuristicAgent()])
def test_does_not_waste_trump_on_worthless_trick(agent: GreedyAgent | HeuristicAgent) -> None:
    assert agent.act(obs_for("2O 5E 12B", lead="4C")) == Card.parse("5E")


def test_heuristic_banks_points_when_winning_in_suit() -> None:
    assert HeuristicAgent().act(obs_for("1C 7C 5E", lead="4C")) == Card.parse("1C")
    cautious = HeuristicAgent(HeuristicParams(secure_points=False))
    assert cautious.act(obs_for("1C 7C 5E", lead="4C")) == Card.parse("7C")


def test_heuristic_saves_trumps_for_valuable_tricks_until_the_endgame() -> None:
    hand, lead = "2O 5E 6B", "10C"  # the led jack is worth 2 points
    assert HeuristicAgent().act(obs_for(hand, lead)) == Card.parse("5E")
    assert HeuristicAgent().act(obs_for(hand, lead, stock="")) == Card.parse("2O")
