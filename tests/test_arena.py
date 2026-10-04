import pytest

from brisca import DECK, Card
from brisca.agents import GreedyAgent, RandomAgent
from brisca.arena import MatchResult, play_game, play_match
from brisca.observation import Observation


class CheatingAgent:
    name = "cheater"

    def act(self, obs: Observation) -> Card:
        return next(card for card in DECK if card not in obs.hand)


def test_play_game_reaches_the_end() -> None:
    final = play_game([RandomAgent(seed=0), RandomAgent(seed=1)], seed=0)
    assert final.is_terminal
    assert sum(final.scores) == 120


def test_play_game_rejects_illegal_moves() -> None:
    with pytest.raises(ValueError, match="cheater"):
        play_game([CheatingAgent(), RandomAgent(seed=0)], seed=0)


def test_play_game_needs_two_agents() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        play_game([RandomAgent()], seed=0)


def test_match_plays_each_deal_from_both_seats() -> None:
    result = play_match(GreedyAgent(), RandomAgent(seed=0), deals=25, seed=0)
    assert result.games == 50
    assert result.points_for + result.points_against == 50 * 120


def test_duplicate_match_between_identical_deterministic_agents_is_even() -> None:
    result = play_match(GreedyAgent(), GreedyAgent(), deals=30, seed=0)
    assert result.wins == result.losses
    assert result.score == 0.5


def test_match_result_score_counts_draws_as_half() -> None:
    assert MatchResult(wins=3, draws=2, losses=5, points_for=0, points_against=0).score == 0.4
