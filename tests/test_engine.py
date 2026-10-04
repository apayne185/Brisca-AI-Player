import dataclasses
import random

import pytest

from brisca import (
    DECK,
    Card,
    IllegalActionError,
    Trick,
    legal_actions,
    new_game,
    returns,
    step,
    winner,
)
from tests.helpers import cards, make_state, random_playout


class TestNewGame:
    def test_deals_three_cards_each_and_turns_up_trump(self) -> None:
        state = new_game(seed=0)
        assert [len(hand) for hand in state.hands] == [3, 3]
        assert len(state.stock) == 34
        assert state.stock[-1] == state.trump_card
        assert state.trump == state.trump_card.suit

    def test_uses_every_card_once(self) -> None:
        state = new_game(seed=1)
        dealt = [*state.hands[0], *state.hands[1], *state.stock]
        assert sorted(dealt) == sorted(DECK)

    def test_is_reproducible_from_seed(self) -> None:
        assert new_game(seed=42) == new_game(seed=42)
        assert new_game(seed=42) == new_game(seed=random.Random(42))
        assert new_game(seed=42) != new_game(seed=43)

    @pytest.mark.parametrize("first_player", [0, 1])
    def test_first_player_leads(self, first_player: int) -> None:
        state = new_game(seed=0, first_player=first_player)
        assert state.to_play == state.leader == first_player


class TestStep:
    def test_leading_passes_turn_without_scoring(self) -> None:
        state = make_state("1O 2C 3E", "4B 5O 6C", trump="7O", stock="10B 7O")
        after = step(state, Card.parse("2C"))
        assert after.current_trick == cards("2C")
        assert after.hands[0] == cards("1O 3E")
        assert after.to_play == 1
        assert after.leader == 0
        assert after.scores == (0, 0)

    def test_completed_trick_scores_and_winner_draws_first(self) -> None:
        # Player 0 leads the 3 of copas; player 1 trumps it with the 2 of oros.
        state = make_state("3C 4E", "2O 5B", trump="7O", stock="1B 12E 7O", trick="", to_play=0)
        state = step(step(state, Card.parse("3C")), Card.parse("2O"))

        assert state.history == (Trick(leader=0, cards=cards("3C 2O"), winner=1),)
        assert state.scores == (0, 10)
        assert state.to_play == state.leader == 1
        assert state.current_trick == ()
        assert state.hands[1] == cards("5B 1B")  # winner draws the top card
        assert state.hands[0] == cards("4E 12E")
        assert state.stock == cards("7O")

    def test_no_draw_once_stock_is_empty(self) -> None:
        state = make_state("1C 4E", "2C 5B", trump="7O", to_play=0)
        state = step(step(state, Card.parse("1C")), Card.parse("2C"))
        assert state.hands == (cards("4E"), cards("5B"))
        assert state.scores == (11, 0)

    def test_follower_who_wins_leads_next_trick(self) -> None:
        state = make_state("4E", "1E", trump="7O", to_play=0, scores=(0, 0))
        state = step(step(state, Card.parse("4E")), Card.parse("1E"))
        assert state.to_play == 1

    def test_rejects_card_not_in_hand(self) -> None:
        state = make_state("1O 2C 3E", "4B 5O 6C", trump="7O", stock="10B 7O")
        with pytest.raises(IllegalActionError):
            step(state, Card.parse("4B"))

    def test_is_pure(self) -> None:
        state = new_game(seed=3)
        snapshot = dataclasses.replace(state)
        step(state, legal_actions(state)[0])
        assert state == snapshot


class TestGameEnd:
    def test_full_game_lasts_twenty_tricks_and_awards_120_points(self) -> None:
        *_, final = random_playout(deal_seed=7, policy_rng=random.Random(7))
        assert final.is_terminal
        assert len(final.history) == 20
        assert sum(final.scores) == 120
        assert final.hands == ((), ())
        assert legal_actions(final) == ()

    def test_cannot_play_after_game_over(self) -> None:
        *_, final = random_playout(deal_seed=7, policy_rng=random.Random(7))
        with pytest.raises(IllegalActionError):
            step(final, DECK[0])

    @pytest.mark.parametrize(
        ("scores", "expected_winner", "expected_returns"),
        [((61, 59), 0, (1.0, -1.0)), ((0, 120), 1, (-1.0, 1.0)), ((60, 60), None, (0.0, 0.0))],
    )
    def test_winner_needs_more_than_sixty(
        self,
        scores: tuple[int, int],
        expected_winner: int | None,
        expected_returns: tuple[float, float],
    ) -> None:
        *_, final = random_playout(deal_seed=0, policy_rng=random.Random(0))
        final = dataclasses.replace(final, scores=scores)
        assert winner(final) == expected_winner
        assert returns(final) == expected_returns

    def test_winner_requires_finished_game(self) -> None:
        with pytest.raises(ValueError, match="not over"):
            winner(new_game(seed=0))


def test_str_is_readable() -> None:
    text = str(make_state("1O", "", trump="7O"))
    assert "trump=7O" in text
    assert "p0=[1O]" in text
    assert "p1=[-]" in text
