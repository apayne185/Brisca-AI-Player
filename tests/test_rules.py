import pytest

from brisca import Card, Suit, trick_winner
from tests.helpers import cards

TRUMP = Suit.OROS


@pytest.mark.parametrize(
    ("played", "expected", "reason"),
    [
        ("3C 1C", 1, "ace beats three in the led suit"),
        ("1C 3C", 0, "three does not beat ace"),
        ("7C 10C", 1, "jack beats seven"),
        ("2C 4C", 1, "four beats two"),
        ("12C 11C", 0, "king beats knight"),
        ("7C 1E", 0, "off-suit ace cannot win"),
        ("1C 2O", 1, "lowest trump beats highest non-trump"),
        ("2O 1C", 0, "led trump beats off-suit ace"),
        ("3O 1O", 1, "higher trump wins"),
        ("1O 3O", 0, "lower trump loses"),
    ],
)
def test_trick_winner(played: str, expected: int, reason: str) -> None:
    assert trick_winner(cards(played), TRUMP) == expected, reason


def test_single_card_trick_is_won_by_leader() -> None:
    assert trick_winner([Card(2, Suit.BASTOS)], TRUMP) == 0
