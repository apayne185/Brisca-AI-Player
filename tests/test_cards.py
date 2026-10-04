import pytest

from brisca import DECK, TOTAL_POINTS, Card, Suit


def test_deck_has_forty_unique_cards() -> None:
    assert len(DECK) == 40
    assert len(set(DECK)) == 40


def test_deck_points_total_120() -> None:
    assert sum(card.points for card in DECK) == TOTAL_POINTS


@pytest.mark.parametrize(
    ("rank", "points"), [(1, 11), (3, 10), (12, 4), (11, 3), (10, 2), (7, 0), (2, 0)]
)
def test_card_points(rank: int, points: int) -> None:
    assert Card(rank, Suit.OROS).points == points


def test_strength_order() -> None:
    order = [1, 3, 12, 11, 10, 7, 6, 5, 4, 2]
    strengths = [Card(rank, Suit.COPAS).strength for rank in order]
    assert strengths == sorted(strengths, reverse=True)


def test_ordinal_round_trips() -> None:
    assert [card.ordinal for card in DECK] == list(range(40))
    assert all(Card.from_ordinal(card.ordinal) == card for card in DECK)


@pytest.mark.parametrize("card", DECK)
def test_parse_round_trips(card: Card) -> None:
    assert Card.parse(str(card)) == card


@pytest.mark.parametrize("text", ["8O", "1X", "13C", "0E"])
def test_parse_rejects_invalid_cards(text: str) -> None:
    with pytest.raises(ValueError, match=r"invalid rank|unknown suit"):
        Card.parse(text)
