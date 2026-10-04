"""Small card-evaluation helpers shared by rule-based agents."""

from __future__ import annotations

from brisca.cards import Card, Suit
from brisca.engine import beats


def card_cost(card: Card, trump: Suit, trump_cost: float = 5.0) -> float:
    """How much it hurts to give a card away: its points, trump status and strength."""
    return card.points + (trump_cost if card.suit == trump else 0.0) + card.strength / 10


def cheapest(cards: tuple[Card, ...] | list[Card], trump: Suit, trump_cost: float = 5.0) -> Card:
    return min(cards, key=lambda c: card_cost(c, trump, trump_cost))


def winners_against(hand: tuple[Card, ...], lead: Card, trump: Suit) -> list[Card]:
    """Cards in ``hand`` that would take the trick from ``lead``."""
    return [card for card in hand if beats(card, lead, trump)]
