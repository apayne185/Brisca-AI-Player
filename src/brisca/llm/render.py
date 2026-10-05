"""Describe game positions in plain language for a language model."""

from __future__ import annotations

from collections.abc import Iterable

from brisca.cards import Card, Suit
from brisca.observation import Observation

RANK_NAMES = {1: "Ace", 3: "Three", 10: "Jack", 11: "Knight", 12: "King"}

RULES = """\
Brisca is a two-player trick-taking game with a 40-card Spanish deck: suits
oros, copas, espadas and bastos, ranks 1-7 and 10-12. Each player holds three
cards. The suit of the face-up card is trump for the whole game; that card is
the last one drawn from the deck.

- Any card may be played: there is no obligation to follow suit.
- A trick is won by the highest trump played or, if there is none, by the
  highest card of the suit that was led.
- Card strength, high to low: Ace (1), Three (3), King (12), Knight (11),
  Jack (10), then 7, 6, 5, 4, 2.
- Points: Ace 11, Three 10, King 4, Knight 3, Jack 2, all others 0. The deck
  holds 120 points; more than 60 wins.
- The trick winner draws first and leads the next trick. When the deck is
  empty, the last three tricks are played from the cards in hand."""


def card_name(card: Card) -> str:
    """``Card(1, OROS)`` -> ``"Ace of oros (1O, 11 pts)"``."""
    rank = RANK_NAMES.get(card.rank, str(card.rank))
    return f"{rank} of {card.suit.name.lower()} ({card}, {card.points} pts)"


def _cards(cards: Iterable[Card]) -> str:
    named = [card_name(c) for c in cards]
    return ", ".join(named) if named else "none"


def describe(obs: Observation) -> str:
    """A complete, factual account of what the player to move can see."""
    me, them = obs.scores[obs.player], obs.scores[obs.opponent]
    played = sorted(obs.played_cards() - set(obs.current_trick), key=lambda c: c.ordinal)
    trumps_out = [c for c in played if c.suit == obs.trump]
    lines = [
        f"Trump suit: {obs.trump.name.lower()} (face-up card: {card_name(obs.trump_card)}).",
        f"Score: you {me}, opponent {them}. Points still in play: {120 - me - them}.",
        f"Cards left in the deck: {obs.stock_size}."
        + (" The deck is empty." if obs.stock_size == 0 else ""),
        f"Your hand: {_cards(obs.hand)}.",
    ]
    if obs.current_trick:
        lines.append(f"The opponent led: {_cards(obs.current_trick)}. You play second.")
    else:
        lines.append("You lead this trick.")
    known = obs.known_opponent_cards()
    if known:
        lines.append(f"The opponent is known to hold: {_cards(known)}.")
    lines += [
        f"Tricks completed: {len(obs.history)} of 20.",
        f"Trumps already played: {_cards(trumps_out)}.",
        "High cards (aces and threes) already played: "
        + _cards(c for c in played if c.rank in (1, 3))
        + ".",
    ]
    return "\n".join(lines)


def suit_names() -> dict[str, str]:
    return {s.symbol: s.name.lower() for s in Suit}
