"""Two-player Brisca rules as pure functions over an immutable ``GameState``.

Each player holds three cards. The card under the dealt hands is turned face up:
its suit is trump and the card itself sits at the bottom of the stock, so it is
the last card drawn. Players need not follow suit. After each trick the winner
draws first, then the loser, while the stock lasts; the winner leads next.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from brisca.cards import DECK, POINTS, STRENGTH, TOTAL_POINTS, Card, Suit

NUM_PLAYERS = 2
HAND_SIZE = 3
NUM_TRICKS = len(DECK) // NUM_PLAYERS

Seed = int | random.Random | None


class IllegalActionError(ValueError):
    """Raised when a player tries to play a card they do not hold, or out of turn."""


@dataclass(frozen=True, slots=True)
class Trick:
    leader: int
    cards: tuple[Card, ...]
    winner: int

    @property
    def points(self) -> int:
        return sum(card.points for card in self.cards)


@dataclass(frozen=True, slots=True)
class GameState:
    hands: tuple[tuple[Card, ...], ...]
    stock: tuple[Card, ...]
    """Draw pile, top first. While non-empty, its last card is the face-up ``trump_card``."""
    trump_card: Card
    current_trick: tuple[Card, ...]
    to_play: int
    scores: tuple[int, ...]
    history: tuple[Trick, ...]

    @property
    def trump(self) -> Suit:
        return self.trump_card.suit

    @property
    def leader(self) -> int:
        """The player who led (or is about to lead) the current trick."""
        return (self.to_play - len(self.current_trick)) % NUM_PLAYERS

    @property
    def is_terminal(self) -> bool:
        return len(self.history) == NUM_TRICKS

    def __str__(self) -> str:
        def cards(cs: Sequence[Card]) -> str:
            return " ".join(map(str, cs)) or "-"

        return (
            f"trump={self.trump_card} stock={len(self.stock)} to_play={self.to_play} "
            f"scores={self.scores} trick=[{cards(self.current_trick)}] "
            + " ".join(f"p{i}=[{cards(hand)}]" for i, hand in enumerate(self.hands))
        )


def make_rng(seed: Seed) -> random.Random:
    """Accept a seed or an existing generator so callers control reproducibility."""
    return seed if isinstance(seed, random.Random) else random.Random(seed)


def new_game(seed: Seed = None, first_player: int = 0) -> GameState:
    """Shuffle, deal three cards each and turn up the trump card."""
    deck = list(DECK)
    make_rng(seed).shuffle(deck)
    hands = tuple(tuple(deck[i * HAND_SIZE : (i + 1) * HAND_SIZE]) for i in range(NUM_PLAYERS))
    dealt = NUM_PLAYERS * HAND_SIZE
    trump_card = deck[dealt]
    return GameState(
        hands=hands,
        stock=(*deck[dealt + 1 :], trump_card),
        trump_card=trump_card,
        current_trick=(),
        to_play=first_player,
        scores=(0,) * NUM_PLAYERS,
        history=(),
    )


def beats(challenger: Card, incumbent: Card, trump: Suit) -> bool:
    """Whether ``challenger`` takes the trick from the currently winning card."""
    if challenger.suit == incumbent.suit:
        return STRENGTH[challenger.rank] > STRENGTH[incumbent.rank]
    return challenger.suit == trump


def trick_winner(cards: Sequence[Card], trump: Suit) -> int:
    """Position (in play order) of the card that wins the trick."""
    best = 0
    for i in range(1, len(cards)):
        if beats(cards[i], cards[best], trump):
            best = i
    return best


def legal_actions(state: GameState) -> tuple[Card, ...]:
    """Any card in hand may be played: Brisca has no obligation to follow suit."""
    if state.is_terminal:
        return ()
    return state.hands[state.to_play]


def step(state: GameState, card: Card) -> GameState:
    """Play ``card`` for the player to move and return the resulting state."""
    # Hot path for search and self-play: avoid redundant copies and lookups.
    player = state.to_play
    hand = state.hands[player]
    try:
        i = hand.index(card)
    except ValueError:  # also covers terminal states, where every hand is empty
        raise IllegalActionError(f"player {player} cannot play {card} in state: {state}") from None

    hands = list(state.hands)
    hands[player] = hand[:i] + hand[i + 1 :]
    trick = (*state.current_trick, card)

    if len(trick) < NUM_PLAYERS:
        return GameState(
            hands=tuple(hands),
            stock=state.stock,
            trump_card=state.trump_card,
            current_trick=trick,
            to_play=(player + 1) % NUM_PLAYERS,
            scores=state.scores,
            history=state.history,
        )

    leader = (player + 1) % NUM_PLAYERS
    winner = (leader + trick_winner(trick, state.trump_card.suit)) % NUM_PLAYERS

    scores = list(state.scores)
    scores[winner] += sum([POINTS[c.rank] for c in trick])

    stock = state.stock
    if stock:
        for offset in range(NUM_PLAYERS):
            drawer = (winner + offset) % NUM_PLAYERS
            hands[drawer] = (*hands[drawer], stock[offset])
        stock = stock[NUM_PLAYERS:]

    return GameState(
        hands=tuple(hands),
        stock=stock,
        trump_card=state.trump_card,
        current_trick=(),
        to_play=winner,
        scores=tuple(scores),
        history=(*state.history, Trick(leader=leader, cards=trick, winner=winner)),
    )


def winner(state: GameState) -> int | None:
    """The player with more than half the points, or ``None`` for a 60-60 draw."""
    if not state.is_terminal:
        raise ValueError("game is not over")
    for player, score in enumerate(state.scores):
        if score * 2 > TOTAL_POINTS:
            return player
    return None


def returns(state: GameState) -> tuple[float, ...]:
    """Zero-sum terminal reward per player: +1 win, -1 loss, 0 draw."""
    won = winner(state)
    if won is None:
        return (0.0,) * NUM_PLAYERS
    return tuple(1.0 if player == won else -1.0 for player in range(NUM_PLAYERS))
