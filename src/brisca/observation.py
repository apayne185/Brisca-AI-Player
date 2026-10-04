"""What a single player can see, and sampling hidden states consistent with it.

Brisca is a game of imperfect information: the opponent's hand and the order of
the stock are hidden. Agents act on an ``Observation`` (an information set) and
search algorithms such as ISMCTS draw concrete ``GameState`` samples from it with
``determinize``.
"""

from __future__ import annotations

from dataclasses import dataclass

from brisca.cards import DECK, Card, Suit
from brisca.engine import NUM_PLAYERS, GameState, Seed, Trick, make_rng


@dataclass(frozen=True, slots=True)
class Observation:
    player: int
    hand: tuple[Card, ...]
    trump_card: Card
    current_trick: tuple[Card, ...]
    to_play: int
    scores: tuple[int, ...]
    history: tuple[Trick, ...]
    stock_size: int
    opponent_hand_size: int

    @property
    def trump(self) -> Suit:
        return self.trump_card.suit

    @property
    def opponent(self) -> int:
        return (self.player + 1) % NUM_PLAYERS

    def played_cards(self) -> frozenset[Card]:
        """Cards in completed tricks and on the table."""
        played = {card for trick in self.history for card in trick.cards}
        return frozenset(played.union(self.current_trick))

    def known_opponent_cards(self) -> tuple[Card, ...]:
        """Opponent cards whose location is public.

        The face-up trump card is the last card of the stock. Once the stock is
        empty, if it is neither in our hand nor played, the opponent drew it.
        """
        if (
            self.stock_size == 0
            and self.trump_card not in self.hand
            and self.trump_card not in self.played_cards()
        ):
            return (self.trump_card,)
        return ()

    def unseen_cards(self) -> tuple[Card, ...]:
        """Cards that could be in the opponent's hand or the hidden part of the stock."""
        seen = self.played_cards() | frozenset(self.hand) | {self.trump_card}
        return tuple(card for card in DECK if card not in seen)


def observe(state: GameState, player: int) -> Observation:
    opponent = (player + 1) % NUM_PLAYERS
    return Observation(
        player=player,
        hand=state.hands[player],
        trump_card=state.trump_card,
        current_trick=state.current_trick,
        to_play=state.to_play,
        scores=state.scores,
        history=state.history,
        stock_size=len(state.stock),
        opponent_hand_size=len(state.hands[opponent]),
    )


def determinize(obs: Observation, seed: Seed = None) -> GameState:
    """Sample a full game state uniformly from those consistent with ``obs``.

    No card-play constraint reveals voids in Brisca (players need not follow
    suit), so every assignment of unseen cards is equally likely.
    """
    unseen = list(obs.unseen_cards())
    known = obs.known_opponent_cards()
    n_hidden = obs.opponent_hand_size - len(known)
    if len(unseen) != n_hidden + max(obs.stock_size - 1, 0):
        raise ValueError("observation is internally inconsistent")

    make_rng(seed).shuffle(unseen)
    opponent_hand = (*known, *unseen[:n_hidden])
    stock = (*unseen[n_hidden:], obs.trump_card) if obs.stock_size else ()

    hands = [opponent_hand] * NUM_PLAYERS
    hands[obs.player] = obs.hand
    return GameState(
        hands=tuple(hands),
        stock=stock,
        trump_card=obs.trump_card,
        current_trick=obs.current_trick,
        to_play=obs.to_play,
        scores=obs.scores,
        history=obs.history,
    )
