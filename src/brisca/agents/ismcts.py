"""Single-Observer Information Set Monte Carlo Tree Search (SO-ISMCTS).

Plain MCTS assumes the full state is known. ISMCTS (Cowling, Powley & Whitehouse,
2012) instead searches a tree over the *player's information set*: each
iteration samples a hidden state consistent with the observation
(``determinize``) and descends only through moves legal in that sample. Because
a node's children are not all available in every sample, UCB uses each child's
*availability* count instead of the parent's visit count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from brisca.agents.base import Agent
from brisca.cards import Card
from brisca.engine import GameState, Seed, make_rng, returns, step
from brisca.observation import Observation, determinize, observe


@dataclass(eq=False, slots=True)
class _Node:
    parent: _Node | None = None
    move: Card | None = None
    player: int = -1
    """The player who made ``move``; rewards are stored from their point of view."""
    children: dict[Card, _Node] = field(default_factory=dict)
    visits: int = 0
    reward: float = 0.0
    availability: int = 0

    def ucb(self, exploration: float) -> float:
        return self.reward / self.visits + exploration * math.sqrt(
            math.log(self.availability) / self.visits
        )


class ISMCTSAgent:
    """Information-set MCTS with random (or agent-guided) rollouts.

    Rewards are win/draw/loss mapped to 1/0.5/0 for the player who moved into
    each node, so the same tree serves both players.
    """

    name = "ismcts"

    def __init__(
        self,
        iterations: int = 1000,
        exploration: float = 0.7,
        rollout_agent: Agent | None = None,
        seed: Seed = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_agent = rollout_agent
        self._rng = make_rng(seed)

    def act(self, obs: Observation) -> Card:
        if len(obs.hand) == 1:
            return obs.hand[0]

        root = _Node()
        for _ in range(self.iterations):
            state = determinize(obs, self._rng)
            node, state = self._select_and_expand(root, state)
            final = self._rollout(state)
            self._backpropagate(node, final)

        best = max(root.children.values(), key=lambda n: n.visits)
        assert best.move is not None
        return best.move

    def _select_and_expand(self, node: _Node, state: GameState) -> tuple[_Node, GameState]:
        while not state.is_terminal:
            legal = state.hands[state.to_play]
            available = [node.children[c] for c in legal if c in node.children]
            for child in available:
                child.availability += 1

            untried = [c for c in legal if c not in node.children]
            if untried:
                move = self._rng.choice(untried)
                child = _Node(parent=node, move=move, player=state.to_play, availability=1)
                node.children[move] = child
                return child, step(state, move)

            node = max(available, key=lambda n: n.ucb(self.exploration))
            assert node.move is not None
            state = step(state, node.move)
        return node, state

    def _rollout(self, state: GameState) -> GameState:
        rng, agent = self._rng, self.rollout_agent
        while not state.is_terminal:
            if agent is None:
                move = rng.choice(state.hands[state.to_play])
            else:
                move = agent.act(observe(state, state.to_play))
            state = step(state, move)
        return state

    @staticmethod
    def _backpropagate(node: _Node | None, final: GameState) -> None:
        values = [(r + 1) / 2 for r in returns(final)]
        while node is not None:
            node.visits += 1
            if node.player >= 0:
                node.reward += values[node.player]
            node = node.parent
