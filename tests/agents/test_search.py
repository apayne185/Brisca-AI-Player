import math
import random
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brisca import GameState, returns, step
from brisca.agents import DeterminizedAlphaBetaAgent, GreedyAgent, ISMCTSAgent
from brisca.agents.alphabeta import alphabeta, evaluate
from brisca.observation import observe
from tests.helpers import random_playout


def minimax(state: GameState, depth: int, player: int) -> float:
    """Reference search without pruning."""
    if depth == 0 or state.is_terminal:
        return evaluate(state, player)
    values = [minimax(step(state, c), depth - 1, player) for c in state.hands[state.to_play]]
    return max(values) if state.to_play == player else min(values)


def solve_outcome(state: GameState, player: int) -> float:
    """Exact game-theoretic outcome (+1/0/-1) of a perfect-information position."""
    if state.is_terminal:
        return returns(state)[player]
    values = [solve_outcome(step(state, c), player) for c in state.hands[state.to_play]]
    return max(values) if state.to_play == player else min(values)


def endgames(seed: int) -> list[GameState]:
    """Positions after the stock runs out, where Brisca is perfect information."""
    return [
        s
        for s in random_playout(seed, random.Random(seed))
        if not s.stock and not s.is_terminal and len(s.hands[s.to_play]) > 1
    ]


@settings(max_examples=30, deadline=None)
@given(
    deal_seed=st.integers(0, 2**32 - 1),
    policy=st.randoms(use_true_random=False),
    depth=st.integers(1, 3),
)
def test_pruning_does_not_change_the_minimax_value(
    deal_seed: int, policy: random.Random, depth: int
) -> None:
    for state in list(random_playout(deal_seed, policy))[:-1:7]:
        player = state.to_play
        assert alphabeta(state, depth, -math.inf, math.inf, player) == minimax(state, depth, player)


@pytest.mark.parametrize("seed", range(10))
def test_alphabeta_plays_endgames_perfectly(seed: int) -> None:
    agent = DeterminizedAlphaBetaAgent(seed=seed)
    for state in endgames(seed):
        player = state.to_play
        best = max(minimax(step(state, c), 6, player) for c in state.hands[player])
        chosen = agent.act(observe(state, player))
        assert minimax(step(state, chosen), 6, player) == best


@pytest.mark.parametrize("seed", range(10))
def test_ismcts_never_throws_away_a_won_endgame(seed: int) -> None:
    agent = ISMCTSAgent(iterations=300, seed=seed)
    for state in endgames(seed):
        player = state.to_play
        best = max(solve_outcome(step(state, c), player) for c in state.hands[player])
        chosen = agent.act(observe(state, player))
        assert solve_outcome(step(state, chosen), player) == best


def test_ismcts_supports_agent_guided_rollouts() -> None:
    state = next(iter(random_playout(0, random.Random(0))))
    agent = ISMCTSAgent(iterations=10, rollout_agent=GreedyAgent(), seed=0)
    assert agent.act(observe(state, 0)) in state.hands[0]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ISMCTSAgent(iterations=0),
        lambda: DeterminizedAlphaBetaAgent(samples=0),
        lambda: DeterminizedAlphaBetaAgent(depth=0),
    ],
)
def test_search_agents_reject_empty_budgets(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="positive"):
        build()
