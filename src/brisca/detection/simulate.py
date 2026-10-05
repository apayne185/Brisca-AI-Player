"""Synthetic gameplay telemetry for a population of humans and bots.

There is no public Brisca telemetry, so this module simulates it. The aim is
not realism in every detail but a population with the structure that makes
bot detection hard in practice:

* Humans vary in skill and tempo, tire over a session, get distracted, and
  think longer when more is at stake.
* Bots run one of the project's agents. Their timing comes in four styles of
  increasing sophistication: ``naive`` (fast and flat), ``jittered`` (random
  human-scale delays), ``humanized`` (delays shaped like a human's, but keyed
  on a cruder notion of difficulty) and ``mimic`` (the human timing model
  itself, stakes included). Evaluation holds ``mimic`` out of training to
  test the detector against an adversary it has never seen.

Each move is logged as an event, with annotations a real platform could
compute by replaying its server-side game logs: agreement with known bot
policies and, in endgames that are solved exactly, whether the move was optimal.
"""

from __future__ import annotations

import csv
import math
import multiprocessing
import random
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any

import duckdb

from brisca.agents import GreedyAgent, HeuristicAgent, ISMCTSAgent, RandomAgent
from brisca.agents.alphabeta import alphabeta
from brisca.agents.base import Agent
from brisca.cards import Card
from brisca.engine import GameState, beats, new_game, step, winner
from brisca.observation import Observation, observe

BOT_STYLES = ("naive", "jittered", "humanized", "mimic")
BOT_POLICIES = ("heuristic", "ppo", "ismcts")


@dataclass(frozen=True)
class Profile:
    player_id: int
    is_bot: bool
    bot_style: str | None
    policy: str
    skill: float
    """Humans only: 0 (novice) to 1 (expert). Bots: NaN."""
    tempo: float
    """Typical seconds per considered move."""


@dataclass(frozen=True, slots=True)
class MoveEvent:
    player_id: int
    session_id: int
    game_id: int
    move_idx: int
    ts: float
    think_s: float
    card: int
    n_options: int
    stakes: float
    is_leading: bool
    stock_size: int
    agrees_heuristic: bool
    agrees_greedy: bool
    endgame_decision: bool
    endgame_optimal: bool | None


@dataclass(frozen=True, slots=True)
class GameEvent:
    player_id: int
    session_id: int
    game_id: int
    start_ts: float
    end_ts: float
    points: int
    won: bool


def stakes(obs: Observation) -> float:
    """How consequential the decision is, in [0, 1].

    Following, it is the spread of immediate point swings across the hand;
    leading, the spread of points the player might give away.
    """
    hand = obs.hand
    if obs.current_trick:
        lead = obs.current_trick[0]
        swings = [(lead.points + c.points) * (1 if beats(c, lead, obs.trump) else -1) for c in hand]
    else:
        swings = [c.points for c in hand]
    return min(1.0, (max(swings) - min(swings)) / 22)


def sample_profile(player_id: int, bot_rate: float, rng: random.Random) -> Profile:
    tempo = rng.lognormvariate(math.log(1.6), 0.35)
    if rng.random() >= bot_rate:
        return Profile(player_id, False, None, "human", rng.betavariate(2, 2), tempo)
    return Profile(
        player_id, True, rng.choice(BOT_STYLES), rng.choice(BOT_POLICIES), math.nan, tempo
    )


class HumanPolicy:
    """Plays well with a probability that rises with skill and falls with fatigue."""

    name = "human"

    def __init__(self, skill: float, rng: random.Random) -> None:
        self.skill = skill
        self.rng = rng
        self.heuristic = HeuristicAgent()
        self.greedy = GreedyAgent()
        self.random = RandomAgent(seed=rng.randrange(2**31))
        self.fatigue = 0.0

    def act(self, obs: Observation) -> Card:
        p_good = max(0.1, 0.35 + 0.55 * self.skill - self.fatigue)
        u = self.rng.random()
        if u < p_good:
            return self.heuristic.act(obs)
        if u < p_good + (1 - p_good) / 2:
            return self.greedy.act(obs)
        return self.random.act(obs)


def think_time(profile: Profile, obs: Observation, rng: random.Random) -> float:
    n_options = len(obs.hand)
    if profile.bot_style == "naive":
        return 0.08 + 0.05 * rng.random()
    if profile.bot_style == "jittered":
        return rng.lognormvariate(math.log(1.5), 0.5)
    if n_options == 1:  # humans (and bots imitating them) play forced cards quickly
        return profile.tempo * 0.35 * rng.lognormvariate(0, 0.3)
    if profile.bot_style == "humanized":
        # Imitates human pacing, but scales with hand size rather than true stakes.
        return profile.tempo * (0.6 + 0.4 * n_options / 3) * rng.lognormvariate(0, 0.45)
    # Humans, and mimic bots copying them exactly.
    seconds = profile.tempo * (1 + 1.2 * stakes(obs)) * rng.lognormvariate(0, 0.45)
    if rng.random() < 0.03:  # distraction
        seconds += rng.expovariate(1 / 8)
    return seconds


def between_games(profile: Profile, rng: random.Random) -> float:
    if profile.bot_style == "naive":
        return 1.0
    if profile.bot_style == "jittered":
        return rng.lognormvariate(math.log(5), 0.5)
    gap = rng.lognormvariate(math.log(15), 0.8)
    return gap + (rng.expovariate(1 / 300) if rng.random() < 0.1 else 0.0)


def _endgame_optimal(state: GameState, card: Card) -> bool:
    """Whether ``card`` is optimal in a perfect-information endgame (empty stock)."""
    player, depth = state.to_play, sum(len(h) for h in state.hands)

    def value(c: Card) -> float:
        return alphabeta(step(state, c), depth - 1, -math.inf, math.inf, player)

    return value(card) == max(value(c) for c in state.hands[player])


def _bot_policy(profile: Profile, seed: int) -> Agent:
    if profile.policy == "heuristic":
        return HeuristicAgent()
    if profile.policy == "ismcts":
        return ISMCTSAgent(iterations=150, seed=seed)
    from brisca.rl import PolicyAgent

    return PolicyAgent.from_checkpoint("models/ppo-v2.pt")


def simulate_player(profile: Profile, seed: int) -> tuple[list[MoveEvent], list[GameEvent]]:
    rng = random.Random(seed)
    human = None if profile.is_bot else HumanPolicy(profile.skill, rng)
    policy: Agent = human if human is not None else _bot_policy(profile, seed)
    heuristic, greedy = HeuristicAgent(), GreedyAgent()
    house: list[Agent] = [HeuristicAgent(), GreedyAgent(), ISMCTSAgent(iterations=30, seed=seed)]

    moves: list[MoveEvent] = []
    games: list[GameEvent] = []
    clock = rng.uniform(0, 86_400)
    game_id = 0
    for session_id in range(rng.randint(2, 4)):
        clock += rng.uniform(3_600, 86_400)
        for game_in_session in range(rng.randint(3, 6)):
            if human is not None:
                human.fatigue = 0.03 * game_in_session
            seat = rng.randrange(2)
            opponent = rng.choice(house)
            state = new_game(rng.randrange(2**31), first_player=rng.randrange(2))
            start, move_idx = clock, 0
            while not state.is_terminal:
                obs = observe(state, state.to_play)
                if state.to_play != seat:
                    state = step(state, opponent.act(obs))
                    clock += rng.uniform(0.5, 2.0)
                    continue
                card = policy.act(obs)
                think = think_time(profile, obs, rng)
                clock += think
                endgame = not state.stock and len(obs.hand) > 1
                moves.append(
                    MoveEvent(
                        player_id=profile.player_id,
                        session_id=session_id,
                        game_id=game_id,
                        move_idx=move_idx,
                        ts=clock,
                        think_s=think,
                        card=card.ordinal,
                        n_options=len(obs.hand),
                        stakes=stakes(obs),
                        is_leading=not obs.current_trick,
                        stock_size=obs.stock_size,
                        agrees_heuristic=heuristic.act(obs) == card,
                        agrees_greedy=greedy.act(obs) == card,
                        endgame_decision=endgame,
                        endgame_optimal=_endgame_optimal(state, card) if endgame else None,
                    )
                )
                state = step(state, card)
                move_idx += 1
            games.append(
                GameEvent(
                    profile.player_id,
                    session_id,
                    game_id,
                    start,
                    clock,
                    state.scores[seat],
                    winner(state) == seat,
                )
            )
            game_id += 1
            clock += between_games(profile, rng)
    return moves, games


def _simulate_chunk(profiles: list[Profile], seed: int) -> tuple[list[MoveEvent], list[GameEvent]]:
    moves: list[MoveEvent] = []
    games: list[GameEvent] = []
    for profile in profiles:
        m, g = simulate_player(profile, seed * 1_000_003 + profile.player_id)
        moves.extend(m)
        games.extend(g)
    return moves, games


def _chunks(items: list[Profile], size: int) -> Iterator[list[Profile]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


_SQL_TYPES = {"int": "INTEGER", "float": "DOUBLE", "bool": "BOOLEAN", "str": "VARCHAR"}


def _load(
    con: duckdb.DuckDBPyConnection, name: str, cls: type, rows: list[tuple[Any, ...]]
) -> None:
    """Bulk-load rows through a temporary CSV; DuckDB's executemany is row-at-a-time."""
    columns = {f.name: _SQL_TYPES[str(f.type).split(" | ")[0]] for f in fields(cls)}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.csv"
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv(?, header = false, "
            f"columns = {columns!r}, nullstr = '')",
            [str(path)],
        )


def simulate_population(
    db: str | Path,
    players: int,
    bot_rate: float = 0.15,
    seed: int = 0,
    workers: int | None = None,
) -> dict[str, Any]:
    """Simulate ``players`` players and write ``players``, ``games`` and ``moves`` tables."""
    rng = random.Random(seed)
    profiles = [sample_profile(i, bot_rate, rng) for i in range(players)]
    chunks = list(_chunks(profiles, 10))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_simulate_chunk, chunks, [seed] * len(chunks)))

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    _load(con, "players", Profile, [astuple(p) for p in profiles])
    _load(con, "games", GameEvent, [astuple(g) for _, gs in results for g in gs])
    _load(con, "moves", MoveEvent, [astuple(m) for ms, _ in results for m in ms])
    summary = {
        "players": players,
        "bots": sum(p.is_bot for p in profiles),
        "moves": sum(len(ms) for ms, _ in results),
    }
    con.close()
    return summary
