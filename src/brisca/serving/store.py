"""In-memory store for demo games.

Bounded and thread-safe, enough for a single-process demo. A horizontally
scaled deployment would keep games in Redis or a database instead.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from brisca.engine import GameState


@dataclass
class Game:
    state: GameState
    agent_id: str
    updated: float = field(default_factory=time.monotonic)


class GameStore:
    def __init__(self, max_games: int = 1000, ttl_seconds: float = 3600) -> None:
        self.max_games = max_games
        self.ttl_seconds = ttl_seconds
        self._games: OrderedDict[str, Game] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._games)

    def create(self, state: GameState, agent_id: str) -> str:
        game_id = uuid.uuid4().hex
        with self._lock:
            self._evict()
            self._games[game_id] = Game(state, agent_id)
        return game_id

    def get(self, game_id: str) -> Game | None:
        with self._lock:
            game = self._games.get(game_id)
            if game is not None:
                self._games.move_to_end(game_id)
            return game

    def update(self, game_id: str, state: GameState) -> None:
        with self._lock:
            game = self._games[game_id]
            game.state = state
            game.updated = time.monotonic()
            self._games.move_to_end(game_id)

    def _evict(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        while self._games:
            oldest_id, oldest = next(iter(self._games.items()))
            if len(self._games) < self.max_games and oldest.updated >= cutoff:
                break
            del self._games[oldest_id]
