"""FastAPI application: agent inference, bot scoring, a playable demo and metrics.

    uvicorn brisca.serving.app:app --port 8000

Configuration comes from environment variables (see ``Settings``).
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from brisca.agents.base import Agent
from brisca.agents.ismcts import ISMCTSAgent
from brisca.cards import Card
from brisca.engine import GameState, IllegalActionError, new_game, step, winner
from brisca.observation import observe
from brisca.serving.metrics import Metrics
from brisca.serving.schemas import (
    AgentInfo,
    BotScoreRequest,
    BotScoreResponse,
    Contribution,
    GameView,
    Hint,
    MoveAdvice,
    MoveIn,
    MoveRequest,
    MoveResponse,
    NewGameRequest,
    TrickView,
)
from brisca.serving.store import GameStore
from brisca.tournament import AgentSpec, TournamentConfig, build_agent

log = logging.getLogger("brisca.serving")
STATIC = Path(__file__).parent / "static"
HUMAN, AI = 0, 1


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("BRISCA_MODELS_DIR", "models")))
    agents_config: Path | None = field(
        default_factory=lambda: Path(p) if (p := os.getenv("BRISCA_AGENTS_CONFIG")) else None
    )
    drift_window: int = field(default_factory=lambda: int(os.getenv("BRISCA_DRIFT_WINDOW", "500")))
    drift_every: int = 50
    """Recompute PSI after this many scored sessions."""
    llm_explanations: bool = field(
        default_factory=lambda: os.getenv("BRISCA_LLM_EXPLANATIONS", "") == "1"
    )
    """Ask Claude to explain hints (needs the 'llm' extra and Anthropic credentials)."""
    hint_iterations: int = 1500

    def agent_specs(self) -> tuple[AgentSpec, ...]:
        if self.agents_config is not None:
            return TournamentConfig.from_toml(self.agents_config).agents
        specs = [
            AgentSpec("random", "random", {"seed": 0}),
            AgentSpec("greedy", "greedy"),
            AgentSpec("heuristic", "heuristic"),
            AgentSpec("ismcts", "ismcts", {"iterations": 500, "seed": 0}),
            AgentSpec("alphabeta", "alphabeta", {"samples": 10, "depth": 4, "seed": 0}),
        ]
        ppo = self.models_dir / "ppo-v2.pt"
        if ppo.exists():
            specs.append(AgentSpec("ppo", "ppo", {"path": str(ppo)}))
        return tuple(specs)


class Service:
    """Everything the routes need, loaded once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metrics = Metrics.create()
        self.games = GameStore()
        self.specs: dict[str, AgentSpec] = {}
        self.agents: dict[str, Agent] = {}
        self.detector: Any = None
        self.recent: deque[dict[str, float | None]] = deque(maxlen=settings.drift_window)
        self.psi: dict[str, float] = {}
        self._scored = 0

    def load(self) -> None:
        for spec in self.settings.agent_specs():
            try:
                self.agents[spec.id] = build_agent(spec)
                self.specs[spec.id] = spec
            except (ImportError, OSError) as exc:  # optional extras or missing files
                log.warning("skipping agent %s: %s", spec.id, exc)
        detector_dir = self.settings.models_dir / "bot-detector"
        try:
            from brisca.detection.model import Detector

            self.detector = Detector.load(detector_dir)
        except (ImportError, OSError) as exc:
            log.warning("bot detector unavailable: %s", exc)

    def agent(self, agent_id: str) -> Agent:
        try:
            return self.agents[agent_id]
        except KeyError:
            raise HTTPException(404, f"unknown agent {agent_id!r}") from None

    def decide(self, agent_id: str, state_or_obs: Any) -> Card:
        agent = self.agent(agent_id)
        start = time.perf_counter()
        card = agent.act(state_or_obs)
        self.metrics.decision_latency.labels(agent_id).observe(time.perf_counter() - start)
        return card

    def record_for_drift(self, features: dict[str, float | None]) -> None:
        self.recent.append(features)
        self._scored += 1
        if self._scored % self.settings.drift_every == 0:
            self.refresh_drift()

    def refresh_drift(self) -> None:
        from brisca.detection.drift import psi

        reference = self.detector.metadata.get("reference", {})
        for feature, profile in reference.items():
            value = psi(profile, [row.get(feature) for row in self.recent])
            if not math.isnan(value):
                self.psi[feature] = value
                self.metrics.feature_psi.labels(feature).set(value)


def create_app(settings: Settings | None = None) -> FastAPI:
    service = Service(settings or Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service.load()
        yield

    app = FastAPI(
        title="Brisca AI",
        description="Game-playing agents and gameplay bot detection for Brisca.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.middleware("http")
    async def record_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        service.metrics.latency.labels(route).observe(time.perf_counter() - start)
        service.metrics.requests.labels(route, request.method, str(response.status_code)).inc()
        return response

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(response: Response) -> dict[str, Any]:
        status = {"agents": sorted(service.agents), "bot_detector": service.detector is not None}
        if not service.agents:
            response.status_code = 503
        return status

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(service.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/agents", tags=["agents"])
    def agents() -> list[AgentInfo]:
        return [AgentInfo(id=s.id, type=s.type, params=s.params) for s in service.specs.values()]

    @app.post("/v1/move", tags=["agents"])
    def move(request: MoveRequest) -> MoveResponse:
        """Choose a card for the player to move, from that player's observation."""
        obs = request.observation.to_observation()
        if obs.to_play != obs.player:
            raise HTTPException(422, "it is not this player's turn")
        start = time.perf_counter()
        try:
            card = service.decide(request.agent, obs)
        except ValueError as exc:  # e.g. an observation no real game could produce
            raise HTTPException(422, str(exc)) from exc
        return MoveResponse(
            agent=request.agent, card=str(card), latency_ms=1000 * (time.perf_counter() - start)
        )

    @app.post("/v1/bot-score", tags=["bot detection"])
    def bot_score(request: BotScoreRequest) -> BotScoreResponse:
        """Probability that a session was played by a bot, with per-feature explanations."""
        detector = service.detector
        if detector is None:
            raise HTTPException(503, "bot detector not loaded")
        missing = set(detector.features) - set(request.features)
        if missing:
            raise HTTPException(422, f"missing features: {sorted(missing)}")
        values = [request.features[f] for f in detector.features]
        row = np.array([[np.nan if v is None else v for v in values]], dtype=np.float64)
        probability = float(detector.predict_proba(row)[0])
        flagged = bool(detector.flag(row)[0])
        contributions = sorted(
            (
                Contribution(feature=f, value=v, contribution=float(c))
                for f, v, c in zip(detector.features, values, detector.explain(row)[0], strict=True)
            ),
            key=lambda c: -abs(c.contribution),
        )
        service.metrics.bot_scores.observe(probability)
        if flagged:
            service.metrics.flags.inc()
        service.record_for_drift(dict(request.features))
        return BotScoreResponse(
            probability=probability, flagged=flagged, contributions=contributions
        )

    @app.get("/v1/drift", tags=["bot detection"])
    def drift() -> dict[str, Any]:
        """PSI per feature for the most recent scored sessions vs the training data."""
        return {"window": len(service.recent), "psi": service.psi}

    def view(game_id: str, state: GameState, agent_id: str) -> GameView:
        last = state.history[-1] if state.history else None
        names = {HUMAN: "you", AI: agent_id}
        result = None
        if state.is_terminal:
            won = winner(state)
            result = "draw" if won is None else ("win" if won == HUMAN else "loss")
        return GameView(
            game_id=game_id,
            agent=agent_id,
            hand=[str(c) for c in state.hands[HUMAN]],
            trump_card=str(state.trump_card),
            trump_drawn=not state.stock,
            current_trick=[str(c) for c in state.current_trick],
            your_turn=not state.is_terminal and state.to_play == HUMAN,
            scores={"you": state.scores[HUMAN], agent_id: state.scores[AI]},
            stock_size=len(state.stock),
            last_trick=(
                TrickView(
                    leader=names[last.leader],
                    cards=[str(c) for c in last.cards],
                    winner=names[last.winner],
                    points=last.points,
                )
                if last
                else None
            ),
            finished=state.is_terminal,
            result=result,
        )

    def ai_turns(state: GameState, agent_id: str) -> GameState:
        while not state.is_terminal and state.to_play == AI:
            state = step(state, service.decide(agent_id, observe(state, AI)))
        return state

    @app.post("/v1/games", tags=["demo"])
    def create_game(request: NewGameRequest) -> GameView:
        service.agent(request.agent)
        seed = request.seed if request.seed is not None else random.randrange(2**31)
        state = new_game(seed, first_player=seed % 2)
        state = ai_turns(state, request.agent)
        game_id = service.games.create(state, request.agent)
        service.metrics.games.labels("started", request.agent).inc()
        return view(game_id, state, request.agent)

    @app.get("/v1/games/{game_id}", tags=["demo"])
    def get_game(game_id: str) -> GameView:
        game = service.games.get(game_id)
        if game is None:
            raise HTTPException(404, "game not found or expired")
        return view(game_id, game.state, game.agent_id)

    @app.post("/v1/games/{game_id}/moves", tags=["demo"])
    def play(game_id: str, move: MoveIn) -> GameView:
        game = service.games.get(game_id)
        if game is None:
            raise HTTPException(404, "game not found or expired")
        if game.state.to_play != HUMAN or game.state.is_terminal:
            raise HTTPException(409, "it is not your turn")
        try:
            state = step(game.state, Card.parse(move.card))
        except IllegalActionError as exc:
            raise HTTPException(422, f"{move.card} is not in your hand") from exc
        state = ai_turns(state, game.agent_id)
        service.games.update(game_id, state)
        if state.is_terminal:
            service.metrics.games.labels("finished", game.agent_id).inc()
        return view(game_id, state, game.agent_id)

    @app.post("/v1/games/{game_id}/hint", tags=["demo"])
    def game_hint(game_id: str) -> Hint:
        """The engine's recommended card, its win estimates and (optionally) an explanation."""
        game = service.games.get(game_id)
        if game is None:
            raise HTTPException(404, "game not found or expired")
        if game.state.to_play != HUMAN or game.state.is_terminal:
            raise HTTPException(409, "it is not your turn")
        obs = observe(game.state, HUMAN)
        moves = ISMCTSAgent(iterations=service.settings.hint_iterations).search(obs)
        explanation = None
        if service.settings.llm_explanations:
            try:
                from brisca.llm.explain import explain

                explanation = explain(obs, moves)
            except Exception:  # an explanation is optional; never fail the hint
                log.exception("hint explanation failed")
        return Hint(
            card=str(moves[0].card),
            moves=[MoveAdvice(card=str(m.card), win_chance=m.value) for m in moves],
            explanation=explanation,
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
