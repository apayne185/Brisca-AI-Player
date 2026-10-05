import random
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xgboost")
pytest.importorskip("torch")

from fastapi.testclient import TestClient

from brisca import legal_actions, new_game, step
from brisca.observation import observe
from brisca.serving.app import Settings, create_app
from brisca.serving.store import GameStore

MODELS = Path(__file__).parents[2] / "models"
DECISION = {
    "agree_heuristic": 0.99,
    "agree_greedy": 0.7,
    "endgame_accuracy": 1.0,
    "avg_points": 70.0,
    "win_rate": 0.8,
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(models_dir=MODELS, drift_every=5))) as client:
        yield client


def observation_json(seed: int = 0, moves: int = 7) -> dict[str, object]:
    state = new_game(seed)
    rng = random.Random(seed)
    for _ in range(moves):
        state = step(state, rng.choice(legal_actions(state)))
    obs = observe(state, state.to_play)
    return {
        "player": obs.player,
        "hand": [str(c) for c in obs.hand],
        "trump_card": str(obs.trump_card),
        "current_trick": [str(c) for c in obs.current_trick],
        "to_play": obs.to_play,
        "scores": list(obs.scores),
        "history": [
            {"leader": t.leader, "cards": [str(c) for c in t.cards], "winner": t.winner}
            for t in obs.history
        ],
        "stock_size": obs.stock_size,
        "opponent_hand_size": obs.opponent_hand_size,
    }


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert {"random", "heuristic", "ismcts", "ppo"} <= set(ready["agents"])
    assert ready["bot_detector"] is True


@pytest.mark.parametrize("agent", ["random", "heuristic", "ismcts", "alphabeta", "ppo"])
def test_move_returns_a_card_from_the_hand(client: TestClient, agent: str) -> None:
    obs = observation_json()
    response = client.post("/v1/move", json={"agent": agent, "observation": obs})
    assert response.status_code == 200, response.text
    assert response.json()["card"] in obs["hand"]  # type: ignore[operator]


def test_move_validates_input(client: TestClient) -> None:
    obs = observation_json()
    assert client.post("/v1/move", json={"agent": "nope", "observation": obs}).status_code == 404
    bad_card = {**obs, "hand": ["9O"]}
    assert (
        client.post("/v1/move", json={"agent": "random", "observation": bad_card}).status_code
        == 422
    )
    wrong_turn = {**obs, "to_play": 1 - obs["player"]}  # type: ignore[operator]
    assert (
        client.post("/v1/move", json={"agent": "random", "observation": wrong_turn}).status_code
        == 422
    )
    impossible = {**obs, "stock_size": 30}
    assert (
        client.post("/v1/move", json={"agent": "ismcts", "observation": impossible}).status_code
        == 422
    )


def test_bot_score(client: TestClient) -> None:
    response = client.post("/v1/bot-score", json={"features": DECISION})
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["flagged"], bool)
    contributions = [abs(c["contribution"]) for c in body["contributions"]]
    assert contributions == sorted(contributions, reverse=True)
    assert {c["feature"] for c in body["contributions"]} == set(DECISION)


def test_bot_score_requires_every_feature(client: TestClient) -> None:
    response = client.post("/v1/bot-score", json={"features": {"agree_heuristic": 1.0}})
    assert response.status_code == 422
    assert "missing features" in response.json()["detail"]


def test_drift_is_tracked_after_enough_sessions(client: TestClient) -> None:
    for i in range(10):
        client.post("/v1/bot-score", json={"features": {**DECISION, "avg_points": 40.0 + i}})
    drift = client.get("/v1/drift").json()
    assert drift["window"] >= 10
    assert set(drift["psi"]) == set(DECISION)
    assert drift["psi"]["avg_points"] > 0.25, "a shifted feature shows major drift"
    assert "brisca_bot_feature_psi" in client.get("/metrics").text


def test_full_demo_game(client: TestClient) -> None:
    game = client.post("/v1/games", json={"agent": "heuristic", "seed": 11}).json()
    assert game["your_turn"]
    while not game["finished"]:
        game = client.post(
            f"/v1/games/{game['game_id']}/moves", json={"card": game["hand"][0]}
        ).json()
    assert sum(game["scores"].values()) == 120
    assert game["result"] in {"win", "loss", "draw"}
    assert client.get(f"/v1/games/{game['game_id']}").json() == game
    late = client.post(f"/v1/games/{game['game_id']}/moves", json={"card": "1O"})
    assert late.status_code == 409


def test_demo_game_errors(client: TestClient) -> None:
    assert client.post("/v1/games", json={"agent": "nope"}).status_code == 404
    assert client.get("/v1/games/missing").status_code == 404
    game = client.post("/v1/games", json={"agent": "random", "seed": 2}).json()
    not_mine = next(c for c in ("1O", "1C", "1E", "1B", "3O") if c not in game["hand"])
    move = client.post(f"/v1/games/{game['game_id']}/moves", json={"card": not_mine})
    assert move.status_code == 422


def test_metrics_and_frontend(client: TestClient) -> None:
    client.get("/health")
    text = client.get("/metrics").text
    assert 'brisca_http_requests_total{method="GET",route="/health",status="200"}' in text
    assert "brisca_agent_decision_seconds_bucket" in text
    assert "Brisca AI" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert {a["id"] for a in client.get("/v1/agents").json()} >= {"random", "ppo"}


def test_agents_can_come_from_a_config_file(tmp_path: Path) -> None:
    config = tmp_path / "agents.toml"
    config.write_text('[[agents]]\nid = "only"\ntype = "greedy"\n')
    settings = Settings(models_dir=tmp_path, agents_config=config)
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").json() == {"agents": ["only"], "bot_detector": False}
        assert client.post("/v1/bot-score", json={"features": DECISION}).status_code == 503


def test_store_evicts_oldest_games() -> None:
    store = GameStore(max_games=2)
    ids = [store.create(new_game(i), "random") for i in range(3)]
    assert len(store) == 2
    assert store.get(ids[0]) is None
    assert store.get(ids[2]) is not None
