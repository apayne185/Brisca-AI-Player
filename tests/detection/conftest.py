from pathlib import Path

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("torch")


@pytest.fixture(scope="session")
def telemetry_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small population, simulated once for the whole test session."""
    from brisca.detection.simulate import simulate_population

    db = tmp_path_factory.mktemp("telemetry") / "telemetry.duckdb"
    simulate_population(db, players=80, bot_rate=0.4, seed=1, workers=4)
    return db
