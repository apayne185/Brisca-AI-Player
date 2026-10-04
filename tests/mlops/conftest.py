from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("optuna")
pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Each test gets its own tracking store and artifact directory."""
    import mlflow

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield tmp_path
    if mlflow.active_run():
        mlflow.end_run()
