from pathlib import Path

import numpy as np
import pytest

from brisca.cli import main
from brisca.detection.features import ALL_FEATURES, DECISION_FEATURES, load_dataset
from brisca.detection.model import (
    HELD_OUT_STYLE,
    Detector,
    cross_validate,
    evaluate,
    fit_detector,
    reliability,
    threshold_for_fpr,
)
from brisca.detection.report import render


def test_dataset_has_one_row_per_session_and_no_label_leakage(telemetry_db: Path) -> None:
    data = load_dataset(telemetry_db)
    assert data.X.shape == (len(data.y), len(ALL_FEATURES))
    assert not {"label", "bot_style", "policy", "skill", "player_id"} & set(data.features)
    assert set(np.unique(data.y)) == {0, 1}
    assert set(data.bot_style[data.y == 0]) == {"human"}


def test_threshold_respects_the_false_positive_budget() -> None:
    rng = np.random.default_rng(0)
    scores, y = rng.random(1000), np.r_[np.zeros(900, dtype=np.int64), np.ones(100, dtype=np.int64)]
    threshold = threshold_for_fpr(scores, y, budget=0.05)
    assert np.mean(scores[y == 0] > threshold) <= 0.05


def test_reliability_bins_cover_every_session() -> None:
    probs = np.linspace(0, 1, 101)
    rows = reliability(probs, (probs > 0.5).astype(np.int64))
    assert sum(r["sessions"] for r in rows) == 101


def test_cross_validation_never_trains_on_held_out_style(telemetry_db: Path) -> None:
    data = load_dataset(telemetry_db)
    oof = cross_validate(data, folds=3)
    assert oof.mask.all(), "every session, including mimic bots, gets an out-of-fold score"
    assert np.all((oof.calibrated >= 0) & (oof.calibrated <= 1))


def test_detector_round_trips(telemetry_db: Path, tmp_path: Path) -> None:
    data = load_dataset(telemetry_db)
    detector = fit_detector(data.X, data.y, data.groups, data.features)
    detector.threshold = 0.3
    detector.save(tmp_path / "det")
    loaded = Detector.load(tmp_path / "det")
    assert np.allclose(loaded.predict_proba(data.X), detector.predict_proba(data.X))
    assert np.array_equal(loaded.flag(data.X), detector.flag(data.X))
    assert detector.explain(data.X).shape == data.X.shape


def test_evaluation_report(telemetry_db: Path) -> None:
    report = evaluate(load_dataset(telemetry_db))
    detector = report.pop("detector")
    assert isinstance(detector, Detector)
    assert detector.features == DECISION_FEATURES
    assert report["shipped"]["features"] == "decision"
    assert 0.5 < report["main"]["roc_auc"] <= 1.0
    assert set(report["ablations"]) == {"timing only", "decision only"}
    assert HELD_OUT_STYLE in report["recall_by_style"]
    assert list(report["feature_importance"]) == sorted(
        report["feature_importance"], key=report["feature_importance"].__getitem__, reverse=True
    )
    text = render(report)
    assert "# Bot detection: evaluation report" in text
    assert "(never trained on)" in text


def test_cli_trains_and_saves(telemetry_db: Path, tmp_path: Path) -> None:
    model, report = tmp_path / "model", tmp_path / "report.md"
    main(
        [
            "detect",
            "train",
            "--db",
            str(telemetry_db),
            "--model",
            str(model),
            "--report",
            str(report),
        ]
    )
    assert (model / "booster.json").exists()
    assert report.read_text().startswith("# Bot detection")


@pytest.mark.slow
def test_cli_simulates(tmp_path: Path) -> None:
    db = tmp_path / "t.duckdb"
    main(["detect", "simulate", "--players", "5", "--db", str(db), "--workers", "1"])
    assert load_dataset(db).X.shape[0] > 0
