"""Train, evaluate and persist the bot detector.

Design choices, each driven by how the model would be used (flagging accounts
for human review, where a false accusation is costly):

* **Grouped cross-validation.** All of a player's sessions land in the same
  fold, so the model is never scored on a player it has seen in training.
* **Operating point by false-positive budget.** The threshold is set so that at
  most 1% of human sessions are flagged; recall is reported at that point.
* **Calibration.** Isotonic regression fitted on players held out from the
  booster, so flagged probabilities can be read as probabilities.
* **Unseen adversary.** ``mimic`` bots are never trained on; recall on them
  estimates how the detector holds up when bots adapt.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brisca.detection.drift import reference_profile
from brisca.detection.features import (
    ALL_FEATURES,
    DECISION_FEATURES,
    TIMING_FEATURES,
    Dataset,
    FloatArray,
)

HELD_OUT_STYLE = "mimic"
FPR_BUDGET = 0.01
ABLATIONS = {
    "timing only": TIMING_FEATURES,
    "decision only": DECISION_FEATURES,
}
SKILL_BANDS = {"novice (<0.33)": (0.0, 0.33), "mid": (0.33, 0.67), "expert (>0.67)": (0.67, 1.01)}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "eval_metric": "aucpr",
    "n_jobs": 4,
}


@dataclass
class Detector:
    features: tuple[str, ...]
    booster: xgb.XGBClassifier
    calibrator: IsotonicRegression
    threshold: float = 0.5
    """Raw-score threshold for flagging, set from the false-positive budget."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def raw_score(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.booster.predict_proba(X)[:, 1], dtype=np.float64)

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Calibrated probability that each session is a bot."""
        return np.asarray(self.calibrator.predict(self.raw_score(X)), dtype=np.float64)

    def flag(self, X: FloatArray) -> npt.NDArray[np.bool_]:
        return self.raw_score(X) > self.threshold

    def explain(self, X: FloatArray) -> FloatArray:
        """Per-feature SHAP contributions (log-odds), from XGBoost's TreeSHAP."""
        contribs = self.booster.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return np.asarray(contribs[:, :-1], dtype=np.float64)  # last column is the bias

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "booster.json")
        (directory / "detector.json").write_text(
            json.dumps(
                {
                    "features": list(self.features),
                    "threshold": self.threshold,
                    "calibration": {
                        "x": self.calibrator.X_thresholds_.tolist(),
                        "y": self.calibrator.y_thresholds_.tolist(),
                    },
                    "metadata": self.metadata,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> Detector:
        directory = Path(directory)
        spec = json.loads((directory / "detector.json").read_text())
        booster = xgb.XGBClassifier()
        booster.load_model(directory / "booster.json")
        calibrator = IsotonicRegression(out_of_bounds="clip")
        x, y = np.asarray(spec["calibration"]["x"]), np.asarray(spec["calibration"]["y"])
        calibrator.fit(x, y)
        return cls(
            tuple(spec["features"]), booster, calibrator, spec["threshold"], spec["metadata"]
        )


def fit_detector(
    X: FloatArray,
    y: npt.NDArray[np.int64],
    groups: npt.NDArray[np.int64],
    features: tuple[str, ...],
    seed: int = 0,
) -> Detector:
    """Fit the booster on 80% of players and the calibrator on the other 20%."""
    fit_idx, cal_idx = next(
        GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups)
    )
    booster = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    booster.fit(X[fit_idx], y[fit_idx])
    raw = booster.predict_proba(X[cal_idx])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, y[cal_idx])
    return Detector(features, booster, calibrator)


def threshold_for_fpr(
    scores: FloatArray, y: npt.NDArray[np.int64], budget: float = FPR_BUDGET
) -> float:
    """Smallest threshold at which at most ``budget`` of negatives score above it."""
    return float(np.quantile(scores[y == 0], 1 - budget, method="higher"))


@dataclass(frozen=True)
class OutOfFold:
    raw: FloatArray
    calibrated: FloatArray
    mask: npt.NDArray[np.bool_]
    """Rows that received an out-of-fold prediction."""


def cross_validate(
    data: Dataset,
    features: tuple[str, ...] = ALL_FEATURES,
    folds: int = 5,
    seed: int = 0,
    exclude_style: str | None = HELD_OUT_STYLE,
) -> OutOfFold:
    """Out-of-fold scores with player-grouped folds; ``exclude_style`` never trains."""
    X, y = data.select(features), data.y
    raw = np.full(len(y), np.nan)
    calibrated = np.full(len(y), np.nan)
    trainable = data.bot_style != exclude_style
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y, data.groups):
        train_idx = train_idx[trainable[train_idx]]
        detector = fit_detector(X[train_idx], y[train_idx], data.groups[train_idx], features, seed)
        raw[test_idx] = detector.raw_score(X[test_idx])
        calibrated[test_idx] = detector.predict_proba(X[test_idx])
    return OutOfFold(raw, calibrated, ~np.isnan(raw))


def ranking_metrics(scores: FloatArray, y: npt.NDArray[np.int64]) -> dict[str, float]:
    threshold = threshold_for_fpr(scores, y)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "recall_at_1pct_fpr": float(np.mean(scores[y == 1] > threshold)),
    }


def _logistic_oof(data: Dataset, seed: int) -> FloatArray:
    X, y = data.select(ALL_FEATURES), data.y
    scores = np.full(len(y), np.nan)
    trainable = data.bot_style != HELD_OUT_STYLE
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y, data.groups):
        train_idx = train_idx[trainable[train_idx]]
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
        )
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return scores


def _seen_and_unseen(raw: FloatArray, data: Dataset) -> dict[str, float]:
    """Ranking metrics on trained-on styles, plus recall on the held-out style.

    Both use the threshold that meets the false-positive budget on humans.
    """
    seen = data.bot_style != HELD_OUT_STYLE
    threshold = threshold_for_fpr(raw[seen], data.y[seen])
    return {
        **ranking_metrics(raw[seen], data.y[seen]),
        "unseen_recall": float(np.mean(raw[data.bot_style == HELD_OUT_STYLE] > threshold)),
    }


def reliability(probs: FloatArray, y: npt.NDArray[np.int64], bins: int = 5) -> list[dict[str, Any]]:
    """Mean predicted vs observed bot rate in equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        in_bin = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if in_bin.any():
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "sessions": int(in_bin.sum()),
                    "predicted": float(probs[in_bin].mean()),
                    "observed": float(y[in_bin].mean()),
                }
            )
    return rows


FEATURE_SETS = {"all": ALL_FEATURES, "timing": TIMING_FEATURES, "decision": DECISION_FEATURES}


def evaluate(data: Dataset, seed: int = 0, ship: str = "decision") -> dict[str, Any]:
    """The full evaluation behind the report and the model card.

    Diagnostics (slices, calibration, SHAP) describe the all-features model.
    The returned ``detector`` uses the ``ship`` feature set, by default
    decision features only, which hold up against the unseen adversary.
    """
    seen = data.bot_style != HELD_OUT_STYLE
    main = cross_validate(data, ALL_FEATURES, seed=seed)
    y_seen = data.y[seen]
    raw_seen = main.raw[seen]
    threshold = threshold_for_fpr(raw_seen, y_seen)
    flagged = main.raw > threshold

    report: dict[str, Any] = {
        "sessions": len(data.y),
        "players": len(np.unique(data.groups)),
        "bot_share": float(data.y.mean()),
        "main": {
            **_seen_and_unseen(main.raw, data),
            "brier_raw": float(brier_score_loss(y_seen, raw_seen)),
            "brier_calibrated": float(brier_score_loss(y_seen, main.calibrated[seen])),
        },
        "ablations": {
            name: _seen_and_unseen(cross_validate(data, feats, seed=seed).raw, data)
            for name, feats in ABLATIONS.items()
        },
        "logistic_baseline": _seen_and_unseen(_logistic_oof(data, seed), data),
        "recall_by_style": {
            style: float(flagged[data.bot_style == style].mean())
            for style in sorted(set(data.bot_style) - {"human"})
        },
        "recall_by_policy": {
            policy: float(flagged[(data.policy == policy) & seen].mean())
            for policy in sorted(set(data.policy) - {"human"})
        },
        "human_fpr_by_skill": {},
        "reliability": reliability(main.calibrated[seen], y_seen),
    }
    humans = data.bot_style == "human"
    for label, (lo, hi) in SKILL_BANDS.items():
        in_band = humans & (data.skill >= lo) & (data.skill < hi)
        report["human_fpr_by_skill"][label] = float(flagged[in_band].mean())

    diagnostic = fit_detector(
        data.X[seen], data.y[seen], data.groups[seen], data.features, seed=seed
    )
    shap = np.abs(diagnostic.explain(data.X[seen])).mean(axis=0)
    report["feature_importance"] = dict(
        sorted(zip(data.features, map(float, shap), strict=True), key=lambda kv: -kv[1])
    )

    features = FEATURE_SETS[ship]
    shipped_oof = main if ship == "all" else cross_validate(data, features, seed=seed)
    X = data.select(features)
    detector = fit_detector(X[seen], data.y[seen], data.groups[seen], features, seed=seed)
    detector.threshold = threshold_for_fpr(shipped_oof.raw[seen], y_seen)
    shipped_flagged = shipped_oof.raw > detector.threshold
    report["shipped"] = {
        "features": ship,
        **_seen_and_unseen(shipped_oof.raw, data),
        "recall_by_policy": {
            policy: float(shipped_flagged[data.policy == policy].mean())
            for policy in sorted(set(data.policy) - {"human"})
        },
        "human_fpr_by_skill": {
            label: float(shipped_flagged[humans & (data.skill >= lo) & (data.skill < hi)].mean())
            for label, (lo, hi) in SKILL_BANDS.items()
        },
    }
    detector.metadata = {
        "cv": {k: v for k, v in report["shipped"].items() if not isinstance(v, dict)},
        "fpr_budget": FPR_BUDGET,
        "reference": reference_profile(X[seen], features),
    }
    report["detector"] = detector
    return report
