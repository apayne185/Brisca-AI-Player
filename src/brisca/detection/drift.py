"""Population Stability Index (PSI) for monitoring feature drift.

The detector stores, per feature, decile edges of its training data. In
production the same bins are filled from recently scored sessions and compared
with PSI. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major
shift worth investigating (or retraining for).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

_EPS = 1e-4


def reference_profile(
    X: npt.NDArray[np.float64], features: Sequence[str], bins: int = 10
) -> dict[str, dict[str, list[float]]]:
    """Quantile bin edges and training proportions for each feature."""
    profile = {}
    for j, name in enumerate(features):
        values = X[:, j][~np.isnan(X[:, j])]
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        profile[name] = {
            "edges": edges.tolist(),
            "proportions": _proportions(values, edges).tolist(),
        }
    return profile


def _proportions(
    values: npt.NDArray[np.float64], edges: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return np.asarray(counts / max(1, int(counts.sum())), dtype=np.float64)


def psi(reference: dict[str, list[float]], values: Sequence[float | None]) -> float:
    """PSI between a feature's training distribution and ``values``."""
    observed = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if observed.size == 0:
        return float("nan")
    expected = np.clip(np.asarray(reference["proportions"]), _EPS, None)
    actual = np.clip(_proportions(observed, np.asarray(reference["edges"])), _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
