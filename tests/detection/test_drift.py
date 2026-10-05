import numpy as np
import pytest

from brisca.detection.drift import psi, reference_profile


@pytest.fixture
def reference() -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 1))
    return reference_profile(X, ["x"])


def test_reference_bins_hold_equal_shares(reference: dict[str, dict[str, list[float]]]) -> None:
    profile = reference["x"]
    assert len(profile["edges"]) == 9
    assert np.allclose(profile["proportions"], 0.1, atol=0.01)


def test_same_distribution_is_stable(reference: dict[str, dict[str, list[float]]]) -> None:
    sample = np.random.default_rng(1).normal(size=2000)
    assert psi(reference["x"], sample.tolist()) < 0.02


def test_shifted_distribution_is_flagged(reference: dict[str, dict[str, list[float]]]) -> None:
    shifted = np.random.default_rng(2).normal(loc=1.0, size=2000)
    assert psi(reference["x"], shifted.tolist()) > 0.25


def test_missing_values_are_ignored(reference: dict[str, dict[str, list[float]]]) -> None:
    assert np.isnan(psi(reference["x"], [None, float("nan")]))
    assert psi(reference["x"], [0.0, None]) >= 0.0
