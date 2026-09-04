"""Тесты Adversarial Validation."""
import numpy as np
import pandas as pd

from drift_guardian.adversarial import adversarial_validation
from drift_guardian.config import DriftConfig


def _make(rng, n, shift=0.0):
    return pd.DataFrame(
        {
            "a": rng.normal(0 + shift, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.choice(list("xyz"), n),
        }
    )


def test_no_drift_auc_near_half():
    rng = np.random.default_rng(10)
    ref, cur = _make(rng, 1500), _make(rng, 1500)
    report = adversarial_validation(ref, cur, ["a", "b", "c"], DriftConfig())
    assert report is not None
    assert report.roc_auc < 0.6


def test_strong_drift_high_auc_and_importances():
    rng = np.random.default_rng(11)
    ref, cur = _make(rng, 1500), _make(rng, 1500, shift=1.5)
    report = adversarial_validation(ref, cur, ["a", "b", "c"], DriftConfig())
    assert report is not None
    assert report.roc_auc > 0.65
    assert report.severity == "critical"
    assert report.top_features[0]["feature"] == "a"  # сдвинута именно колонка "a"


def test_too_small_sample_returns_none():
    rng = np.random.default_rng(12)
    ref, cur = _make(rng, 30), _make(rng, 30)
    assert adversarial_validation(ref, cur, ["a", "b", "c"], DriftConfig()) is None
