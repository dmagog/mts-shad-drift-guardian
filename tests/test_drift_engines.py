"""Тесты статистических движков дрейфа."""
import numpy as np
import pandas as pd

from drift_guardian.config import DriftConfig
from drift_guardian.drift_engines import (
    analyze_categorical_column,
    analyze_numeric_column,
    psi_from_counts,
)


def _frames(ref_values, cur_values, name="x"):
    return pd.DataFrame({name: ref_values}), pd.DataFrame({name: cur_values})


def test_numeric_no_drift_is_ok():
    rng = np.random.default_rng(1)
    ref, cur = _frames(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "ok"


def test_numeric_strong_shift_is_critical():
    rng = np.random.default_rng(2)
    ref, cur = _frames(rng.normal(0, 1, 4000), rng.normal(1.0, 1, 4000))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "critical"
    psi_test = next(t for t in report.tests if t.name == "psi")
    assert psi_test.statistic > 0.2
    ks_test = next(t for t in report.tests if t.name == "ks")
    assert ks_test.p_value < 0.05


def test_numeric_constant_reference_does_not_crash():
    ref, cur = _frames(np.ones(100), np.ones(100) * 2)
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.kind == "numeric"  # главное — нет исключения на вырожденных бинах


def test_numeric_small_sample_skipped():
    ref, cur = _frames([1.0, 2.0, 3.0], [1.0, 2.0])
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "ok"
    assert report.tests[0].name == "skipped"


def test_categorical_shift_is_critical():
    rng = np.random.default_rng(3)
    ref, cur = _frames(
        rng.choice(list("abc"), 3000, p=[0.6, 0.3, 0.1]),
        rng.choice(list("abc"), 3000, p=[0.2, 0.3, 0.5]),
    )
    report = analyze_categorical_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "critical"


def test_categorical_same_distribution_is_ok():
    rng = np.random.default_rng(4)
    ref, cur = _frames(
        rng.choice(list("abc"), 3000, p=[0.5, 0.3, 0.2]),
        rng.choice(list("abc"), 3000, p=[0.5, 0.3, 0.2]),
    )
    report = analyze_categorical_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "ok"


def test_psi_zero_for_identical_counts():
    counts = np.array([100.0, 200.0, 300.0])
    assert abs(psi_from_counts(counts, counts)) < 1e-9
