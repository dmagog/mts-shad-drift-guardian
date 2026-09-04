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


def test_psi_new_category_is_smoothed():
    ref_counts = np.array([600.0, 300.0, 100.0, 0.0])
    one_percent = np.array([594.0, 297.0, 99.0, 10.0])
    five_percent = np.array([570.0, 285.0, 95.0, 50.0])
    assert psi_from_counts(ref_counts, one_percent) < 0.1   # 1 % новой категории — ещё ok
    assert psi_from_counts(ref_counts, five_percent) > 0.2  # 5 % — уже critical


def test_wasserstein_thresholds_aligned_with_psi():
    """Сдвиг на 0.2σ не должен давать алерт ни по одной метрике эффекта."""
    rng = np.random.default_rng(5)
    ref, cur = _frames(rng.normal(0, 1, 20000), rng.normal(0.2, 1, 20000))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "ok"


def test_small_batch_without_drift_is_ok_and_marked_underpowered():
    rng = np.random.default_rng(8)
    ref, cur = _frames(rng.normal(0, 1, 20000), rng.normal(0, 1, 150))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "ok"
    psi_test = next(t for t in report.tests if t.name == "psi")
    assert psi_test.details["underpowered"] is True
    assert psi_test.details["noise_floor"] > 0.1


def test_small_batch_with_strong_shift_still_critical():
    rng = np.random.default_rng(9)
    ref, cur = _frames(rng.normal(0, 1, 20000), rng.normal(1.5, 1, 150))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "critical"


def test_noise_floor_decreases_with_sample_size():
    from drift_guardian.drift_engines import bootstrap_noise, reference_bin_edges

    rng = np.random.default_rng(10)
    ref = rng.normal(0, 1, 20000)
    edges = reference_bin_edges(ref, 10)
    ref_counts, _ = np.histogram(ref, bins=edges)
    floors = []
    for n in (100, 500, 5000):
        cur_counts, _ = np.histogram(rng.normal(0, 1, n), bins=edges)
        floors.append(bootstrap_noise(ref_counts, cur_counts, np.random.default_rng(0))["psi_noise"])
    assert floors[0] > floors[1] > floors[2]
    assert floors[2] < 0.01


def test_guard_can_be_disabled():
    rng = np.random.default_rng(11)
    ref, cur = _frames(rng.normal(0, 1, 20000), rng.normal(0, 1, 150))
    config = DriftConfig(sample_size_guard=False)
    report = analyze_numeric_column("x", ref, cur, config, alpha_effective=0.05)
    psi_test = next(t for t in report.tests if t.name == "psi")
    assert "underpowered" not in psi_test.details


def test_infinite_values_do_not_break_numeric_tests():
    rng = np.random.default_rng(12)
    ref, cur = _frames(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000))
    cur.loc[:20, "x"] = np.inf
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    wasserstein = next(t for t in report.tests if t.name == "wasserstein_norm")
    assert np.isfinite(wasserstein.statistic)
    assert report.severity == "ok"


def test_near_constant_reference_uses_pooled_scale():
    rng = np.random.default_rng(0)
    ref, cur = _frames(np.full(2000, 1e-6) + rng.normal(0, 1e-15, 2000), rng.uniform(0, 5, 2000))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    w = next(t for t in report.tests if t.name == "wasserstein_norm")
    assert w.details["scale"] == "pooled" and 0.5 < w.statistic < 5
    assert report.severity == "critical"


def test_changed_constant_is_detected():
    ref, cur = _frames(np.full(500, 1.0), np.full(500, 2.0))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "critical"


def test_identical_constants_are_not_drift():
    ref, cur = _frames(np.full(500, 1.0), np.full(500, 1.0))
    report = analyze_numeric_column("x", ref, cur, DriftConfig(), alpha_effective=0.05)
    assert report.severity == "ok"
