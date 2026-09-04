"""Тесты модуля Data Quality."""
import numpy as np
import pandas as pd

from drift_guardian.config import DriftConfig
from drift_guardian.data_quality import check_data_quality


def _issues_by_check(issues):
    return {i.check: i for i in issues}


def test_missing_surge_detected():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    cur.loc[:199, "x"] = np.nan  # 20% пропусков против 0% в эталоне
    issues = check_data_quality(ref, cur, ["x"], [], DriftConfig())
    found = _issues_by_check(issues)
    assert "missing_values" in found
    assert found["missing_values"].severity == "critical"


def test_new_category_detected():
    ref = pd.DataFrame({"color": ["a"] * 500 + ["b"] * 500})
    cur = pd.DataFrame({"color": ["a"] * 450 + ["b"] * 450 + ["c"] * 100})
    issues = check_data_quality(ref, cur, [], ["color"], DriftConfig())
    found = _issues_by_check(issues)
    assert "new_categories" in found
    assert found["new_categories"].severity == "critical"
    assert "'color'" in found["new_categories"].message


def test_out_of_range_detected():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.uniform(0, 1, 1000)})
    cur = pd.DataFrame({"x": rng.uniform(0, 1, 1000)})
    cur.loc[:99, "x"] = 5.0  # 10% значений за пределами эталонного диапазона
    issues = check_data_quality(ref, cur, ["x"], [], DriftConfig())
    found = _issues_by_check(issues)
    assert "out_of_range" in found
    assert found["out_of_range"].severity == "critical"


def test_constant_column_detected():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 100)})
    cur = pd.DataFrame({"x": np.ones(100)})
    issues = check_data_quality(ref, cur, ["x"], [], DriftConfig())
    assert any(i.check == "constant_column" for i in issues)


def test_clean_data_has_no_issues():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame(
        {"x": rng.normal(0, 1, 1000), "c": rng.choice(list("ab"), 1000)}
    )
    cur = pd.DataFrame(
        {"x": rng.normal(0, 1, 1000), "c": rng.choice(list("ab"), 1000)}
    )
    issues = check_data_quality(ref, cur, ["x"], ["c"], DriftConfig())
    assert issues == []


def test_missing_noise_on_tiny_batch_is_not_flagged():
    rng = np.random.default_rng(40)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 20000)})
    ref.loc[rng.random(20000) < 0.08, "x"] = np.nan
    cur = pd.DataFrame({"x": rng.normal(0, 1, 10)})
    cur.loc[:1, "x"] = np.nan  # 20 % пропусков, но всего две ячейки
    issues = check_data_quality(ref, cur, ["x"], [], DriftConfig())
    assert not any(i.check == "missing_values" for i in issues)


def test_single_outlier_in_hundred_rows_is_not_flagged():
    rng = np.random.default_rng(41)
    ref = pd.DataFrame({"x": rng.uniform(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.uniform(0, 1, 100)})
    cur.loc[0, "x"] = 5.0
    issues = check_data_quality(ref, cur, ["x"], [], DriftConfig())
    assert not any(i.check == "out_of_range" for i in issues)


def test_non_finite_values_are_reported():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur.loc[:9, "x"] = np.inf
    issues = check_data_quality(ref, cur, ["x"], [], DriftConfig())
    found = _issues_by_check(issues)
    assert found["non_finite"].severity == "warning"
    assert "10 бесконечных" in found["non_finite"].message


def test_new_categories_hint_when_only_spaces_or_case_differ():
    ref = pd.DataFrame({"c": ["Москва"] * 500 + ["Регионы"] * 500})
    cur = pd.DataFrame({"c": ["Москва "] * 500 + ["регионы"] * 500})
    issues = check_data_quality(ref, cur, [], ["c"], DriftConfig())
    found = _issues_by_check(issues)
    assert found["new_categories"].severity == "critical"
    assert "пробелами или регистром" in found["new_categories"].message


def test_disjoint_ranges_hint_for_time_like_column():
    ref = pd.DataFrame({"t": np.arange(0, 1000, dtype=float)})
    cur = pd.DataFrame({"t": np.arange(1000, 1500, dtype=float)})
    issues = check_data_quality(ref, cur, ["t"], [], DriftConfig())
    hint = next(i for i in issues if i.check == "disjoint_ranges")
    assert hint.severity == "ok" and "исключите" in hint.message
