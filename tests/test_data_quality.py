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
