"""Сквозные тесты оркестратора и выходного контракта."""
import numpy as np
import pandas as pd

from drift_guardian import DriftConfig, analyze
from drift_guardian.demo import make_demo

EXPECTED_KEYS = {
    "overall_severity", "recommendation", "alerts", "schema", "data_quality",
    "columns", "adversarial", "meta", "target_drift", "prediction_drift",
}


def _make(rng, n, drift=False):
    frame = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "income": rng.lognormal(11, 0.5, n),
            "city": rng.choice(["msk", "spb", "other"], n, p=[0.4, 0.2, 0.4]),
        }
    )
    if drift:
        frame["age"] = frame["age"] + 8
        frame.loc[: int(n * 0.1), "city"] = "online"
        frame.loc[rng.random(n) < 0.2, "income"] = np.nan
    return frame


def test_contract_keys_present():
    rng = np.random.default_rng(20)
    report = analyze(_make(rng, 2000), _make(rng, 2000))
    assert isinstance(report, dict)
    assert EXPECTED_KEYS <= set(report.keys())
    assert report["meta"]["reference_rows"] == 2000
    assert report["target_drift"] is None


def test_no_drift_overall_ok():
    rng = np.random.default_rng(21)
    report = analyze(_make(rng, 3000), _make(rng, 3000))
    assert report["overall_severity"] == "ok"
    assert report["alerts"] == []


def test_drift_overall_critical_with_alerts():
    rng = np.random.default_rng(22)
    report = analyze(_make(rng, 3000), _make(rng, 3000, drift=True))
    assert report["overall_severity"] == "critical"
    assert report["alerts"]
    assert "переобучение" in report["recommendation"]
    drifted = {c["column"]: c["severity"] for c in report["columns"]}
    assert drifted["age"] == "critical"


def test_missing_column_is_critical():
    rng = np.random.default_rng(23)
    ref = _make(rng, 1000)
    cur = _make(rng, 1000).drop(columns=["age"])
    report = analyze(ref, cur)
    assert report["overall_severity"] == "critical"
    assert any(i["check"] == "missing_column" for i in report["schema"])


def test_config_override_respected():
    rng = np.random.default_rng(24)
    config = DriftConfig(bonferroni=False)
    report = analyze(_make(rng, 1000), _make(rng, 1000), config)
    assert report["meta"]["alpha_effective"] == config.thresholds.alpha


def test_exclude_columns_are_not_analyzed():
    rng = np.random.default_rng(25)
    report = analyze(
        _make(rng, 1500), _make(rng, 1500, drift=True), DriftConfig(exclude_columns=["age"])
    )
    assert "age" not in [c["column"] for c in report["columns"]]
    assert report["meta"]["excluded_columns"] == ["age"]
    assert all("'age'" not in alert for alert in report["alerts"] if "дрейф по признаку" in alert)


def test_target_drift_is_reported_separately_as_concept_drift():
    reference, current = make_demo("concept_drift", 4000, 2000, seed=3)
    report = analyze(reference, current, DriftConfig(target_column="target"))
    assert report["target_drift"]["column"] == "target"
    assert report["target_drift"]["severity"] == "critical"
    assert "target" not in [c["column"] for c in report["columns"]]
    assert all(c["severity"] != "critical" for c in report["columns"])
    assert report["overall_severity"] == "critical"
    assert "концептуальный" in report["recommendation"]
    assert any("целевой переменной" in alert for alert in report["alerts"])


def test_missing_target_column_gives_warning_not_crash():
    rng = np.random.default_rng(26)
    report = analyze(_make(rng, 500), _make(rng, 500), DriftConfig(target_column="nope"))
    assert report["target_drift"] is None
    assert any(i["check"] == "target_column_missing" for i in report["schema"])


def test_declared_bounds_and_categories_from_contract():
    rng = np.random.default_rng(27)
    ref = pd.DataFrame({"x": rng.uniform(0, 1, 800), "c": rng.choice(list("ab"), 800)})
    cur = pd.DataFrame({"x": rng.uniform(0, 1, 800), "c": rng.choice(list("ab"), 800)})
    config = DriftConfig(value_bounds={"x": (0.0, 0.5)}, allowed_categories={"c": ["a"]})
    report = analyze(ref, cur, config)
    checks = {i["check"]: i for i in report["data_quality"]}
    assert checks["out_of_range"]["severity"] == "critical"
    assert "контракта" in checks["out_of_range"]["message"]
    assert checks["new_categories"]["severity"] == "critical"
