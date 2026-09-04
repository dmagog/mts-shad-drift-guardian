"""Сквозные тесты оркестратора и выходного контракта."""
import numpy as np
import pandas as pd

from drift_guardian import DriftConfig, analyze

EXPECTED_KEYS = {
    "overall_severity",
    "recommendation",
    "alerts",
    "schema",
    "data_quality",
    "columns",
    "adversarial",
    "meta",
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
