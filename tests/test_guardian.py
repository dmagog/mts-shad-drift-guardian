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


def test_identical_frames_are_not_flagged_as_drift():
    """Строки-двойники не должны «подсказывать» adversarial-модели метку."""
    reference, _ = make_demo("no_drift", 3000, 1000, seed=30)
    report = analyze(reference, reference, DriftConfig(target_column="target"))
    assert report["overall_severity"] == "ok"
    assert report["adversarial"]["overlap_share"] == 1.0
    assert "идентичны" in report["adversarial"]["note"]


def test_partial_overlap_is_excluded_from_adversarial():
    reference, fresh = make_demo("no_drift", 4000, 1000, seed=31)
    current = pd.concat([reference.sample(3000, random_state=1), fresh], ignore_index=True)
    report = analyze(reference, current, DriftConfig(target_column="target"))
    assert 0.7 <= report["adversarial"]["overlap_share"] <= 0.8
    assert report["adversarial"]["roc_auc"] < 0.6
    assert report["overall_severity"] == "ok"


def test_duplicate_column_names_abort_analysis():
    rng = np.random.default_rng(32)
    ref = _make(rng, 300)
    cur = _make(rng, 300)
    ref.columns = ["age", "age", "city"]
    cur.columns = ["age", "age", "city"]
    report = analyze(ref, cur)
    assert report["overall_severity"] == "critical"
    assert any(i["check"] == "duplicate_columns" for i in report["schema"])
    assert report["columns"] == []


def test_tiny_batch_is_reported_as_insufficient():
    reference, current = make_demo("no_drift", 5000, 200, seed=33)
    report = analyze(reference, current.head(10), DriftConfig(target_column="target"))
    assert report["overall_severity"] == "warning"
    assert report["meta"]["insufficient_data"] is True
    assert "Недостаточно данных" in report["recommendation"]
    assert report["alerts"][0].startswith("ВНИМАНИЕ: батч содержит 10 строк")
    # шум долей на десяти строках не должен превращаться в замечания к качеству
    assert all(i["severity"] == "ok" for i in report["data_quality"])


def test_small_batch_ok_mentions_low_sensitivity():
    reference, current = make_demo("no_drift", 5000, 200, seed=34)
    report = analyze(reference, current.head(30), DriftConfig(target_column="target"))
    assert report["overall_severity"] == "ok"
    assert report["meta"]["insufficient_data"] is False
    assert "мал" in report["recommendation"]


def test_segment_drift_is_localized_and_surfaced():
    reference, current = make_demo("no_drift", 8000, 3000, seed=50)
    current = current.copy()
    mask = current["region"] == "Москва"
    current.loc[mask, "age"] = np.clip(current.loc[mask, "age"] + 10, 18, 85)
    config = DriftConfig(target_column="target", segment_column="region", adversarial_enabled=False)
    report = analyze(reference, current, config)
    by_label = {s["label"]: s for s in report["segments"]}
    assert by_label["Москва"]["overall_severity"] == "critical"
    assert all(s["overall_severity"] == "ok" for label, s in by_label.items() if label != "Москва")
    assert report["overall_severity"] in ("warning", "critical")
    assert any("region=Москва" in alert for alert in report["alerts"])
    assert "region" not in by_label["Москва"]["psi_by_column"]  # внутри сегмента колонка константна


def test_segment_absent_in_current_batch_is_flagged():
    reference, current = make_demo("no_drift", 4000, 2000, seed=51)
    current = current[current["region"] != "Санкт-Петербург"]
    report = analyze(reference, current, DriftConfig(segment_column="region", adversarial_enabled=False))
    spb = next(s for s in report["segments"] if s["label"] == "Санкт-Петербург")
    assert spb["rows"] == 0 and spb["overall_severity"] == "warning" and spb["insufficient"] is True


def test_missing_segment_column_is_reported():
    reference, current = make_demo("no_drift", 1000, 500, seed=52)
    report = analyze(reference, current, DriftConfig(segment_column="nope", adversarial_enabled=False))
    assert report["segments"] is None
    assert any(i["check"] == "segment_column_missing" for i in report["schema"])


def test_column_thresholds_change_only_that_column():
    reference, current = make_demo("mean_shift", 6000, 2000, seed=60)
    base = analyze(reference, current, DriftConfig(target_column="target", adversarial_enabled=False))
    relaxed = analyze(
        reference, current,
        DriftConfig(
            target_column="target", adversarial_enabled=False,
            column_thresholds={"age": {
                "psi_warning": 1.0, "psi_critical": 2.0, "js_warning": 1.0, "js_critical": 1.0,
                "wasserstein_warning": 5.0, "wasserstein_critical": 9.0,
            }},
        ),
    )
    by_col = lambda r: {c["column"]: c["severity"] for c in r["columns"]}  # noqa: E731
    assert by_col(base)["age"] == "critical"
    assert by_col(relaxed)["age"] == "ok"
    assert by_col(base)["income"] == by_col(relaxed)["income"]


def test_tiny_reference_is_insufficient_not_critical():
    reference, current = make_demo("no_drift", 2000, 1500, seed=70)
    report = analyze(reference.head(10), current, DriftConfig(target_column="target"))
    assert report["overall_severity"] == "warning"
    assert report["meta"]["insufficient_data"] is True
    assert report["data_quality"] == [] and report["columns"] == []
    assert "эталон содержит 10 строк" in report["alerts"][-1]
    assert "эталон 10 строк" in report["recommendation"]


def test_segment_truncation_is_reported_and_target_segment_rejected():
    rng = np.random.default_rng(71)
    reference, current = make_demo("no_drift", 3000, 1500, seed=71)
    reference = reference.assign(city=rng.integers(0, 12, len(reference)).astype(str))
    current = current.assign(city=rng.integers(0, 12, len(current)).astype(str))
    report = analyze(reference, current, DriftConfig(segment_column="city", max_segments=5, adversarial_enabled=False))
    assert len(report["segments"]) == 5 and report["meta"]["segment_values_total"] == 12
    rejected = analyze(reference, current, DriftConfig(target_column="target", segment_column="target", adversarial_enabled=False))
    assert rejected["segments"] is None
    assert any(i["check"] == "segment_column_is_target" for i in rejected["schema"])
