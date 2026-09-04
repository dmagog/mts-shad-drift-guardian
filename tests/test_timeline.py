"""Тесты режима временного ряда."""
import pandas as pd

from drift_guardian import DriftConfig, run_timeline, run_timeline_from_frame, split_by_period
from drift_guardian.demo import make_timeline_demo


def test_split_by_period_monthly():
    _, stream = make_timeline_demo(n_periods=3, rows_per_period=200, seed=1)
    batches = split_by_period(stream, "date", "M")
    assert [label for label, _ in batches] == ["2026-01", "2026-02", "2026-03"]
    assert all(len(batch) == 200 and "date" not in batch.columns for _, batch in batches)


def test_split_by_period_handles_strings_and_bad_values():
    frame = pd.DataFrame({"dt": ["2026-01-05", "2026-01-20", "мусор", "2026-02-01"], "x": [1, 2, 3, 4]})
    batches = split_by_period(frame, "dt", "M")
    assert [label for label, _ in batches] == ["2026-01", "2026-02"]
    assert len(batches[0][1]) == 2


def test_timeline_detects_growing_drift():
    reference, stream = make_timeline_demo(n_periods=8, rows_per_period=1500, seed=2)
    config = DriftConfig(target_column="target", adversarial_enabled=False)
    timeline = run_timeline_from_frame(reference, stream, "date", "M", config)
    periods = timeline["periods"]
    assert len(periods) == 8
    assert periods[0]["overall_severity"] == "ok"
    assert periods[-1]["overall_severity"] == "critical"
    ages = [p["psi_by_column"]["age"] for p in periods]
    assert ages[-1] > ages[0]
    assert periods[-1]["target_severity"] in ("warning", "critical")
    assert "age" in timeline["columns"]


def test_run_timeline_accepts_explicit_batches():
    reference, stream = make_timeline_demo(n_periods=2, rows_per_period=300, seed=3)
    batches = split_by_period(stream, "date", "M")
    timeline = run_timeline(reference, batches, DriftConfig(adversarial_enabled=False))
    assert [p["label"] for p in timeline["periods"]] == ["2026-01", "2026-02"]
    assert timeline["reference_rows"] == 20_000


def test_unparseable_dates_are_counted():
    _, stream = make_timeline_demo(n_periods=2, rows_per_period=300, seed=5)
    stream["date"] = stream["date"].astype(str)
    stream.loc[:49, "date"] = "мусор"
    reference, _ = make_timeline_demo(n_periods=1, rows_per_period=100, seed=5)
    timeline = run_timeline_from_frame(reference, stream, "date", "M", DriftConfig(adversarial_enabled=False))
    assert timeline["meta"]["dropped_rows"] == 50
    assert timeline["meta"]["total_rows"] == 600


def test_tiny_period_is_marked_insufficient():
    reference, stream = make_timeline_demo(n_periods=2, rows_per_period=300, seed=6)
    stream = pd.concat([stream[stream["date"] < "2026-02-01"], stream[stream["date"] >= "2026-02-01"].head(12)])
    timeline = run_timeline_from_frame(reference, stream, "date", "M", DriftConfig(adversarial_enabled=False))
    tiny = timeline["periods"][-1]
    assert tiny["rows"] == 12
    assert tiny["insufficient"] is True
    assert tiny["overall_severity"] == "warning"
