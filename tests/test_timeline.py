"""Тесты режима временного ряда."""
import numpy as np
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


def _stream_with_jump(n_periods: int = 6, rows: int = 1500, jump_at: int = 3, seed: int = 11):
    """Поток, где с периода ``jump_at`` возраст резко сдвигается и дальше не меняется."""
    from drift_guardian.demo import make_base

    frames = []
    for i in range(n_periods):
        rng = np.random.default_rng([seed, i])
        batch = make_base(rng, rows)
        if i >= jump_at:
            batch["age"] = np.clip(batch["age"] + 8, 18, 85)
        batch.insert(0, "date", pd.Timestamp("2026-01-01") + pd.DateOffset(months=i))
        frames.append(batch)
    return pd.concat(frames, ignore_index=True)


def test_compare_previous_separates_jump_from_accumulated_drift():
    reference, _ = make_timeline_demo(n_periods=1, rows_per_period=100, seed=11)
    stream = _stream_with_jump()
    config = DriftConfig(target_column="target", adversarial_enabled=False)
    timeline = run_timeline_from_frame(reference, stream, "date", "M", config, compare_previous=True)
    periods = timeline["periods"]
    assert timeline["meta"]["compare_previous"] is True
    assert periods[0]["vs_previous"] is None
    # относительно эталона: все периоды после скачка критичны (накопленный дрейф)
    assert [p["overall_severity"] for p in periods[3:]] == ["critical"] * 3
    # относительно предыдущего периода: критичен только период скачка
    jumps = [p["vs_previous"]["overall_severity"] for p in periods[1:]]
    assert jumps[2] == "critical"
    assert jumps[3] == "ok" and jumps[4] == "ok"


def test_compare_previous_off_by_default():
    reference, stream = make_timeline_demo(n_periods=2, rows_per_period=200, seed=12)
    timeline = run_timeline_from_frame(reference, stream, "date", "M", DriftConfig(adversarial_enabled=False))
    assert all(p["vs_previous"] is None for p in timeline["periods"])
    assert timeline["meta"]["compare_previous"] is False


def test_compare_previous_with_tiny_previous_period_is_insufficient_not_critical():
    reference, stream = make_timeline_demo(n_periods=3, rows_per_period=800, seed=13)
    first = stream[stream["date"] < "2026-02-01"].head(10)
    stream = pd.concat([first, stream[stream["date"] >= "2026-02-01"]])
    timeline = run_timeline_from_frame(
        reference, stream, "date", "M", DriftConfig(target_column="target", adversarial_enabled=False),
        compare_previous=True,
    )
    second = timeline["periods"][1]
    assert second["overall_severity"] == "ok"  # относительно эталона всё в порядке
    assert second["vs_previous"]["insufficient"] is True
    assert second["vs_previous"]["overall_severity"] == "warning"


def test_timeline_skips_segments_per_period():
    import time

    reference, stream = make_timeline_demo(n_periods=4, rows_per_period=800, seed=14)
    plain = DriftConfig(target_column="target", adversarial_enabled=False)
    with_segments = DriftConfig(target_column="target", segment_column="region", adversarial_enabled=False)
    started = time.perf_counter()
    run_timeline_from_frame(reference, stream, "date", "M", plain)
    base = time.perf_counter() - started
    started = time.perf_counter()
    timeline = run_timeline_from_frame(reference, stream, "date", "M", with_segments)
    with_seg = time.perf_counter() - started
    assert timeline["meta"]["config"]["segment_column"] is None
    assert with_seg < base * 2.5  # без отключения сегментов было бы в 3–5 раз дольше
