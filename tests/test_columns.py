"""Тесты типизации и автодетекции ролей колонок."""
import numpy as np
import pandas as pd

from drift_guardian import DriftConfig, analyze
from drift_guardian.columns import split_columns


def _frames(n=500, seed=0):
    rng = np.random.default_rng(seed)
    make = lambda size, start: pd.DataFrame(  # noqa: E731
        {
            "customer_id": np.arange(start, start + size),
            "order_uuid": [f"id-{start + i:08d}" for i in range(size)],
            "signup_date": pd.date_range("2026-01-01", periods=size).strftime("%Y-%m-%d"),
            "event_ts": pd.date_range("2026-01-01", periods=size, freq="h"),
            "age": rng.normal(40, 10, size),
            "num_children": rng.integers(0, 4, size),
            "is_active": rng.random(size) < 0.5,
            "city": rng.choice(["msk", "spb"], size),
        }
    )
    return make(n, 0), make(n, n)


def test_identifiers_and_dates_are_skipped_with_reasons():
    ref, cur = _frames()
    numeric, categorical, skipped = split_columns(ref, cur, DriftConfig())
    assert numeric == ["age"]
    assert set(categorical) == {"num_children", "is_active", "city"}
    assert set(skipped) == {"customer_id", "order_uuid", "signup_date", "event_ts"}
    assert "уникальны" in skipped["customer_id"]
    assert "дат" in skipped["signup_date"]


def test_explicit_overrides_beat_heuristics():
    ref, cur = _frames()
    config = DriftConfig(numeric_columns=["customer_id"], categorical_columns=["age"])
    numeric, categorical, skipped = split_columns(ref, cur, config)
    assert "customer_id" in numeric and "age" in categorical
    assert "customer_id" not in skipped


def test_skipped_columns_appear_in_report_as_info():
    ref, cur = _frames()
    report = analyze(ref, cur, DriftConfig(adversarial_enabled=False))
    assert report["meta"]["skipped_reasons"]["customer_id"]
    info = [i for i in report["schema"] if i["check"] == "column_skipped"]
    assert {i["column"] for i in info} == {"customer_id", "order_uuid", "signup_date", "event_ts"}
    assert all(i["severity"] == "ok" for i in info)
    assert report["alerts"] == []
