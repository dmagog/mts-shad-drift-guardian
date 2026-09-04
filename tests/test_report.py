"""Тесты графиков, HTML-отчёта и JSON-экспорта."""
import json

import plotly.graph_objects as go

from drift_guardian import analyze
from drift_guardian.demo import SCENARIOS, make_demo
from drift_guardian.html_report import render_html_report, report_to_json, save_html_report
from drift_guardian.plots import (
    categorical_distribution_figure,
    column_summary_frame,
    feature_importance_figure,
    issues_frame,
    numeric_distribution_figure,
    style_severity,
)


def _demo():
    return make_demo("mixed", ref_rows=3000, cur_rows=1000, seed=1)


def test_demo_is_reproducible():
    a, b = make_demo("mixed", 500, 200, seed=7), make_demo("mixed", 500, 200, seed=7)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert set(SCENARIOS) == {
        "no_drift", "mean_shift", "missing_surge", "new_category", "concept_drift", "mixed"
    }


def test_numeric_figure_has_two_series():
    ref, cur = _demo()
    fig = numeric_distribution_figure(ref["age"], cur["age"], title="age")
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["Эталон", "Текущий батч"]


def test_categorical_figure_shows_new_category():
    ref, cur = _demo()
    fig = categorical_distribution_figure(ref["channel"], cur["channel"])
    assert len(fig.data) == 2
    assert "партнёрская сеть" in list(fig.data[0].x)


def test_feature_importance_figure_orders_features():
    fig = feature_importance_figure(
        [{"feature": "a", "importance": 0.7}, {"feature": "b", "importance": 0.3}]
    )
    assert list(fig.data[0].y) == ["b", "a"]  # самый важный сверху


def test_summary_frame_and_styling():
    ref, cur = _demo()
    report = analyze(ref, cur)
    frame = column_summary_frame(report)
    assert len(frame) == len(report["columns"])
    assert frame.iloc[0]["статус"] == "критично"  # худшие сверху
    style_severity(frame).to_html()  # не падает


def test_issues_frame_columns():
    frame = issues_frame(
        [{"check": "x", "severity": "warning", "message": "m", "column": None, "value": None}]
    )
    assert list(frame.columns) == ["проверка", "признак", "статус", "сообщение"]
    assert frame.iloc[0]["признак"] == "—"


def test_html_report_contains_alerts_and_plots(tmp_path):
    ref, cur = _demo()
    report = analyze(ref, cur)
    html = render_html_report(report, ref, cur, plotlyjs="cdn")
    assert "<html" in html and "КРИТИЧНО" in html and "plotly" in html.lower()
    assert "Что изменилось" in html and "dg-chip" in html
    for column in report["meta"]["numeric_columns"]:
        assert column in html
    path = save_html_report(tmp_path / "report.html", report, ref, cur, plotlyjs="cdn")
    assert path.exists() and path.stat().st_size > 10_000


def test_html_report_has_target_block_and_concept_note():
    from drift_guardian import DriftConfig

    ref, cur = make_demo("concept_drift", ref_rows=3000, cur_rows=1500, seed=3)
    report = analyze(ref, cur, DriftConfig(target_column="target"))
    html = render_html_report(report, ref, cur, plotlyjs="cdn")
    assert "Целевая переменная" in html
    assert "концептуальный дрейф" in html


def test_json_roundtrip():
    ref, cur = _demo()
    report = analyze(ref, cur)
    data = json.loads(report_to_json(report))
    assert data["overall_severity"] == report["overall_severity"]
    assert len(data["columns"]) == len(report["columns"])


def test_timeline_figures_and_html():
    from drift_guardian import DriftConfig, run_timeline_from_frame
    from drift_guardian.demo import make_timeline_demo
    from drift_guardian.html_report import render_timeline_html
    from drift_guardian.plots import timeline_frame, timeline_psi_figure, timeline_severity_figure

    reference, stream = make_timeline_demo(n_periods=3, rows_per_period=300, seed=4)
    timeline = run_timeline_from_frame(
        reference, stream, "date", "M", DriftConfig(adversarial_enabled=False)
    )
    assert len(timeline_psi_figure(timeline).data) >= 1
    assert len(timeline_severity_figure(timeline).data[0].x) == 3
    assert list(timeline_frame(timeline)["период"]) == ["2026-01", "2026-02", "2026-03"]
    html = render_timeline_html(timeline, plotlyjs="cdn")
    assert "2026-03" in html and "Статус по периодам" in html
