"""Тесты загрузчика таблиц и командного интерфейса."""
import json

import pytest

from drift_guardian.cli import main
from drift_guardian.demo import make_demo
from drift_guardian.io import load_table


def test_load_csv_and_parquet(tmp_path):
    reference, _ = make_demo("no_drift", 300, 100, seed=2)
    csv_path, pq_path = tmp_path / "ref.csv", tmp_path / "ref.parquet"
    reference.to_csv(csv_path, index=False)
    reference.to_parquet(pq_path, index=False)
    assert load_table(csv_path).shape == reference.shape
    assert load_table(pq_path).shape == reference.shape


def test_load_table_rejects_unknown_format(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        load_table(path)


def test_cli_writes_reports_and_returns_severity_code(tmp_path, capsys):
    reference, current = make_demo("mixed", 2000, 800, seed=5)
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    json_path, html_path = tmp_path / "out" / "report.json", tmp_path / "out" / "report.html"

    code = main([
        "-r", str(ref_path), "-c", str(cur_path), "--target", "target",
        "--exclude", "num_dependents", "--json", str(json_path), "--html", str(html_path),
    ])

    assert code == 2  # critical
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["overall_severity"] == "critical"
    assert report["target_drift"]["column"] == "target"
    assert "num_dependents" in report["meta"]["excluded_columns"]
    assert html_path.stat().st_size > 10_000
    assert "[CRITICAL]" in capsys.readouterr().out


def test_cli_no_drift_returns_zero(tmp_path):
    reference, current = make_demo("no_drift", 2000, 800, seed=6)
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    reference.to_csv(ref_path, index=False)
    current.to_csv(cur_path, index=False)
    assert main(["-r", str(ref_path), "-c", str(cur_path), "--target", "target", "-q"]) == 0


def test_cli_timeline_mode(tmp_path):
    from drift_guardian.demo import make_timeline_demo

    reference, stream = make_timeline_demo(n_periods=3, rows_per_period=400, seed=8)
    ref_path, stream_path = tmp_path / "ref.csv", tmp_path / "stream.csv"
    reference.to_csv(ref_path, index=False)
    stream.to_csv(stream_path, index=False)
    json_path, html_path = tmp_path / "timeline.json", tmp_path / "timeline.html"
    code = main([
        "-r", str(ref_path), "-c", str(stream_path), "--date-column", "date", "--freq", "M",
        "--target", "target", "--no-adversarial", "--json", str(json_path), "--html", str(html_path),
    ])
    assert code in (0, 1, 2)
    timeline = json.loads(json_path.read_text(encoding="utf-8"))
    assert [p["label"] for p in timeline["periods"]] == ["2026-01", "2026-02", "2026-03"]
    assert "Статус по периодам" in html_path.read_text(encoding="utf-8")
