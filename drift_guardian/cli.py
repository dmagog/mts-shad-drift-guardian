"""Командный интерфейс: анализ дрейфа из терминала, cron или CI.

Два батча:
    drift-guardian --reference reference.csv --current batch.parquet \\
        --target target --exclude customer_id --json report.json --html report.html

Поток во времени (батчи по календарным периодам колонки даты):
    drift-guardian --reference reference.csv --current stream.csv \\
        --date-column date --freq M --target target --html timeline.html

Код возврата: 0 — ok, 1 — warning, 2 — critical (для потока — худший период).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DriftConfig
from .contracts import worst
from .guardian import DriftGuardian
from .html_report import report_to_json, save_html_report, save_timeline_html
from .io import load_table
from .timeline import FREQ_LABELS, run_timeline_from_frame

EXIT_CODES = {"ok": 0, "warning": 1, "critical": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-guardian",
        description="Data Drift Guardian — детекция дрейфа и контроля качества данных.",
    )
    parser.add_argument("-r", "--reference", required=True, help="эталон (CSV/Parquet)")
    parser.add_argument("-c", "--current", required=True, help="текущий батч или поток (CSV/Parquet)")
    parser.add_argument("--config", help="YAML-конфиг порогов и контракта данных")
    parser.add_argument("--target", help="колонка целевой переменной")
    parser.add_argument("--prediction", help="колонка предсказаний модели")
    parser.add_argument("--exclude", nargs="*", default=None, help="колонки, исключаемые из анализа")
    parser.add_argument("--no-adversarial", action="store_true", help="отключить adversarial validation")
    parser.add_argument("--date-column", help="режим потока: колонка даты для разбиения на периоды")
    parser.add_argument("--freq", default="M", choices=sorted(FREQ_LABELS), help="период потока: D, W, M, Q")
    parser.add_argument("--json", help="куда сохранить отчёт JSON")
    parser.add_argument("--html", help="куда сохранить отчёт HTML")
    parser.add_argument(
        "--plotlyjs", choices=["inline", "cdn"], default="cdn",
        help="встроить plotly.js в HTML (inline, офлайн) или подгружать из сети (cdn)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="не печатать алерты")
    return parser


def _write_json(path: str, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")


def _run_two_batches(args, config, reference, current) -> str:
    report = DriftGuardian(config).run(reference, current).to_dict()
    if args.json:
        _write_json(args.json, report_to_json(report))
    if args.html:
        save_html_report(args.html, report, reference, current, plotlyjs=args.plotlyjs)
    severity = report["overall_severity"]
    print(f"[{severity.upper()}] {report['recommendation']}")
    if not args.quiet:
        for alert in report["alerts"]:
            print(f"  - {alert}")
    return severity


def _run_timeline(args, config, reference, stream) -> str:
    timeline = run_timeline_from_frame(reference, stream, args.date_column, args.freq, config)
    if args.json:
        _write_json(args.json, report_to_json(timeline))
    if args.html:
        save_timeline_html(args.html, timeline, plotlyjs=args.plotlyjs)
    periods = timeline["periods"]
    severity = worst(p["overall_severity"] for p in periods)
    print(f"[{severity.upper()}] периодов: {len(periods)} ({FREQ_LABELS[args.freq]}), худший статус — {severity}")
    if not args.quiet:
        for p in periods:
            auc = f"{p['adversarial_auc']:.3f}" if p["adversarial_auc"] is not None else "—"
            print(
                f"  {p['label']}: {p['overall_severity']:<8} critical={p['n_critical']} "
                f"warning={p['n_warning']} adversarial={auc} алертов={len(p['alerts'])}"
            )
    return severity


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DriftConfig.from_yaml(args.config) if args.config else DriftConfig()
    if args.target:
        config.target_column = args.target
    if args.prediction:
        config.prediction_column = args.prediction
    if args.exclude is not None:
        config.exclude_columns = list(args.exclude)
    if args.no_adversarial:
        config.adversarial_enabled = False

    reference = load_table(args.reference)
    current = load_table(args.current)
    if args.date_column:
        severity = _run_timeline(args, config, reference, current)
    else:
        severity = _run_two_batches(args, config, reference, current)
    if not args.quiet:
        if args.json:
            print(f"JSON: {args.json}")
        if args.html:
            print(f"HTML: {args.html}")
    return EXIT_CODES[severity]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
