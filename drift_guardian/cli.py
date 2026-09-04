"""Командный интерфейс: анализ дрейфа из терминала, cron или CI.

    drift-guardian --reference reference.csv --current batch.parquet \
        --target target --exclude customer_id --json report.json --html report.html

Код возврата: 0 — ok, 1 — warning, 2 — critical (удобно для пайплайнов).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DriftConfig
from .guardian import DriftGuardian
from .html_report import report_to_json, save_html_report
from .io import load_table

EXIT_CODES = {"ok": 0, "warning": 1, "critical": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-guardian",
        description="Data Drift Guardian — детекция дрейфа и контроля качества данных.",
    )
    parser.add_argument("-r", "--reference", required=True, help="эталон (CSV/Parquet)")
    parser.add_argument("-c", "--current", required=True, help="текущий батч (CSV/Parquet)")
    parser.add_argument("--config", help="YAML-конфиг порогов и контракта данных")
    parser.add_argument("--target", help="колонка целевой переменной")
    parser.add_argument("--prediction", help="колонка предсказаний модели")
    parser.add_argument("--exclude", nargs="*", default=None, help="колонки, исключаемые из анализа")
    parser.add_argument("--no-adversarial", action="store_true", help="отключить adversarial validation")
    parser.add_argument("--json", help="куда сохранить отчёт JSON")
    parser.add_argument("--html", help="куда сохранить отчёт HTML")
    parser.add_argument(
        "--plotlyjs", choices=["inline", "cdn"], default="cdn",
        help="встроить plotly.js в HTML (inline, офлайн) или подгружать из сети (cdn)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="не печатать алерты")
    return parser


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
    report = DriftGuardian(config).run(reference, current).to_dict()

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_to_json(report), encoding="utf-8")
    if args.html:
        save_html_report(args.html, report, reference, current, plotlyjs=args.plotlyjs)

    severity = report["overall_severity"]
    print(f"[{severity.upper()}] {report['recommendation']}")
    if not args.quiet:
        for alert in report["alerts"]:
            print(f"  - {alert}")
        if args.json:
            print(f"JSON: {args.json}")
        if args.html:
            print(f"HTML: {args.html}")
    return EXIT_CODES[severity]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
