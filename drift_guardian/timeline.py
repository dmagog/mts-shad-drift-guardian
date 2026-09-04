"""Мониторинг во времени: серия батчей против одного эталона.

Продакшн-мониторинг — это не два снимка, а поток: батч за днём, неделей, месяцем.
Модуль режет поток по колонке даты, прогоняет каждый период через ``DriftGuardian``
и собирает компактную сводку: статус периода, PSI по колонкам, число критичных
признаков, ROC-AUC adversarial validation — так видны тренды и момент, когда
дрейф начался.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .config import DriftConfig
from .contracts import DriftReport, Severity
from .guardian import DriftGuardian

FREQ_LABELS = {"D": "день", "W": "неделя", "M": "месяц", "Q": "квартал"}


@dataclass
class PeriodSummary:
    """Сводка одного периода потока."""

    label: str
    rows: int
    overall_severity: Severity
    recommendation: str
    n_critical: int
    n_warning: int
    target_severity: Severity | None
    adversarial_auc: float | None
    alerts: list[str] = field(default_factory=list)
    psi_by_column: dict[str, float] = field(default_factory=dict)
    severity_by_column: dict[str, Severity] = field(default_factory=dict)


def summarize_period(label: str, report: DriftReport, rows: int) -> PeriodSummary:
    psi: dict[str, float] = {}
    severity: dict[str, Severity] = {}
    for col in report.columns:
        severity[col.column] = col.severity
        psi_test = next((t for t in col.tests if t.name == "psi"), None)
        psi[col.column] = float(psi_test.statistic) if psi_test else 0.0
    return PeriodSummary(
        label=label,
        rows=rows,
        overall_severity=report.overall_severity,
        recommendation=report.recommendation,
        n_critical=sum(s == "critical" for s in severity.values()),
        n_warning=sum(s == "warning" for s in severity.values()),
        target_severity=report.target_drift.severity if report.target_drift else None,
        adversarial_auc=report.adversarial.roc_auc if report.adversarial else None,
        alerts=list(report.alerts),
        psi_by_column=psi,
        severity_by_column=severity,
    )


def split_by_period(
    frame: pd.DataFrame, date_column: str, freq: str = "M"
) -> list[tuple[str, pd.DataFrame]]:
    """Режет поток на батчи по календарному периоду (D, W, M, Q); колонка даты убирается."""
    if date_column not in frame.columns:
        raise KeyError(f"Колонка даты '{date_column}' не найдена")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    mask = dates.notna()
    if not mask.any():
        raise ValueError(f"В колонке '{date_column}' нет распознаваемых дат")
    valid = frame.loc[mask].drop(columns=[date_column])
    periods = dates.loc[mask].dt.to_period(freq)
    return [(str(period), valid.loc[periods == period]) for period in sorted(periods.unique())]


def run_timeline(
    reference: pd.DataFrame,
    batches: list[tuple[str, pd.DataFrame]],
    config: DriftConfig | None = None,
) -> dict:
    """Прогоняет каждый батч против эталона и собирает сводку по периодам."""
    guardian = DriftGuardian(config)
    summaries = [
        summarize_period(label, guardian.run(reference, batch), len(batch))
        for label, batch in batches
    ]
    columns: list[str] = []
    for summary in summaries:
        for column in summary.psi_by_column:
            if column not in columns:
                columns.append(column)
    return {
        "reference_rows": int(len(reference)),
        "periods": [asdict(s) for s in summaries],
        "columns": columns,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": guardian.config.to_dict(),
        },
    }


def run_timeline_from_frame(
    reference: pd.DataFrame,
    stream: pd.DataFrame,
    date_column: str,
    freq: str = "M",
    config: DriftConfig | None = None,
) -> dict:
    """Удобная обёртка: поток с колонкой даты → сводка по периодам."""
    return run_timeline(reference, split_by_period(stream, date_column, freq), config)
