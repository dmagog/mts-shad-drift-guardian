"""Компактная сводка одного анализа: используется для периодов потока и для сегментов."""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import DriftReport, Severity


@dataclass
class PeriodSummary:
    """Сводка одного периода потока или одного сегмента."""

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
    insufficient: bool = False
    # Та же сводка относительно предыдущего периода (None для первого периода
    # и если сравнение не запрашивалось).
    vs_previous: dict | None = None


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
        insufficient=bool(report.meta.get("insufficient_data", False)),
    )
