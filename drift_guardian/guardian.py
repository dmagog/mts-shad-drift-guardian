"""Оркестратор Data Drift Guardian: единая точка входа для полного анализа."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .adversarial import adversarial_validation
from .columns import split_columns
from .config import DriftConfig
from .contracts import (
    AdversarialReport,
    ColumnReport,
    DriftReport,
    Issue,
    worst,
)
from .data_quality import check_data_quality
from .drift_engines import analyze_categorical_column, analyze_numeric_column
from .schema import validate_schema

_RECOMMENDATIONS = {
    "ok": "Дрейф не обнаружен — модель можно эксплуатировать в штатном режиме.",
    "warning": "Обнаружены умеренные сдвиги — рекомендуется усиленный мониторинг.",
    "critical": "Зафиксирован критический дрейф — рекомендуется переобучение модели.",
}

_ALERT_PREFIX = {"critical": "КРИТИЧНО", "warning": "ВНИМАНИЕ"}


class DriftGuardian:
    """Главный класс системы: ``run(reference, current)`` -> ``DriftReport``."""

    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()

    def run(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        cfg = self.config
        started_at = datetime.now(timezone.utc)

        schema_issues = validate_schema(reference, current)
        if any(i.check in ("empty_reference", "empty_current") for i in schema_issues):
            return self._aborted_report(schema_issues, reference, current, started_at)

        numeric_cols, categorical_cols, skipped_cols = split_columns(
            reference, current, cfg
        )
        # Колонки со сменой типа исключаем из статанализа —
        # по ним уже есть critical-алерт схемы.
        broken = {i.column for i in schema_issues if i.check == "dtype_mismatch"}
        numeric_cols = [c for c in numeric_cols if c not in broken]
        categorical_cols = [c for c in categorical_cols if c not in broken]

        quality_issues = check_data_quality(
            reference, current, numeric_cols, categorical_cols, cfg
        )

        n_pvalue_tests = len(numeric_cols) + len(categorical_cols)
        alpha_effective = (
            cfg.thresholds.alpha / n_pvalue_tests
            if cfg.bonferroni and n_pvalue_tests
            else cfg.thresholds.alpha
        )

        column_reports: list[ColumnReport] = [
            analyze_numeric_column(c, reference, current, cfg, alpha_effective)
            for c in numeric_cols
        ]
        column_reports += [
            analyze_categorical_column(c, reference, current, cfg, alpha_effective)
            for c in categorical_cols
        ]

        adversarial = adversarial_validation(
            reference, current, [*numeric_cols, *categorical_cols], cfg
        )

        overall = worst(
            [
                *(i.severity for i in schema_issues),
                *(i.severity for i in quality_issues),
                *(c.severity for c in column_reports),
                *([adversarial.severity] if adversarial else []),
            ]
        )
        alerts = self._build_alerts(
            schema_issues, quality_issues, column_reports, adversarial
        )

        meta = {
            "generated_at": started_at.isoformat(timespec="seconds"),
            "reference_rows": int(len(reference)),
            "current_rows": int(len(current)),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "skipped_columns": skipped_cols,
            "alpha_effective": alpha_effective,
            "config": cfg.to_dict(),
        }
        return DriftReport(
            overall_severity=overall,
            recommendation=_RECOMMENDATIONS[overall],
            alerts=alerts,
            schema=schema_issues,
            data_quality=quality_issues,
            columns=column_reports,
            adversarial=adversarial,
            meta=meta,
        )

    @staticmethod
    def _build_alerts(
        schema_issues: list[Issue],
        quality_issues: list[Issue],
        column_reports: list[ColumnReport],
        adversarial: AdversarialReport | None,
    ) -> list[str]:
        alerts: list[str] = []
        for issue in (*schema_issues, *quality_issues):
            if issue.severity != "ok":
                alerts.append(f"{_ALERT_PREFIX[issue.severity]}: {issue.message}")
        for col in column_reports:
            if col.severity == "ok":
                continue
            triggered = [t for t in col.tests if t.severity != "ok"]
            parts = ", ".join(
                f"{t.name}={t.statistic:.3g}"
                + (f" (p={t.p_value:.2g})" if t.p_value is not None else "")
                for t in triggered
            )
            alerts.append(
                f"{_ALERT_PREFIX[col.severity]}: дрейф по признаку '{col.column}' — {parts}."
            )
        if adversarial and adversarial.severity != "ok":
            top = ", ".join(f["feature"] for f in adversarial.top_features[:3])
            alerts.append(
                f"{_ALERT_PREFIX[adversarial.severity]}: adversarial validation различает "
                f"выборки (ROC-AUC={adversarial.roc_auc:.3f}); сильнее всего изменились: {top}."
            )
        return alerts

    def _aborted_report(
        self,
        schema_issues: list[Issue],
        reference: pd.DataFrame,
        current: pd.DataFrame,
        started_at: datetime,
    ) -> DriftReport:
        return DriftReport(
            overall_severity="critical",
            recommendation="Анализ невозможен: проверьте входные данные.",
            alerts=[f"КРИТИЧНО: {i.message}" for i in schema_issues],
            schema=schema_issues,
            data_quality=[],
            columns=[],
            adversarial=None,
            meta={
                "generated_at": started_at.isoformat(timespec="seconds"),
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
                "config": self.config.to_dict(),
            },
        )


def analyze(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig | None = None,
) -> dict:
    """Точка входа по контракту ТЗ.

    На входе — два ``pandas.DataFrame`` (Reference и Current), на выходе —
    структурированный словарь с метриками и флагами алертов.
    """
    return DriftGuardian(config).run(reference, current).to_dict()
