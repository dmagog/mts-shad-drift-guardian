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
    "ok": "Дрейф не обнаружен, модель можно эксплуатировать в штатном режиме.",
    "warning": "Обнаружены умеренные сдвиги, рекомендуется усиленный мониторинг.",
    "critical": "Зафиксирован критический дрейф, рекомендуется переобучение модели.",
}
_CONCEPT_DRIFT_RECOMMENDATION = (
    "Признаки стабильны, но распределение целевой переменной изменилось: вероятен "
    "концептуальный дрейф. Рекомендуется переобучение модели на свежих размеченных данных."
)
_ALERT_PREFIX = {"critical": "КРИТИЧНО", "warning": "ВНИМАНИЕ"}
_SPECIAL_ROLES = {
    "target": "целевой переменной",
    "prediction": "предсказаний модели",
}


def _p_repr(p_value: float) -> str:
    """«p<1e-16» для машинного нуля, иначе «p=0.0023»."""
    return "p<1e-16" if p_value < 1e-16 else f"p={p_value:.2g}"


def _describe_tests(report: ColumnReport) -> str:
    triggered = [t for t in report.tests if t.severity != "ok"]
    return ", ".join(
        f"{t.name}={t.statistic:.3g}"
        + (f" ({_p_repr(t.p_value)})" if t.p_value is not None else "")
        for t in triggered
    )


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

        numeric_cols, categorical_cols, skipped = split_columns(reference, current, cfg)
        skipped_cols = list(skipped)
        kind_of = {**dict.fromkeys(numeric_cols, "numeric"),
                   **dict.fromkeys(categorical_cols, "categorical")}

        # Из анализа признаков убираем: колонки со сменой типа (по ним уже есть
        # critical-алерт схемы), исключённые по контракту, целевую переменную и предсказания.
        broken = {i.column for i in schema_issues if i.check == "dtype_mismatch"}
        excluded = {c for c in (cfg.exclude_columns or []) if c in kind_of}
        special = {c for c in (cfg.target_column, cfg.prediction_column) if c}
        dropped = broken | excluded | special
        numeric_cols = [c for c in numeric_cols if c not in dropped]
        categorical_cols = [c for c in categorical_cols if c not in dropped]

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

        # Отдельные блоки: целевая переменная и предсказания — одна назначенная
        # гипотеза на блок, поэтому без поправки Бонферрони.
        target_drift = self._analyze_special(
            cfg.target_column, "target", kind_of, broken, reference, current, schema_issues
        )
        prediction_drift = self._analyze_special(
            cfg.prediction_column, "prediction", kind_of, broken, reference, current, schema_issues
        )

        adversarial = (
            adversarial_validation(reference, current, [*numeric_cols, *categorical_cols], cfg)
            if cfg.adversarial_enabled
            else None
        )

        # Пропущенные колонки и малые выборки — информационные замечания (severity ok):
        # они не поднимают тревогу, но видны в отчёте и дашборде.
        for col, reason in skipped.items():
            schema_issues.append(
                Issue(check="column_skipped", severity="ok", column=col,
                      message=f"Колонка '{col}' не анализируется: {reason}.")
            )
        underpowered = [
            c.column for c in column_reports
            if any(t.details.get("underpowered") for t in c.tests)
        ]
        if underpowered:
            quality_issues.append(
                Issue(
                    check="small_batch", severity="ok", value=float(len(underpowered)),
                    message=(
                        f"Батч мал для надёжной оценки PSI/JS по колонкам: {', '.join(underpowered)}. "
                        f"Порог warning поднят до шумового уровня ({cfg.noise_quantile:.0%}-й процентиль без дрейфа)."
                    ),
                )
            )

        overall = worst(
            [
                *(i.severity for i in schema_issues),
                *(i.severity for i in quality_issues),
                *(c.severity for c in column_reports),
                *([adversarial.severity] if adversarial else []),
                *([target_drift.severity] if target_drift else []),
                *([prediction_drift.severity] if prediction_drift else []),
            ]
        )
        alerts = self._build_alerts(
            schema_issues, quality_issues, column_reports, adversarial,
            target_drift, prediction_drift,
        )
        feature_worst = worst(
            [*(i.severity for i in quality_issues), *(c.severity for c in column_reports)]
        )
        if target_drift and target_drift.severity == "critical" and feature_worst != "critical":
            recommendation = _CONCEPT_DRIFT_RECOMMENDATION
        else:
            recommendation = _RECOMMENDATIONS[overall]

        meta = {
            "generated_at": started_at.isoformat(timespec="seconds"),
            "reference_rows": int(len(reference)),
            "current_rows": int(len(current)),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "skipped_columns": skipped_cols,
            "skipped_reasons": skipped,
            "underpowered_columns": underpowered,
            "excluded_columns": sorted(excluded),
            "target_column": cfg.target_column,
            "prediction_column": cfg.prediction_column,
            "alpha_effective": alpha_effective,
            "config": cfg.to_dict(),
        }
        return DriftReport(
            overall_severity=overall,
            recommendation=recommendation,
            alerts=alerts,
            schema=schema_issues,
            data_quality=quality_issues,
            columns=column_reports,
            adversarial=adversarial,
            meta=meta,
            target_drift=target_drift,
            prediction_drift=prediction_drift,
        )

    def _analyze_special(
        self,
        column: str | None,
        role: str,
        kind_of: dict[str, str],
        broken: set[str],
        reference: pd.DataFrame,
        current: pd.DataFrame,
        schema_issues: list[Issue],
    ) -> ColumnReport | None:
        """Анализ целевой переменной / предсказаний отдельным блоком."""
        if not column or column in broken:
            return None
        role_name = _SPECIAL_ROLES[role]
        if column not in reference.columns or column not in current.columns:
            schema_issues.append(
                Issue(
                    check=f"{role}_column_missing",
                    severity="warning",
                    column=column,
                    message=(
                        f"Колонка {role_name} '{column}' отсутствует в одной из выборок — "
                        "блок пропущен."
                    ),
                )
            )
            return None
        kind = kind_of.get(column)
        if kind is None:
            schema_issues.append(
                Issue(
                    check=f"{role}_column_unsupported",
                    severity="warning",
                    column=column,
                    message=f"Тип колонки {role_name} '{column}' не поддерживается — блок пропущен.",
                )
            )
            return None
        alpha = self.config.thresholds.alpha
        if kind == "numeric":
            return analyze_numeric_column(column, reference, current, self.config, alpha)
        return analyze_categorical_column(column, reference, current, self.config, alpha)

    @staticmethod
    def _build_alerts(
        schema_issues: list[Issue],
        quality_issues: list[Issue],
        column_reports: list[ColumnReport],
        adversarial: AdversarialReport | None,
        target_drift: ColumnReport | None,
        prediction_drift: ColumnReport | None,
    ) -> list[str]:
        alerts: list[str] = []
        for issue in (*schema_issues, *quality_issues):
            if issue.severity != "ok":
                alerts.append(f"{_ALERT_PREFIX[issue.severity]}: {issue.message}")
        for special, role in ((target_drift, "target"), (prediction_drift, "prediction")):
            if special and special.severity != "ok":
                alerts.append(
                    f"{_ALERT_PREFIX[special.severity]}: дрейф {_SPECIAL_ROLES[role]} "
                    f"'{special.column}' — {_describe_tests(special)}."
                )
        for col in column_reports:
            if col.severity != "ok":
                alerts.append(
                    f"{_ALERT_PREFIX[col.severity]}: дрейф по признаку '{col.column}' — "
                    f"{_describe_tests(col)}."
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
