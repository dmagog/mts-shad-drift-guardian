"""Оркестратор Data Drift Guardian: единая точка входа для полного анализа."""
from __future__ import annotations

from dataclasses import asdict, replace
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
from .summaries import summarize_period

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
        self._segment_values_total: int | None = None

    def run(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        cfg = self.config
        started_at = datetime.now(timezone.utc)

        schema_issues = validate_schema(reference, current)
        if any(
            i.check in ("empty_reference", "empty_current", "duplicate_columns")
            for i in schema_issues
        ):
            return self._aborted_report(schema_issues, reference, current, started_at)
        # Слишком мало строк с любой стороны: статистика невозможна, а проверки диапазонов
        # и категорий против крошечного эталона дали бы мусорные алерты.
        if min(len(reference), len(current)) < cfg.min_samples:
            return self._insufficient_report(schema_issues, reference, current, started_at)

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
        segments = self._analyze_segments(reference, current, schema_issues)
        critical_segments = [s for s in (segments or []) if s["overall_severity"] == "critical"]
        for segment in critical_segments:
            drifted = [c for c, sev in segment["severity_by_column"].items() if sev == "critical"]
            alerts.append(
                f"КРИТИЧНО: в сегменте {cfg.segment_column}={segment['label']} "
                f"критический дрейф по признакам: {', '.join(map(str, drifted)) or 'см. разбор сегмента'}."
            )
        if critical_segments:
            overall = worst([overall, "warning"])

        feature_worst = worst(
            [*(i.severity for i in quality_issues), *(c.severity for c in column_reports)]
        )
        # Все тесты пропущены (например, колонки почти пустые): честно говорим об этом,
        # а не рапортуем «ok» по пропущенным тестам.
        insufficient = bool(
            column_reports and all(c.tests and c.tests[0].name == "skipped" for c in column_reports)
        )
        notes: list[str] = []
        if adversarial and adversarial.note:
            notes.append(adversarial.note)
        if insufficient:
            overall = worst([overall, "warning"])
            recommendation = (
                "Недостаточно непустых значений для статистических выводов: все тесты пропущены. "
                "Проверьте заполненность колонок или накопите больше данных."
            )
            alerts.insert(0, "ВНИМАНИЕ: во всех колонках меньше минимума непустых значений; тесты пропущены.")
        elif target_drift and target_drift.severity == "critical" and feature_worst != "critical":
            recommendation = _CONCEPT_DRIFT_RECOMMENDATION
        elif overall == "ok" and underpowered:
            recommendation = (
                f"Дрейф не обнаружен, но батч мал ({len(current)} строк): чувствительность PSI и JS "
                "снижена, пороги подняты до шумового уровня."
            )
        elif critical_segments and feature_worst != "critical":
            names = ", ".join(f"{cfg.segment_column}={s['label']}" for s in critical_segments[:3])
            recommendation = (
                f"В объединённых данных дрейф умеренный, но внутри сегментов ({names}) он критический: "
                "проверьте данные и качество модели для этих сегментов."
            )
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
            "insufficient_data": insufficient,
            "notes": notes,
            "segment_column": cfg.segment_column if segments is not None else None,
            "segment_values_total": self._segment_values_total,
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
            segments=segments,
        )

    def _analyze_segments(
        self, reference: pd.DataFrame, current: pd.DataFrame, schema_issues: list[Issue]
    ) -> list[dict] | None:
        """Повторяет анализ внутри каждого из самых частых значений segment_column."""
        cfg = self.config
        column = cfg.segment_column
        if not column:
            return None
        if column in (cfg.target_column, cfg.prediction_column):
            schema_issues.append(
                Issue(
                    check="segment_column_is_target", severity="warning", column=column,
                    message=f"Разрез по '{column}' не имеет смысла: это целевая переменная или предсказание — разрез пропущен.",
                )
            )
            return None
        if column not in reference.columns or column not in current.columns:
            schema_issues.append(
                Issue(
                    check="segment_column_missing", severity="warning", column=column,
                    message=f"Колонка сегментов '{column}' отсутствует в одной из выборок — разрез пропущен.",
                )
            )
            return None
        all_values = reference[column].value_counts(dropna=True)
        self._segment_values_total = int(len(all_values))
        values = list(all_values.index[: cfg.max_segments])
        # Отбираем самые частые значения, но числовые метки (день недели, месяц, версия)
        # показываем по порядку, а не по частоте.
        try:
            values.sort(key=float)
        except (TypeError, ValueError):
            pass
        sub_config = replace(
            cfg,
            segment_column=None,
            adversarial_enabled=cfg.adversarial_enabled and cfg.segment_adversarial,
            exclude_columns=[*(cfg.exclude_columns or []), column],
        )
        guardian = DriftGuardian(sub_config)
        n_ref, n_cur = max(len(reference), 1), max(len(current), 1)
        segments: list[dict] = []
        for value in values:
            ref_part = reference[reference[column] == value]
            cur_part = current[current[column] == value]
            if cur_part.empty:
                summary = {
                    "label": str(value), "rows": 0, "overall_severity": "warning",
                    "recommendation": "Сегмент отсутствует в текущем батче.",
                    "n_critical": 0, "n_warning": 0, "target_severity": None, "adversarial_auc": None,
                    "alerts": [], "psi_by_column": {}, "severity_by_column": {},
                    "insufficient": True, "vs_previous": None,
                }
            else:
                summary = asdict(summarize_period(str(value), guardian.run(ref_part, cur_part), len(cur_part)))
            summary["rows_reference"] = int(len(ref_part))
            summary["share_reference"] = round(len(ref_part) / n_ref, 4)
            summary["share_current"] = round(len(cur_part) / n_cur, 4)
            segments.append(summary)
        return segments

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

    def _insufficient_report(
        self,
        schema_issues: list[Issue],
        reference: pd.DataFrame,
        current: pd.DataFrame,
        started_at: datetime,
    ) -> DriftReport:
        cfg = self.config
        small = "эталон" if len(reference) < cfg.min_samples else "батч"
        rows = len(reference) if small == "эталон" else len(current)
        alert = (
            f"ВНИМАНИЕ: {small} содержит {rows} строк — меньше минимума {cfg.min_samples}; "
            "статистические тесты пропущены."
        )
        return DriftReport(
            overall_severity="warning",
            recommendation=(
                f"Недостаточно данных для статистических выводов: эталон {len(reference)} строк, "
                f"батч {len(current)} строк при минимуме {cfg.min_samples}. Накопите больше данных "
                "или укрупните период."
            ),
            alerts=[*(f"КРИТИЧНО: {i.message}" for i in schema_issues if i.severity == "critical"), alert],
            schema=schema_issues,
            data_quality=[],
            columns=[],
            adversarial=None,
            meta={
                "generated_at": started_at.isoformat(timespec="seconds"),
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
                "numeric_columns": [], "categorical_columns": [], "skipped_columns": [],
                "skipped_reasons": {}, "underpowered_columns": [], "excluded_columns": [],
                "insufficient_data": True, "notes": [],
                "segment_column": None, "segment_values_total": None,
                "target_column": cfg.target_column, "prediction_column": cfg.prediction_column,
                "alpha_effective": cfg.thresholds.alpha, "config": cfg.to_dict(),
            },
        )

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
