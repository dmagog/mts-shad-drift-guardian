"""Контракт результатов анализа.

Вход системы — два ``pandas.DataFrame`` (Reference и Current), выход —
структурированный словарь с метриками и флагами алертов. Внутри результаты
собираются в dataclass-объекты и сериализуются методом ``DriftReport.to_dict()``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

Severity = Literal["ok", "warning", "critical"]

_SEVERITY_ORDER: dict[str, int] = {"ok": 0, "warning": 1, "critical": 2}


def worst(severities: Iterable[Severity]) -> Severity:
    """Наихудшая из степеней серьёзности (ok < warning < critical)."""
    return max(severities, key=_SEVERITY_ORDER.__getitem__, default="ok")


@dataclass
class TestResult:
    """Результат одного статистического теста по одной колонке."""

    name: str
    statistic: float
    severity: Severity
    p_value: float | None = None
    threshold: str | None = None  # человекочитаемое описание порога
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnReport:
    """Сводка дрейфа по одной колонке."""

    column: str
    kind: Literal["numeric", "categorical"]
    severity: Severity
    tests: list[TestResult] = field(default_factory=list)


@dataclass
class Issue:
    """Замечание проверки схемы или качества данных."""

    check: str
    severity: Severity
    message: str
    column: str | None = None
    value: float | None = None


@dataclass
class AdversarialReport:
    """Результат adversarial-валидации «эталон против текущего батча»."""

    roc_auc: float
    severity: Severity
    backend: str
    n_rows_used: int
    top_features: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DriftReport:
    """Полный отчёт анализа. ``to_dict()`` — выходной контракт системы."""

    overall_severity: Severity
    recommendation: str
    alerts: list[str]
    schema: list[Issue]
    data_quality: list[Issue]
    columns: list[ColumnReport]
    adversarial: AdversarialReport | None
    meta: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)
