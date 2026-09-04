"""Schema Validation: проверка соответствия текущего батча контракту эталона."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from .contracts import Issue


def _kind(dtype) -> str:
    if ptypes.is_bool_dtype(dtype):
        return "bool"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    if ptypes.is_numeric_dtype(dtype):
        return "numeric"
    return "categorical"


def validate_schema(reference: pd.DataFrame, current: pd.DataFrame) -> list[Issue]:
    """Сверяет набор колонок и семейства типов между эталоном и текущим батчем."""
    issues: list[Issue] = []

    if reference.empty:
        issues.append(
            Issue(
                check="empty_reference",
                severity="critical",
                message="Эталонный датасет пуст — анализ невозможен.",
            )
        )
    if current.empty:
        issues.append(
            Issue(
                check="empty_current",
                severity="critical",
                message="Текущий батч пуст — анализ невозможен.",
            )
        )
    if issues:
        return issues

    for where, frame in (("эталоне", reference), ("текущем батче", current)):
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        if duplicates:
            issues.append(
                Issue(
                    check="duplicate_columns",
                    severity="critical",
                    column=str(duplicates[0]),
                    message=(
                        f"В {where} дублируются имена колонок: {duplicates[:5]} — "
                        "анализ невозможен, переименуйте колонки."
                    ),
                )
            )
    if issues:
        return issues

    for col in reference.columns:
        if col not in current.columns:
            issues.append(
                Issue(
                    check="missing_column",
                    severity="critical",
                    column=col,
                    message=f"Колонка '{col}' есть в эталоне, но отсутствует в текущем батче.",
                )
            )
    for col in current.columns:
        if col not in reference.columns:
            issues.append(
                Issue(
                    check="extra_column",
                    severity="warning",
                    column=col,
                    message=(
                        f"Колонка '{col}' появилась в текущем батче, но её нет в эталоне — "
                        "исключена из анализа."
                    ),
                )
            )
    for col in reference.columns:
        if col in current.columns:
            ref_kind = _kind(reference[col].dtype)
            cur_kind = _kind(current[col].dtype)
            if ref_kind != cur_kind:
                issues.append(
                    Issue(
                        check="dtype_mismatch",
                        severity="critical",
                        column=col,
                        message=(
                            f"Тип колонки '{col}' изменился: {ref_kind} "
                            f"({reference[col].dtype}) → {cur_kind} ({current[col].dtype})."
                        ),
                    )
                )
    return issues
