"""Определение типов и ролей колонок.

Какие колонки анализировать как числовые, какие как категориальные, а какие
пропустить: даты (dtype datetime или строки вида 2026-09-04), идентификаторы
(почти все значения уникальны) и неподдерживаемые типы. Пропущенные колонки
не исчезают молча — причина попадает в отчёт.
"""
from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as ptypes

from .config import DriftConfig

_DATE_PATTERN = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}|\d{2}[./]\d{2}[./]\d{4})")


def _looks_like_datetime(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return bool(sample.map(lambda v: _DATE_PATTERN.match(v) is not None).mean() >= 0.9)


def _looks_like_identifier(series: pd.Series, config: DriftConfig) -> bool:
    non_null = series.dropna()
    if len(non_null) < config.identifier_min_rows:
        return False
    if not (ptypes.is_integer_dtype(non_null) or ptypes.is_object_dtype(non_null)
            or ptypes.is_string_dtype(non_null)):
        return False
    return non_null.nunique() / len(non_null) >= config.identifier_unique_share


def split_columns(
    reference: pd.DataFrame, current: pd.DataFrame, config: DriftConfig
) -> tuple[list[str], list[str], dict[str, str]]:
    """Возвращает (числовые, категориальные, пропущенные с причиной) общие колонки.

    Эвристика: числовой dtype — числовая колонка, кроме целочисленных с малым
    числом уникальных значений (коды, флаги) — они категориальные. Строки,
    category и bool — категориальные. Даты и идентификаторы пропускаются.
    Списки ``config.numeric_columns`` / ``config.categorical_columns``
    имеют приоритет над эвристикой.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    skipped: dict[str, str] = {}
    forced_numeric = set(config.numeric_columns or [])
    forced_categorical = set(config.categorical_columns or [])

    for col in reference.columns:
        if col not in current.columns:
            continue
        series = reference[col]
        if col in forced_numeric:
            numeric.append(col)
        elif col in forced_categorical:
            categorical.append(col)
        elif ptypes.is_datetime64_any_dtype(series):
            skipped[col] = "дата/время — исключите или превратите в признаки"
        elif ptypes.is_bool_dtype(series):
            categorical.append(col)
        elif ptypes.is_numeric_dtype(series):
            if ptypes.is_integer_dtype(series) and _looks_like_identifier(series, config):
                skipped[col] = "почти все значения уникальны (идентификатор)"
            elif (
                ptypes.is_integer_dtype(series)
                and series.nunique(dropna=True) <= config.categorical_int_unique_limit
            ):
                categorical.append(col)
            else:
                numeric.append(col)
        elif (
            isinstance(series.dtype, pd.CategoricalDtype)
            or ptypes.is_object_dtype(series)
            or ptypes.is_string_dtype(series)
        ):
            if _looks_like_datetime(series):
                skipped[col] = "строки с датами — исключите или превратите в признаки"
            elif _looks_like_identifier(series, config):
                skipped[col] = "почти все значения уникальны (идентификатор или свободный текст)"
            else:
                categorical.append(col)
        else:
            skipped[col] = f"неподдерживаемый тип {series.dtype}"
    return numeric, categorical, skipped
