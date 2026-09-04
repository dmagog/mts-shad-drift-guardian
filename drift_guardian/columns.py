"""Определение типов колонок: какие анализировать как числовые, какие как категориальные."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from .config import DriftConfig


def split_columns(
    reference: pd.DataFrame, current: pd.DataFrame, config: DriftConfig
) -> tuple[list[str], list[str], list[str]]:
    """Возвращает (числовые, категориальные, пропущенные) общие колонки.

    Эвристика: числовой dtype — числовая колонка, кроме целочисленных с малым
    числом уникальных значений (коды, флаги) — они категориальные. Строки,
    category и bool — категориальные. Остальные типы (например, datetime)
    пока не анализируются и попадают в «пропущенные».

    Списки ``config.numeric_columns`` / ``config.categorical_columns``
    имеют приоритет над эвристикой.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    skipped: list[str] = []
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
        elif ptypes.is_bool_dtype(series):
            categorical.append(col)
        elif ptypes.is_numeric_dtype(series):
            if (
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
            categorical.append(col)
        else:
            skipped.append(col)
    return numeric, categorical, skipped
