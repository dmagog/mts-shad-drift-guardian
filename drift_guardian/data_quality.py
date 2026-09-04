"""Data Quality: пропуски, дубликаты, диапазоны, допустимые категории, константы.

Границы диапазонов и множества допустимых категорий берутся из контракта данных
(``DriftConfig.value_bounds`` / ``allowed_categories``), а если он не задан —
выводятся из эталона: [min, max] и множество наблюдавшихся категорий.
"""
from __future__ import annotations

import pandas as pd

from .config import DriftConfig
from .contracts import Issue, grade


def check_data_quality(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: DriftConfig,
) -> list[Issue]:
    """Сравнивает показатели качества текущего батча с эталонными и контрактом."""
    th = config.thresholds
    bounds = config.value_bounds or {}
    allowed = config.allowed_categories or {}
    issues: list[Issue] = []

    # 1. Пропуски: прирост доли NaN по каждой анализируемой колонке.
    for col in [*numeric_cols, *categorical_cols]:
        ref_na = float(reference[col].isna().mean())
        cur_na = float(current[col].isna().mean())
        delta = cur_na - ref_na
        severity = grade(delta, th.missing_delta_warning, th.missing_delta_critical)
        if severity != "ok":
            issues.append(
                Issue(
                    check="missing_values",
                    severity=severity,
                    column=col,
                    value=round(delta, 4),
                    message=(
                        f"Доля пропусков в '{col}' выросла с {ref_na:.1%} до {cur_na:.1%} "
                        f"(+{delta * 100:.1f} п.п.)."
                    ),
                )
            )

    # 2. Полные дубликаты строк.
    ref_dup = float(reference.duplicated().mean())
    cur_dup = float(current.duplicated().mean())
    dup_delta = cur_dup - ref_dup
    severity = grade(dup_delta, th.duplicates_delta_warning, th.duplicates_delta_critical)
    if severity != "ok":
        issues.append(
            Issue(
                check="duplicates",
                severity=severity,
                value=round(dup_delta, 4),
                message=(
                    f"Доля дубликатов строк выросла с {ref_dup:.1%} до {cur_dup:.1%} "
                    f"(+{dup_delta * 100:.1f} п.п.)."
                ),
            )
        )

    # 3. Числовые значения вне допустимого диапазона (контракт или [min, max] эталона).
    for col in numeric_cols:
        cur_clean = pd.to_numeric(current[col], errors="coerce").dropna()
        if cur_clean.empty:
            continue
        if col in bounds:
            lo, hi = float(bounds[col][0]), float(bounds[col][1])
            source = "контракта"
        else:
            ref_clean = pd.to_numeric(reference[col], errors="coerce").dropna()
            if ref_clean.empty:
                continue
            lo, hi = float(ref_clean.min()), float(ref_clean.max())
            source = "эталона"
        share = float(((cur_clean < lo) | (cur_clean > hi)).mean())
        severity = grade(share, th.out_of_range_warning, th.out_of_range_critical)
        if severity != "ok":
            issues.append(
                Issue(
                    check="out_of_range",
                    severity=severity,
                    column=col,
                    value=round(share, 4),
                    message=(
                        f"{share:.1%} значений '{col}' вне диапазона {source} [{lo:g}, {hi:g}]."
                    ),
                )
            )

    # 4. Категории вне допустимого множества (контракт или категории эталона).
    for col in categorical_cols:
        cur_clean = current[col].dropna()
        if cur_clean.empty:
            continue
        if col in allowed:
            known = set(allowed[col])
            source = "контрактом"
        else:
            known = set(reference[col].dropna().unique())
            source = "эталоном"
        is_new = ~cur_clean.isin(known)
        share = float(is_new.mean())
        severity = grade(share, th.new_category_warning, th.new_category_critical)
        if severity != "ok":
            new_values = sorted(map(str, set(cur_clean[is_new].unique())))[:5]
            issues.append(
                Issue(
                    check="new_categories",
                    severity=severity,
                    column=col,
                    value=round(share, 4),
                    message=(
                        f"В '{col}' появились категории, не предусмотренные {source}: "
                        f"{new_values} — {share:.1%} строк текущего батча."
                    ),
                )
            )

    # 5. Колонка стала константной, хотя в эталоне была вариативной.
    for col in [*numeric_cols, *categorical_cols]:
        if (
            reference[col].nunique(dropna=True) > 1
            and current[col].nunique(dropna=True) <= 1
        ):
            issues.append(
                Issue(
                    check="constant_column",
                    severity="warning",
                    column=col,
                    message=f"Колонка '{col}' в текущем батче стала константной.",
                )
            )

    return issues
