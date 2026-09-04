"""Data Quality: пропуски, дубликаты, выход за диапазоны, новые категории."""
from __future__ import annotations

import pandas as pd

from .config import DriftConfig
from .contracts import Issue, Severity


def _grade(value: float, warning: float, critical: float) -> Severity:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


def check_data_quality(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: DriftConfig,
) -> list[Issue]:
    """Сравнивает показатели качества текущего батча с эталонными."""
    th = config.thresholds
    issues: list[Issue] = []

    # 1. Пропуски: прирост доли NaN по каждой анализируемой колонке.
    for col in [*numeric_cols, *categorical_cols]:
        ref_na = float(reference[col].isna().mean())
        cur_na = float(current[col].isna().mean())
        delta = cur_na - ref_na
        severity = _grade(delta, th.missing_delta_warning, th.missing_delta_critical)
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
    severity = _grade(dup_delta, th.duplicates_delta_warning, th.duplicates_delta_critical)
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

    # 3. Числовые значения вне диапазона [min, max] эталона.
    for col in numeric_cols:
        ref_clean = reference[col].dropna()
        cur_clean = current[col].dropna()
        if ref_clean.empty or cur_clean.empty:
            continue
        lo, hi = float(ref_clean.min()), float(ref_clean.max())
        share = float(((cur_clean < lo) | (cur_clean > hi)).mean())
        severity = _grade(share, th.out_of_range_warning, th.out_of_range_critical)
        if severity != "ok":
            issues.append(
                Issue(
                    check="out_of_range",
                    severity=severity,
                    column=col,
                    value=round(share, 4),
                    message=(
                        f"{share:.1%} значений '{col}' вне диапазона эталона [{lo:g}, {hi:g}]."
                    ),
                )
            )

    # 4. Категории, которых не было в эталоне.
    for col in categorical_cols:
        ref_cats = set(reference[col].dropna().unique())
        cur_clean = current[col].dropna()
        if cur_clean.empty:
            continue
        is_new = ~cur_clean.isin(ref_cats)
        share = float(is_new.mean())
        severity = _grade(share, th.new_category_warning, th.new_category_critical)
        if severity != "ok":
            new_values = sorted(map(str, set(cur_clean[is_new].unique())))[:5]
            issues.append(
                Issue(
                    check="new_categories",
                    severity=severity,
                    column=col,
                    value=round(share, 4),
                    message=(
                        f"В '{col}' появились новые категории {new_values} — "
                        f"{share:.1%} строк текущего батча."
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
