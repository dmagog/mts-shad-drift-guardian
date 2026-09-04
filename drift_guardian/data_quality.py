"""Data Quality: пропуски, дубликаты, диапазоны, допустимые категории, константы, бесконечности.

Границы диапазонов и множества допустимых категорий берутся из контракта данных
(``DriftConfig.value_bounds`` / ``allowed_categories``), а если он не задан —
выводятся из эталона: [min, max] и множество наблюдавшихся категорий.

Проверки долей (пропуски, дубликаты, значения вне диапазона, новые категории)
работают по двухключевому правилу: доля должна превысить порог **и** статистически
отличаться от нуля (нижняя граница 95 %-го интервала > 0). Иначе на батче в десять
строк одна пустая ячейка давала бы «+12 п.п. пропусков».
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DriftConfig
from .contracts import Issue, grade

_Z = 1.96


def _confident_delta(p_ref: float, n_ref: int, p_cur: float, n_cur: int) -> bool:
    """Прирост доли статистически отличим от нуля (нормальное приближение)."""
    if n_ref <= 0 or n_cur <= 0:
        return False
    se = math.sqrt(p_ref * (1 - p_ref) / n_ref + p_cur * (1 - p_cur) / n_cur)
    return (p_cur - p_ref) - _Z * se > 0


def _confident_share(share: float, n: int) -> bool:
    """Доля статистически отличима от нуля."""
    if n <= 0 or share <= 0:
        return False
    return share - _Z * math.sqrt(share * (1 - share) / n) > 0


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
        th = config.thresholds_for(col)
        ref_na = float(reference[col].isna().mean())
        cur_na = float(current[col].isna().mean())
        delta = cur_na - ref_na
        severity = grade(delta, th.missing_delta_warning, th.missing_delta_critical)
        if severity != "ok" and _confident_delta(ref_na, len(reference), cur_na, len(current)):
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

    # 2. Полные дубликаты строк (общие пороги: проверка на уровне таблицы).
    th = config.thresholds
    ref_dup = float(reference.duplicated().mean())
    cur_dup = float(current.duplicated().mean())
    dup_delta = cur_dup - ref_dup
    severity = grade(dup_delta, th.duplicates_delta_warning, th.duplicates_delta_critical)
    if severity != "ok" and _confident_delta(ref_dup, len(reference), cur_dup, len(current)):
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

    # 3. Бесконечности — ошибка в данных, а не значение.
    for col in numeric_cols:
        values = pd.to_numeric(current[col], errors="coerce").to_numpy(dtype=float)
        n_inf = int(np.isinf(values).sum())
        if n_inf:
            share = n_inf / len(values)
            issues.append(
                Issue(
                    check="non_finite",
                    severity="critical" if share >= 0.05 else "warning",
                    column=col,
                    value=round(share, 4),
                    message=f"В '{col}' {n_inf} бесконечных значений ({share:.1%}) — ошибка в данных.",
                )
            )

    # 3b. Диапазоны не пересекаются: почти всегда время, счётчик или идентификатор —
    # такой признак «дрейфует» тривиально и заслоняет остальные.
    for col in numeric_cols:
        ref_vals = pd.to_numeric(reference[col], errors="coerce")
        cur_vals = pd.to_numeric(current[col], errors="coerce")
        ref_vals, cur_vals = ref_vals[np.isfinite(ref_vals)], cur_vals[np.isfinite(cur_vals)]
        if len(ref_vals) >= 20 and len(cur_vals) >= 20 and (
            cur_vals.min() > ref_vals.max() or cur_vals.max() < ref_vals.min()
        ):
            issues.append(
                Issue(
                    check="disjoint_ranges",
                    severity="ok",
                    column=col,
                    message=(
                        f"Диапазоны '{col}' в эталоне и батче не пересекаются "
                        f"([{ref_vals.min():g}, {ref_vals.max():g}] против [{cur_vals.min():g}, {cur_vals.max():g}]). "
                        "Если это время, счётчик или идентификатор — исключите колонку из анализа; "
                        "иначе батч целиком вышел за пределы эталона."
                    ),
                )
            )

    # 4. Числовые значения вне допустимого диапазона (контракт или [min, max] эталона).
    for col in numeric_cols:
        th = config.thresholds_for(col)
        cur_clean = pd.to_numeric(current[col], errors="coerce")
        cur_clean = cur_clean[np.isfinite(cur_clean)]
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
        if severity != "ok" and _confident_share(share, len(cur_clean)):
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

    # 5. Категории вне допустимого множества (контракт или категории эталона).
    for col in categorical_cols:
        th = config.thresholds_for(col)
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
        if severity != "ok" and _confident_share(share, len(cur_clean)):
            new_set = set(cur_clean[is_new].unique())
            new_values = sorted(map(str, new_set))[:5]
            known_norm = {str(k).strip().casefold() for k in known}
            near = sum(str(v).strip().casefold() in known_norm for v in new_set)
            hint = ""
            if near == len(new_set):
                hint = (
                    " Отличаются от известных только пробелами или регистром — "
                    "скорее всего, изменился пайплайн подготовки данных."
                )
            issues.append(
                Issue(
                    check="new_categories",
                    severity=severity,
                    column=col,
                    value=round(share, 4),
                    message=(
                        f"В '{col}' появились категории, не предусмотренные {source}: "
                        f"{new_values} — {share:.1%} строк текущего батча.{hint}"
                    ),
                )
            )

    # 6. Колонка стала константной, хотя в эталоне была вариативной.
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
