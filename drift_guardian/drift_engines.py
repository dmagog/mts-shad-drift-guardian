"""Drift Engines: статистические тесты сдвига распределений.

Числовые признаки: KS-тест, PSI, расстояние Вассерштейна, дистанция Йенсена–Шеннона.
Категориальные признаки: хи-квадрат, PSI, дистанция Йенсена–Шеннона.

Про пороги — «двухключевое правило». p-value-тесты (KS, хи-квадрат) фиксируют
сам факт статистически значимого сдвига, но на больших выборках значимым
становится любой микросдвиг, а на одинаковых распределениях они дают ложные
срабатывания с частотой alpha. Поэтому p-value-тест считается сработавшим
(warning), только если одновременно хотя бы одна метрика размера эффекта
(PSI, JS, Вассерштейн) превысила свой порог warning. Сила сдвига — и уровень
critical — определяется только метриками размера эффекта.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .config import DriftConfig
from .contracts import ColumnReport, Severity, TestResult, worst

_EPS = 1e-6


def _grade(value: float, warning: float, critical: float) -> Severity:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


# ---------- подготовка распределений ----------

def reference_bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Края квантильных бинов по эталону; крайние бины открыты в бесконечность."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles)).astype(float)
    if len(edges) < 2:  # вырожденный случай: (почти) константная колонка
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def category_frequencies(
    reference: pd.Series, current: pd.Series, max_categories: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Частоты категорий по объединённому множеству значений.

    Редкие категории (за пределами топ-``max_categories`` по эталону)
    группируются в «__прочее__», чтобы хи-квадрат не разваливался
    на длинных хвостах.
    """
    ref = reference.dropna().astype(object)
    cur = current.dropna().astype(object)
    top = set(ref.value_counts().index[:max_categories])
    ref = ref.where(ref.isin(top), other="__прочее__")
    cur = cur.where(cur.isin(top), other="__прочее__")
    ref_vc = ref.value_counts()
    cur_vc = cur.value_counts()
    cats = sorted(set(ref_vc.index) | set(cur_vc.index), key=str)
    ref_counts = ref_vc.reindex(cats, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur_vc.reindex(cats, fill_value=0).to_numpy(dtype=float)
    return ref_counts, cur_counts, [str(c) for c in cats]


def _smooth(counts: np.ndarray) -> np.ndarray:
    p = counts.astype(float) + _EPS
    return p / p.sum()


def psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Population Stability Index по подсчётам в бинах/категориях."""
    p, q = _smooth(ref_counts), _smooth(cur_counts)
    return float(np.sum((q - p) * np.log(q / p)))


def js_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Дистанция Йенсена–Шеннона (симметризованная KL), диапазон [0, 1]."""
    p, q = _smooth(ref_counts), _smooth(cur_counts)
    return float(jensenshannon(p, q, base=2))


# ---------- анализ колонок ----------

def analyze_numeric_column(
    column: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    alpha_effective: float,
) -> ColumnReport:
    """Полный набор тестов дрейфа для числовой колонки."""
    th = config.thresholds
    ref = pd.to_numeric(reference[column], errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current[column], errors="coerce").dropna().to_numpy(dtype=float)

    if len(ref) < config.min_samples or len(cur) < config.min_samples:
        skipped = TestResult(
            name="skipped",
            statistic=0.0,
            severity="ok",
            details={"reason": f"меньше {config.min_samples} непустых значений"},
        )
        return ColumnReport(column=column, kind="numeric", severity="ok", tests=[skipped])

    tests: list[TestResult] = []

    edges = reference_bin_edges(ref, config.n_bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    psi = psi_from_counts(ref_counts, cur_counts)
    tests.append(
        TestResult(
            name="psi",
            statistic=round(psi, 4),
            severity=_grade(psi, th.psi_warning, th.psi_critical),
            threshold=f"warning >= {th.psi_warning}, critical >= {th.psi_critical}",
            details={"n_bins": len(edges) - 1},
        )
    )

    js = js_from_counts(ref_counts, cur_counts)
    tests.append(
        TestResult(
            name="jensen_shannon",
            statistic=round(js, 4),
            severity=_grade(js, th.js_warning, th.js_critical),
            threshold=f"warning >= {th.js_warning}, critical >= {th.js_critical}",
        )
    )

    scale = float(np.std(ref))
    if scale > 0:
        wasserstein = float(stats.wasserstein_distance(ref, cur)) / scale
        tests.append(
            TestResult(
                name="wasserstein_norm",
                statistic=round(wasserstein, 4),
                severity=_grade(
                    wasserstein, th.wasserstein_warning, th.wasserstein_critical
                ),
                threshold=(
                    f"warning >= {th.wasserstein_warning}, "
                    f"critical >= {th.wasserstein_critical}"
                ),
                details={"scale_std": round(scale, 6)},
            )
        )

    # Двухключевое правило: KS срабатывает только при значимости И заметном эффекте.
    effect_severity = worst(t.severity for t in tests)
    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    significant = bool(ks_p < alpha_effective)
    tests.insert(
        0,
        TestResult(
            name="ks",
            statistic=round(float(ks_stat), 4),
            p_value=float(ks_p),
            severity="warning" if significant and effect_severity != "ok" else "ok",
            threshold=f"p < {alpha_effective:.2g} и размер эффекта >= warning",
            details={"significant": significant},
        ),
    )

    return ColumnReport(
        column=column,
        kind="numeric",
        severity=worst(t.severity for t in tests),
        tests=tests,
    )


def analyze_categorical_column(
    column: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    alpha_effective: float,
) -> ColumnReport:
    """Полный набор тестов дрейфа для категориальной колонки."""
    th = config.thresholds
    ref = reference[column].dropna()
    cur = current[column].dropna()

    if len(ref) < config.min_samples or len(cur) < config.min_samples:
        skipped = TestResult(
            name="skipped",
            statistic=0.0,
            severity="ok",
            details={"reason": f"меньше {config.min_samples} непустых значений"},
        )
        return ColumnReport(column=column, kind="categorical", severity="ok", tests=[skipped])

    ref_counts, cur_counts, cats = category_frequencies(ref, cur, config.max_categories)
    tests: list[TestResult] = []

    psi = psi_from_counts(ref_counts, cur_counts)
    tests.append(
        TestResult(
            name="psi",
            statistic=round(psi, 4),
            severity=_grade(psi, th.psi_warning, th.psi_critical),
            threshold=f"warning >= {th.psi_warning}, critical >= {th.psi_critical}",
            details={"n_categories": len(cats)},
        )
    )

    js = js_from_counts(ref_counts, cur_counts)
    tests.append(
        TestResult(
            name="jensen_shannon",
            statistic=round(js, 4),
            severity=_grade(js, th.js_warning, th.js_critical),
            threshold=f"warning >= {th.js_warning}, critical >= {th.js_critical}",
        )
    )

    # Двухключевое правило: хи-квадрат срабатывает только при значимости И заметном эффекте.
    effect_severity = worst(t.severity for t in tests)
    chi2_stat = chi2_p = None
    if len(cats) >= 2:
        try:
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(
                np.vstack([ref_counts, cur_counts])
            )
        except ValueError:
            chi2_stat = chi2_p = None  # вырожденная таблица — хи-квадрат неприменим
    if chi2_p is not None:
        significant = bool(chi2_p < alpha_effective)
        tests.insert(
            0,
            TestResult(
                name="chi2",
                statistic=round(float(chi2_stat), 4),
                p_value=float(chi2_p),
                severity="warning" if significant and effect_severity != "ok" else "ok",
                threshold=f"p < {alpha_effective:.2g} и размер эффекта >= warning",
                details={"significant": significant},
            ),
        )

    return ColumnReport(
        column=column,
        kind="categorical",
        severity=worst(t.severity for t in tests),
        tests=tests,
    )
