"""Drift Engines: статистические тесты сдвига распределений.

Числовые признаки: KS-тест, PSI, расстояние Вассерштейна, дистанция Йенсена–Шеннона.
Категориальные признаки: хи-квадрат, PSI, дистанция Йенсена–Шеннона.

Три правила, которые отличают движок от «посчитать метрику и сравнить с порогом»:

1. **Двухключевое правило.** p-value-тесты (KS, хи-квадрат) фиксируют факт
   статистически значимого сдвига, но на больших выборках значим любой микросдвиг,
   а на одинаковых распределениях тесты ложно срабатывают с частотой alpha. Поэтому
   p-value-тест засчитывается (warning), только если хотя бы одна метрика размера
   эффекта (PSI, JS, Вассерштейн) превысила порог warning. Уровень critical
   определяют только метрики размера эффекта.

2. **Сглаживание долей.** Доли в бинах/категориях ограничены снизу ``MIN_SHARE``
   (1e-4): без этого PSI «взрывается» на любой категории, которой не было в эталоне
   (1 % новой категории давал PSI > 0.2). Со сглаживанием 1 % → ≈ 0.05, 5 % → ≈ 0.3.

3. **Защита от малых выборок.** PSI и JS смещены вверх на малых батчах: их
   ожидание под нулевой гипотезой ≈ (k − 1)·(1/n_ref + 1/n_cur). Параметрический
   бутстреп по биновым счётчикам оценивает шумовой уровень (99-й процентиль метрики,
   если бы батч был выборкой из эталона) и доверительный интервал оценки. Порог
   warning поднимается до шумового уровня, если тот выше, а колонка помечается
   ``underpowered`` — мониторинг честно сообщает, что батч мал для выводов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .config import DriftConfig, Thresholds
from .contracts import ColumnReport, TestResult, grade, worst

MIN_SHARE = 1e-4


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
    группируются в «__прочее__», чтобы хи-квадрат не разваливался на длинных хвостах.
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


def _smooth_rows(rows: np.ndarray) -> np.ndarray:
    """Доли по строкам с нижней границей MIN_SHARE (защита от нулей в логарифме)."""
    rows = np.asarray(rows, dtype=float)
    totals = rows.sum(axis=1, keepdims=True)
    shares = np.where(totals > 0, rows / np.where(totals > 0, totals, 1.0), 1.0 / rows.shape[1])
    shares = np.clip(shares, MIN_SHARE, None)
    return shares / shares.sum(axis=1, keepdims=True)


def _smooth(counts: np.ndarray) -> np.ndarray:
    return _smooth_rows(np.asarray(counts, dtype=float)[None, :])[0]


def _psi_js_rows(ref_rows: np.ndarray, cur_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSI и JS-дистанция построчно (векторизовано для бутстрепа)."""
    p, q = _smooth_rows(ref_rows), _smooth_rows(cur_rows)
    psi = np.sum((q - p) * np.log(q / p), axis=1)
    m = (p + q) / 2
    js_div = 0.5 * np.sum(p * np.log2(p / m), axis=1) + 0.5 * np.sum(q * np.log2(q / m), axis=1)
    return psi, np.sqrt(np.clip(js_div, 0.0, None))


def psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Population Stability Index по подсчётам в бинах/категориях."""
    p, q = _smooth(ref_counts), _smooth(cur_counts)
    return float(np.sum((q - p) * np.log(q / p)))


def js_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Дистанция Йенсена–Шеннона (корень из симметризованной KL), диапазон [0, 1]."""
    p, q = _smooth(ref_counts), _smooth(cur_counts)
    return float(jensenshannon(p, q, base=2))


def bootstrap_noise(
    ref_counts: np.ndarray,
    cur_counts: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 200,
    quantile: float = 0.99,
) -> dict[str, float | list[float]]:
    """Параметрический бутстреп по биновым счётчикам.

    ``psi_noise`` / ``js_noise`` — ``quantile``-й процентиль метрики при отсутствии дрейфа
    (обе выборки из распределения эталона при тех же размерах): шумовой уровень для данного n.
    ``psi_ci95`` / ``js_ci95`` — доверительный интервал оценки при наблюдаемых распределениях.
    """
    ref_counts = np.asarray(ref_counts, dtype=float)
    cur_counts = np.asarray(cur_counts, dtype=float)
    n_ref, n_cur = int(round(ref_counts.sum())), int(round(cur_counts.sum()))
    p_hat = _smooth(ref_counts)
    q_hat = _smooth(cur_counts)
    p_hat, q_hat = p_hat / p_hat.sum(), q_hat / q_hat.sum()

    ref_null = rng.multinomial(n_ref, p_hat, size=n_boot)
    cur_null = rng.multinomial(n_cur, p_hat, size=n_boot)
    cur_alt = rng.multinomial(n_cur, q_hat, size=n_boot)
    psi_null, js_null = _psi_js_rows(ref_null, cur_null)
    psi_alt, js_alt = _psi_js_rows(ref_null, cur_alt)
    return {
        "psi_noise": float(np.quantile(psi_null, quantile)),
        "js_noise": float(np.quantile(js_null, quantile)),
        "psi_ci95": [float(v) for v in np.quantile(psi_alt, [0.025, 0.975])],
        "js_ci95": [float(v) for v in np.quantile(js_alt, [0.025, 0.975])],
    }


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _skipped(column: str, kind: str, config: DriftConfig) -> ColumnReport:
    skipped = TestResult(
        name="skipped",
        statistic=0.0,
        severity="ok",
        details={"reason": f"меньше {config.min_samples} непустых значений"},
    )
    return ColumnReport(column=column, kind=kind, severity="ok", tests=[skipped])


def _effect_tests(
    ref_counts: np.ndarray,
    cur_counts: np.ndarray,
    config: DriftConfig,
    extra_details: dict,
    thresholds: Thresholds | None = None,
) -> list[TestResult]:
    """PSI и JS с защитой от малых выборок (общая часть числового и категориального анализа)."""
    th = thresholds or config.thresholds
    psi = psi_from_counts(ref_counts, cur_counts)
    js = js_from_counts(ref_counts, cur_counts)

    noise = None
    if config.sample_size_guard:
        rng = np.random.default_rng(config.random_state)
        noise = bootstrap_noise(
            ref_counts, cur_counts, rng, config.bootstrap_samples, config.noise_quantile
        )

    tests: list[TestResult] = []
    for name, value, warning, critical, noise_key, ci_key in (
        ("psi", psi, th.psi_warning, th.psi_critical, "psi_noise", "psi_ci95"),
        ("jensen_shannon", js, th.js_warning, th.js_critical, "js_noise", "js_ci95"),
    ):
        details = dict(extra_details) if name == "psi" else {}
        warning_eff, critical_eff = warning, critical
        if noise is not None:
            floor = noise[noise_key]
            underpowered = floor > warning
            warning_eff = max(warning, floor)
            critical_eff = max(critical, warning_eff)
            details.update(
                {
                    "noise_floor": round(floor, 4),
                    "noise_quantile": config.noise_quantile,
                    "ci95": [round(v, 4) for v in noise[ci_key]],
                    "underpowered": underpowered,
                }
            )
        threshold = f"warning >= {warning_eff:.3g}, critical >= {critical_eff:.3g}"
        if details.get("underpowered"):
            threshold += " (порог поднят до шумового уровня: батч мал)"
        tests.append(
            TestResult(
                name=name,
                statistic=round(value, 4),
                severity=grade(value, warning_eff, critical_eff),
                threshold=threshold,
                details=details,
            )
        )
    return tests


# ---------- анализ колонок ----------

def analyze_numeric_column(
    column: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    alpha_effective: float,
) -> ColumnReport:
    """Полный набор тестов дрейфа для числовой колонки."""
    th = config.thresholds_for(column)
    # Бесконечности трактуются как пропуски: модуль качества данных сообщает о них отдельно.
    ref = _finite(reference[column])
    cur = _finite(current[column])
    if len(ref) < config.min_samples or len(cur) < config.min_samples:
        return _skipped(column, "numeric", config)

    edges = reference_bin_edges(ref, config.n_bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    tests = _effect_tests(ref_counts, cur_counts, config, {"n_bins": len(edges) - 1}, th)

    # Нормировка на std эталона осмысленна, только если эталон не константен:
    # при std ≈ 0 получались бы астрономические «сдвиги в σ».
    # Если эталон константен (std ≈ 0), нормируем на разброс объединённой выборки:
    # «константа изменилась» тогда даёт сдвиг около 2σ, а не астрономическое число.
    scale = float(np.std(ref))
    tolerance = 1e-9 * max(1.0, float(np.abs(ref).max()), float(np.abs(cur).max()))
    scale_kind = "reference"
    if scale <= tolerance:
        scale, scale_kind = float(np.std(np.concatenate([ref, cur]))), "pooled"
    if scale > tolerance:
        wasserstein = float(stats.wasserstein_distance(ref, cur)) / scale
        tests.append(
            TestResult(
                name="wasserstein_norm",
                statistic=round(wasserstein, 4),
                severity=grade(wasserstein, th.wasserstein_warning, th.wasserstein_critical),
                threshold=(
                    f"warning >= {th.wasserstein_warning}, critical >= {th.wasserstein_critical}"
                ),
                details={"scale_std": round(scale, 6), "scale": scale_kind},
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
        column=column, kind="numeric", severity=worst(t.severity for t in tests), tests=tests
    )


def analyze_categorical_column(
    column: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    alpha_effective: float,
) -> ColumnReport:
    """Полный набор тестов дрейфа для категориальной колонки."""
    th = config.thresholds_for(column)
    ref = reference[column].dropna()
    cur = current[column].dropna()
    if len(ref) < config.min_samples or len(cur) < config.min_samples:
        return _skipped(column, "categorical", config)

    ref_counts, cur_counts, cats = category_frequencies(ref, cur, config.max_categories)
    n_total = len(set(ref.unique()) | set(cur.unique()))
    tests = _effect_tests(
        ref_counts, cur_counts, config,
        {"n_categories": len(cats), "n_categories_total": n_total}, th,
    )

    # Двухключевое правило: хи-квадрат срабатывает только при значимости И заметном эффекте.
    effect_severity = worst(t.severity for t in tests)
    chi2_stat = chi2_p = None
    if len(cats) >= 2:
        try:
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(np.vstack([ref_counts, cur_counts]))
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
        column=column, kind="categorical", severity=worst(t.severity for t in tests), tests=tests
    )
