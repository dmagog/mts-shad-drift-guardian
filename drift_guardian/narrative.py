"""Текстовый слой: заголовки вердикта, объяснения «что изменилось», факты, форматирование.

Одна реализация для дашборда и HTML-отчёта, чтобы формулировки не расходились.
"""
from __future__ import annotations

import math

from .plots import SEVERITY_RANK, STATUS_LABELS

HEADLINES = {
    "ok": "Дрейф не обнаружен",
    "warning": "Умеренный дрейф",
    "critical": "Критический дрейф",
}
CHECK_LABELS = {
    "missing_values": "пропуски", "duplicates": "дубликаты", "out_of_range": "диапазон",
    "new_categories": "категории", "constant_column": "константа", "missing_column": "схема",
    "extra_column": "схема", "dtype_mismatch": "схема", "column_skipped": "пропущено",
    "small_batch": "малый батч", "target_column_missing": "таргет",
    "prediction_column_missing": "предсказания", "target_column_unsupported": "таргет",
    "prediction_column_unsupported": "предсказания",
}
SPECIAL_TITLES = {"target_drift": "Целевая переменная", "prediction_drift": "Предсказания модели"}
CONCEPT_DRIFT_NOTE = (
    "Признаки стабильны, а целевая переменная изменилась: вероятен концептуальный дрейф, "
    "то есть изменилась связь между признаками и целью, а не сами входные данные."
)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def fmt_num(value, digits: int = 3) -> str:
    return "—" if _missing(value) else f"{float(value):.{digits}f}"


def fmt_p(value) -> str:
    if _missing(value):
        return "—"
    return "p < 1e-16" if value < 1e-16 else f"p = {value:.2g}"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Число с существительным по правилам русского языка: 1 сегмент, 2 сегмента, 5 сегментов."""
    n_abs = abs(int(n))
    if 11 <= n_abs % 100 <= 14:
        form = many
    elif n_abs % 10 == 1:
        form = one
    elif 2 <= n_abs % 10 <= 4:
        form = few
    else:
        form = many
    return f"{n} {form}"


def psi_of(col: dict) -> float:
    return next((t["statistic"] for t in col["tests"] if t["name"] == "psi"), 0.0)


def drifted_columns(report: dict) -> list[dict]:
    """Признаки с дрейфом: сначала критичные, внутри уровня — по убыванию PSI."""
    return sorted(
        (c for c in report.get("columns", []) if c["severity"] != "ok"),
        key=lambda c: (SEVERITY_RANK[c["severity"]], -psi_of(c)),
    )


def issues_by_column(report: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for issue in report.get("data_quality", []):
        if issue.get("column") and issue["severity"] != "ok":
            grouped.setdefault(issue["column"], []).append(issue)
    return grouped


def explain_column(col: dict, issues: list[dict]) -> str:
    """Одна строка «почему признак поплыл» для карточки."""
    tests = {t["name"]: t for t in col["tests"]}
    parts: list[str] = []
    if col["kind"] == "numeric":
        w_test = tests.get("wasserstein_norm", {})
        w = w_test.get("statistic")
        if w is not None and w >= 0.1:
            parts.append(f"сдвиг на {w:.2f}σ")
        if w_test.get("details", {}).get("scale") == "pooled":
            parts.append("эталон почти константен")
    for issue in issues:
        value = issue.get("value") or 0.0
        if issue["check"] == "missing_values":
            parts.append(f"пропусков +{value * 100:.0f} п.п.")
        elif issue["check"] == "out_of_range":
            parts.append(f"{value:.0%} значений вне диапазона")
        elif issue["check"] == "new_categories":
            parts.append(f"новые категории у {value:.0%} строк")
        elif issue["check"] == "constant_column":
            parts.append("стала константой")
    if not parts:
        js = tests.get("jensen_shannon", {}).get("statistic")
        base = "изменились доли категорий" if col["kind"] == "categorical" else "изменилась форма распределения"
        parts.append(f"{base} (JS {js:.2f})" if js is not None else base)
    if any(t.get("details", {}).get("underpowered") for t in col["tests"]):
        parts.append("батч мал")
    text = ", ".join(parts)
    return text[0].upper() + text[1:]


def card_metric(col: dict, issues: list[dict], thresholds) -> tuple[str, float, float, float]:
    """Метрика для полосы карточки: PSI, а если он неинформативен (почти константный эталон,
    признак поплыл за диапазон) — доля значений вне диапазона или новых категорий."""
    psi = psi_of(col)
    if psi < thresholds.psi_warning:
        for issue in issues:
            if issue["check"] == "out_of_range":
                return ("вне диапазона", float(issue.get("value") or 0.0),
                        thresholds.out_of_range_warning, thresholds.out_of_range_critical)
            if issue["check"] == "new_categories":
                return ("новые категории", float(issue.get("value") or 0.0),
                        thresholds.new_category_warning, thresholds.new_category_critical)
    return ("PSI", psi, thresholds.psi_warning, thresholds.psi_critical)


def test_lines(col: dict) -> list[tuple[str, str, str]]:
    """Строки для списка тестов: (severity, «тест: статистика, p», подпись статуса)."""
    lines = []
    for t in col["tests"]:
        if t["name"] == "skipped":
            continue
        text = f"{t['name']}: {t['statistic']:.4g}"
        if t.get("p_value") is not None:
            text += f", {fmt_p(t['p_value'])}"
        lines.append((t["severity"], text, STATUS_LABELS[t["severity"]]))
    return lines


def hero_facts(report: dict) -> list[tuple[str, str]]:
    meta = report["meta"]
    columns = report.get("columns", [])
    n_drifted = sum(c["severity"] != "ok" for c in columns)
    n_quality = sum(i["severity"] != "ok" for i in [*report.get("schema", []), *report.get("data_quality", [])])
    adversarial = report.get("adversarial")
    facts = [
        (" признаков с дрейфом", f"{n_drifted} из {len(columns)}"),
        (" " + plural(n_quality, "замечание", "замечания", "замечаний").split(" ", 1)[1] + " к данным",
         str(n_quality)),
        (" adversarial AUC", f"{adversarial['roc_auc']:.2f}" if adversarial else "выкл."),
        (" строк: эталон / батч", f"{fmt_int(meta['reference_rows'])} / {fmt_int(meta['current_rows'])}"),
    ]
    target = report.get("target_drift")
    if target:
        facts.insert(1, (" целевая переменная", STATUS_LABELS[target["severity"]]))
    return facts


def features_stable(report: dict) -> bool:
    return all(c["severity"] != "critical" for c in report.get("columns", []))


def grid_rows(n: int, per_row: int = 3) -> list[int]:
    """Размеры рядов сетки карточек без «сироты»: 4 → 2+2, 5 → 3+2, 7 → 3+2+2."""
    if n <= 0:
        return []
    if n <= per_row:
        return [n]
    rows = [per_row] * (n // per_row)
    rest = n % per_row
    if rest == 1 and per_row == 3:
        rows[-1] = 2
        rows.append(2)
    elif rest:
        rows.append(rest)
    return rows
