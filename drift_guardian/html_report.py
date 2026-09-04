"""Самодостаточный HTML-отчёт с графиками Plotly и экспорт отчёта в JSON."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from .plots import (
    SEVERITY_RANK,
    STATUS_LABELS,
    categorical_distribution_figure,
    column_summary_frame,
    feature_importance_figure,
    numeric_distribution_figure,
    severity_from_label,
    timeline_psi_figure,
    timeline_severity_figure,
)

SUMMARY_HEADERS = [
    "Признак", "Тип", "Статус", "PSI", "JS", "Вассерштейн (норм.)", "KS / χ²", "p-value",
]
SPECIAL_TITLES = {
    "target_drift": "Целевая переменная",
    "prediction_drift": "Предсказания модели",
}
CONCEPT_DRIFT_NOTE = (
    "Признаки стабильны, а целевая переменная изменилась — вероятен концептуальный дрейф: "
    "изменилась связь между признаками и целью, а не сами входные данные."
)

_TEMPLATE = Template(
    """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; background: #f9f9f7; margin: 0; padding: 24px 32px; }
  main { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h2 { font-size: 18px; margin: 28px 0 12px; }
  .muted { color: #52514e; font-size: 13px; }
  .banner { padding: 14px 18px; border-radius: 8px; margin: 16px 0; border: 1px solid rgba(11,11,11,.1); font-size: 15px; }
  .banner.ok { background: #e8f6e8; } .banner.warning { background: #fff4d6; } .banner.critical { background: #fbe4e4; }
  .kpis { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
  .kpi { background: #fcfcfb; border: 1px solid #e1e0d9; border-radius: 8px; padding: 10px 14px; min-width: 150px; }
  .kpi .v { font-size: 22px; font-weight: 600; } .kpi .l { color: #52514e; font-size: 12px; }
  table { border-collapse: collapse; width: 100%; background: #fcfcfb; font-size: 13px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #e1e0d9; text-align: left; vertical-align: top; }
  th { color: #52514e; font-weight: 600; }
  td.num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
  tr.critical { background: #fbe4e4; } tr.warning { background: #fff4d6; }
  details { background: #fcfcfb; border: 1px solid #e1e0d9; border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
  summary { cursor: pointer; font-weight: 600; }
  ul { padding-left: 20px; } li { margin: 4px 0; }
  .note { background: #fff4d6; border-radius: 8px; padding: 10px 14px; margin: 8px 0; }
</style>
</head>
<body><main>
<header>
  <h1>{{ title }}</h1>
  <p class="muted">Сформирован {{ generated_at }}</p>
</header>

<section class="banner {{ severity }}"><strong>{{ status_label }}</strong> — {{ recommendation }}</section>

<div class="kpis">
  <div class="kpi"><div class="v">{{ meta.reference_rows }}</div><div class="l">строк в эталоне</div></div>
  <div class="kpi"><div class="v">{{ meta.current_rows }}</div><div class="l">строк в батче</div></div>
  <div class="kpi"><div class="v">{{ n_columns }}</div><div class="l">признаков проверено</div></div>
  <div class="kpi"><div class="v">{{ alerts|length }}</div><div class="l">алертов</div></div>
  {% if adversarial %}<div class="kpi"><div class="v">{{ '%.3f'|format(adversarial.roc_auc) }}</div><div class="l">adversarial ROC-AUC</div></div>{% endif %}
</div>

<section>
  <h2>Алерты</h2>
  {% if alerts %}<ul>{% for alert in alerts %}<li>{{ alert }}</li>{% endfor %}</ul>
  {% else %}<p>Алертов нет — распределения стабильны.</p>{% endif %}
</section>

{% for block in special_blocks %}
<section>
  <h2>{{ block.title }} «{{ block.column }}» — {{ block.label }}</h2>
  {% if block.note %}<p class="note">{{ block.note }}</p>{% endif %}
  <table>
    <thead><tr><th>Тест</th><th>Статистика</th><th>p-value</th><th>Статус</th></tr></thead>
    <tbody>{% for t in block.tests %}<tr class="{{ t.severity }}"><td>{{ t.name }}</td><td class="num">{{ t.statistic }}</td><td class="num">{{ t.p_value }}</td><td>{{ t.label }}</td></tr>{% endfor %}</tbody>
  </table>
  {{ block.html|safe }}
</section>
{% endfor %}

<section>
  <h2>Сводка по признакам</h2>
  {% if summary_rows %}
  <table>
    <thead><tr>{% for header in summary_headers %}<th>{{ header }}</th>{% endfor %}</tr></thead>
    <tbody>
    {% for row in summary_rows %}
      <tr class="{{ row.severity }}">{% for cell in row.cells %}<td class="{{ 'num' if loop.index > 3 else '' }}">{{ cell }}</td>{% endfor %}</tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}<p>Нет признаков для анализа.</p>{% endif %}
</section>

<section>
  <h2>Схема и качество данных</h2>
  {% if issues %}
  <table>
    <thead><tr><th>Проверка</th><th>Признак</th><th>Статус</th><th>Сообщение</th></tr></thead>
    <tbody>{% for issue in issues %}<tr class="{{ issue.severity }}"><td>{{ issue.check }}</td><td>{{ issue.column or '—' }}</td><td>{{ issue.label }}</td><td>{{ issue.message }}</td></tr>{% endfor %}</tbody>
  </table>
  {% else %}<p>Замечаний к схеме и качеству данных нет.</p>{% endif %}
</section>

<section>
  <h2>Распределения: эталон против текущего батча</h2>
  {% for plot in plots %}
  <details {% if plot.open %}open{% endif %}><summary>{{ plot.label }} · {{ plot.column }}</summary>{{ plot.html|safe }}</details>
  {% endfor %}
  {% if plots_skipped %}<p class="muted">Показаны {{ plots|length }} признаков; ещё {{ plots_skipped }} без изменений скрыто.</p>{% endif %}
</section>

<section>
  <h2>Adversarial validation</h2>
  {% if adversarial %}
  <p>ROC-AUC = <strong>{{ '%.3f'|format(adversarial.roc_auc) }}</strong> ({{ adversarial.label }}), бэкенд {{ adversarial.backend }}, использовано строк: {{ adversarial.n_rows_used }}.
  Значение около 0.5 означает, что классификатор не отличает эталон от батча; чем выше — тем сильнее изменилась совместная структура данных.</p>
  {{ importance_html|safe }}
  {% else %}<p>Adversarial validation не выполнялась.</p>{% endif %}
</section>

<footer class="muted">
  <p>Пороги: PSI warning ≥ {{ th.psi_warning }}, critical ≥ {{ th.psi_critical }} · JS warning ≥ {{ th.js_warning }}, critical ≥ {{ th.js_critical }} · Вассерштейн/σ warning ≥ {{ th.wasserstein_warning }}, critical ≥ {{ th.wasserstein_critical }} · α = {{ th.alpha }}{% if meta.alpha_effective %} (с поправкой {{ '%.3g'|format(meta.alpha_effective) }}){% endif %} · adversarial ROC-AUC warning ≥ {{ th.adversarial_auc_warning }}, critical ≥ {{ th.adversarial_auc_critical }}.</p>
  <p>Data Drift Guardian · итоговый проект 4.0 Школы аналитиков данных МТС</p>
</footer>
</main></body></html>
"""
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_p(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    return "<1e-16" if value < 1e-16 else f"{value:.2g}"


def _distribution_figure(kind: str, reference: pd.Series, current: pd.Series, title: str) -> go.Figure:
    if kind == "numeric":
        return numeric_distribution_figure(reference, current, title=title)
    return categorical_distribution_figure(reference, current, title=title)


def _figures_to_html(figures: list[go.Figure], plotlyjs: str) -> list[str]:
    """Первая фигура несёт plotly.js (inline или cdn), остальные — только данные."""
    first = True if plotlyjs == "inline" else "cdn"
    return [
        fig.to_html(
            full_html=False,
            include_plotlyjs=first if index == 0 else False,
            config={"displaylogo": False, "responsive": True},
            default_height=360,
        )
        for index, fig in enumerate(figures)
    ]


def render_html_report(
    report: dict,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    title: str = "Data Drift Guardian — отчёт о дрейфе данных",
    max_plots: int = 40,
    plotlyjs: str = "inline",
) -> str:
    """Собирает HTML-отчёт по словарю ``analyze()`` и исходным данным.

    ``plotlyjs="inline"`` встраивает plotly.js (файл ~4 МБ, работает офлайн),
    ``"cdn"`` подгружает библиотеку из сети (файл лёгкий).
    """
    meta = report.get("meta", {}) or {}
    severity = report["overall_severity"]
    thresholds = (meta.get("config") or {}).get("thresholds") or {}

    summary_rows = []
    for _, row in column_summary_frame(report).iterrows():
        summary_rows.append(
            {
                "severity": severity_from_label(row["статус"]),
                "cells": [
                    row["признак"], row["тип"], row["статус"],
                    _fmt(row["PSI"]), _fmt(row["JS"]), _fmt(row["Вассерштейн (норм.)"]),
                    _fmt(row["KS / χ²"]), _fmt_p(row["p-value"]),
                ],
            }
        )

    issues = [
        {**issue, "label": STATUS_LABELS[issue["severity"]]}
        for issue in (*report.get("schema", []), *report.get("data_quality", []))
    ]

    # Отдельные блоки: целевая переменная и предсказания.
    # Ранг 0 — critical; если минимальный ранг среди признаков > 0, критичных признаков нет.
    features_without_critical = (
        min((SEVERITY_RANK[c["severity"]] for c in report.get("columns", [])), default=2) > 0
    )
    special_figs: list[go.Figure] = []
    special_meta: list[dict] = []
    for key, block_title in SPECIAL_TITLES.items():
        block = report.get(key)
        if not block:
            continue
        name = block["column"]
        if name not in reference.columns or name not in current.columns:
            continue
        note = ""
        if key == "target_drift" and block["severity"] == "critical" and features_without_critical:
            note = CONCEPT_DRIFT_NOTE
        special_figs.append(_distribution_figure(block["kind"], reference[name], current[name], name))
        special_meta.append(
            {
                "title": block_title,
                "column": name,
                "label": STATUS_LABELS[block["severity"]],
                "note": note,
                "tests": [
                    {
                        "name": t["name"],
                        "statistic": _fmt(t["statistic"], 4),
                        "p_value": _fmt_p(t.get("p_value")),
                        "severity": t["severity"],
                        "label": STATUS_LABELS[t["severity"]],
                    }
                    for t in block["tests"]
                ],
            }
        )

    columns_sorted = sorted(
        report.get("columns", []),
        key=lambda c: (SEVERITY_RANK[c["severity"]], c["column"]),
    )
    column_figs: list[go.Figure] = []
    plot_meta: list[dict] = []
    for col in columns_sorted[:max_plots]:
        name = col["column"]
        if name not in reference.columns or name not in current.columns:
            continue
        column_figs.append(_distribution_figure(col["kind"], reference[name], current[name], name))
        plot_meta.append(
            {"column": name, "label": STATUS_LABELS[col["severity"]], "open": col["severity"] != "ok"}
        )
    plots_skipped = max(0, len(columns_sorted) - len(plot_meta))

    adversarial = report.get("adversarial")
    has_importance = bool(adversarial and adversarial.get("top_features"))
    figures = [*special_figs, *column_figs]
    if has_importance:
        figures.append(feature_importance_figure(adversarial["top_features"]))

    htmls = _figures_to_html(figures, plotlyjs)
    n_special, n_columns = len(special_figs), len(column_figs)
    special_blocks = [
        {**m, "html": h} for m, h in zip(special_meta, htmls[:n_special], strict=True)
    ]
    plots = [
        {**m, "html": h} for m, h in zip(plot_meta, htmls[n_special:n_special + n_columns], strict=True)
    ]
    importance_html = htmls[n_special + n_columns] if has_importance else ""

    return _TEMPLATE.render(
        title=title,
        generated_at=meta.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
        severity=severity,
        status_label=STATUS_LABELS[severity],
        recommendation=report.get("recommendation", ""),
        meta=meta,
        n_columns=len(report.get("columns", [])),
        alerts=report.get("alerts", []),
        special_blocks=special_blocks,
        summary_headers=SUMMARY_HEADERS,
        summary_rows=summary_rows,
        issues=issues,
        plots=plots,
        plots_skipped=plots_skipped,
        adversarial=(
            {**adversarial, "label": STATUS_LABELS[adversarial["severity"]]}
            if adversarial
            else None
        ),
        importance_html=importance_html,
        th=thresholds,
    )


def save_html_report(
    path: str | Path,
    report: dict,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    **kwargs: Any,
) -> Path:
    """Сохраняет HTML-отчёт на диск и возвращает путь."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html_report(report, reference, current, **kwargs), encoding="utf-8")
    return target


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def report_to_json(report: dict, indent: int = 2) -> str:
    """Сериализует словарь отчёта в JSON (кириллица без экранирования)."""
    return json.dumps(report, ensure_ascii=False, indent=indent, default=_json_default)


_TIMELINE_TEMPLATE = Template(
    """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; background: #f9f9f7; margin: 0; padding: 24px 32px; }
  main { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 28px 0 12px; }
  .muted { color: #52514e; font-size: 13px; }
  .banner { padding: 14px 18px; border-radius: 8px; margin: 16px 0; border: 1px solid rgba(11,11,11,.1); font-size: 15px; }
  .banner.ok { background: #e8f6e8; } .banner.warning { background: #fff4d6; } .banner.critical { background: #fbe4e4; }
  table { border-collapse: collapse; width: 100%; background: #fcfcfb; font-size: 13px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #e1e0d9; text-align: left; vertical-align: top; }
  th { color: #52514e; font-weight: 600; } td.num { font-variant-numeric: tabular-nums; text-align: right; }
  tr.critical { background: #fbe4e4; } tr.warning { background: #fff4d6; }
  details { background: #fcfcfb; border: 1px solid #e1e0d9; border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
  summary { cursor: pointer; font-weight: 600; } ul { padding-left: 20px; } li { margin: 4px 0; }
</style>
</head>
<body><main>
<header><h1>{{ title }}</h1><p class="muted">Сформирован {{ generated_at }} · эталон {{ reference_rows }} строк · периодов: {{ periods|length }}</p></header>
<section class="banner {{ last.overall_severity }}"><strong>Последний период {{ last.label }}: {{ last_label }}</strong> — {{ last.recommendation }}</section>
<section><h2>Статус по периодам</h2>{{ severity_html|safe }}</section>
<section><h2>PSI по периодам</h2>{{ psi_html|safe }}</section>
<section>
  <h2>Сводка</h2>
  <table>
    <thead><tr><th>Период</th><th>Строк</th><th>Статус</th><th>Таргет</th><th>critical</th><th>warning</th><th>Adversarial AUC</th><th>Алертов</th></tr></thead>
    <tbody>{% for p in periods %}<tr class="{{ p.overall_severity }}"><td>{{ p.label }}</td><td class="num">{{ p.rows }}</td><td>{{ p.status_label }}</td><td>{{ p.target_label }}</td><td class="num">{{ p.n_critical }}</td><td class="num">{{ p.n_warning }}</td><td class="num">{{ p.auc }}</td><td class="num">{{ p.alerts|length }}</td></tr>{% endfor %}</tbody>
  </table>
</section>
<section>
  <h2>Алерты по периодам</h2>
  {% for p in periods %}<details {% if p.overall_severity != 'ok' %}open{% endif %}><summary>{{ p.status_label }} · {{ p.label }}</summary>{% if p.alerts %}<ul>{% for a in p.alerts %}<li>{{ a }}</li>{% endfor %}</ul>{% else %}<p class="muted">Алертов нет.</p>{% endif %}</details>{% endfor %}
</section>
<footer class="muted"><p>Data Drift Guardian · итоговый проект 4.0 Школы аналитиков данных МТС</p></footer>
</main></body></html>
"""
)


def render_timeline_html(
    timeline: dict,
    *,
    title: str = "Data Drift Guardian — мониторинг во времени",
    plotlyjs: str = "inline",
) -> str:
    """HTML-отчёт по серии батчей: статус и PSI по периодам, сводная таблица, алерты."""
    thresholds = (timeline.get("meta", {}).get("config") or {}).get("thresholds") or {}
    figures = [
        timeline_severity_figure(timeline),
        timeline_psi_figure(
            timeline,
            psi_warning=thresholds.get("psi_warning", 0.1),
            psi_critical=thresholds.get("psi_critical", 0.2),
        ),
    ]
    severity_html, psi_html = _figures_to_html(figures, plotlyjs)
    periods = [
        {
            **p,
            "status_label": STATUS_LABELS[p["overall_severity"]],
            "target_label": STATUS_LABELS[p["target_severity"]] if p.get("target_severity") else "—",
            "auc": _fmt(p.get("adversarial_auc")),
        }
        for p in timeline["periods"]
    ]
    last = periods[-1] if periods else {"overall_severity": "ok", "label": "—", "recommendation": ""}
    return _TIMELINE_TEMPLATE.render(
        title=title,
        generated_at=timeline.get("meta", {}).get("generated_at", ""),
        reference_rows=timeline.get("reference_rows", 0),
        periods=periods,
        last=last,
        last_label=STATUS_LABELS[last["overall_severity"]],
        severity_html=severity_html,
        psi_html=psi_html,
    )


def save_timeline_html(path: str | Path, timeline: dict, **kwargs: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_timeline_html(timeline, **kwargs), encoding="utf-8")
    return target
