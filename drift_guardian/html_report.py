"""HTML-отчёты в визуальной системе дашборда (``drift_guardian.theme``) и экспорт JSON.

Отчёт по двум батчам повторяет структуру дашборда: вердикт, «что изменилось»,
целевая переменная, замечания к данным, все признаки, распределения, adversarial
validation, журнал алертов. Отчёт по потоку: вердикт последнего периода, динамика,
таблица периодов, алерты по периодам.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from .config import DriftConfig
from .narrative import (
    CHECK_LABELS,
    CONCEPT_DRIFT_NOTE,
    HEADLINES,
    SPECIAL_TITLES,
    card_metric,
    drifted_columns,
    explain_column,
    features_stable,
    fmt_int,
    fmt_num,
    fmt_p,
    hero_facts,
    issues_by_column,
    plural,
    test_lines,
)
from .plots import (
    SEVERITY_RANK,
    STATUS_LABELS,
    categorical_distribution_figure,
    clip,
    column_summary_frame,
    compact_figure,
    feature_importance_figure,
    numeric_distribution_figure,
    segment_heatmap_figure,
    severity_from_label,
    timeline_psi_figure,
    timeline_severity_figure,
)
from .theme import (
    AUTHOR,
    COMPONENT_CSS,
    PROJECT_LINE,
    all_clear,
    chip,
    hero,
    issue_list,
    note,
    section,
    topbar,
)

PAGE_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#0b0b0b;background:#f9f9f7;margin:0;padding:24px 32px;line-height:1.45}
main{max-width:1200px;margin:0 auto}
table{border-collapse:separate;border-spacing:0;width:100%;background:#fff;font-size:13px;border:1px solid #e1e0d9;border-radius:10px;overflow:hidden}
th,td{padding:7px 10px;border-bottom:1px solid #e1e0d9;text-align:left;vertical-align:middle}
th{color:#52514e;font-weight:600;background:#f6f5f1;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
tr:last-child td{border-bottom:0}
details{background:#fff;border:1px solid #e1e0d9;border-radius:10px;padding:8px 14px;margin:8px 0}
summary{cursor:pointer;font-weight:600;display:flex;align-items:center;gap:.6rem;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"›";color:#898781;font-size:18px;line-height:1;transition:transform .15s}
details[open] summary::before{transform:rotate(90deg)}
.dg-alerts{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#52514e;white-space:pre-wrap;margin:8px 0 0}
.dg-card .plotly-graph-div{margin-top:6px}
footer{margin-top:28px}
@media (max-width:800px){.dg-two{grid-template-columns:1fr}}
"""

_PAGE = Template(
    """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>{{ css }}</style>
</head>
<body><main>
{{ body|safe }}
<footer class="dg-meta">{{ footer|safe }}</footer>
</main></body></html>
"""
)

_REPORT_BODY = Template(
    """{{ topbar|safe }}
{{ hero|safe }}
{{ changed_section|safe }}
{% if cards %}<div class="dg-grid" style="grid-template-columns:repeat({{ grid_cols }},minmax(0,1fr))">{% for c in cards %}<div class="dg-card">{{ c.head|safe }}{{ c.html|safe }}</div>{% endfor %}</div>
{% else %}{{ all_clear|safe }}{% endif %}
{% for b in special %}{{ b.section|safe }}<div class="dg-two"><div>{{ b.html|safe }}</div><div>{{ b.list|safe }}{{ b.note|safe }}</div></div>{% endfor %}
{% if segments %}{{ segments_section|safe }}
<table>
<thead><tr><th>Сегмент</th><th>Строк (эталон / батч)</th><th>Доля батча</th><th>Статус</th><th>Critical</th><th>Warning</th><th>Первый алерт</th></tr></thead>
<tbody>{% for s in segments %}<tr><td>{{ s.label }}</td><td class="num">{{ s.rows_reference }} / {{ s.rows }}</td><td class="num">{{ s.share }}</td><td>{{ s.chip|safe }}</td><td class="num">{{ s.n_critical }}</td><td class="num">{{ s.n_warning }}</td><td>{{ s.first_alert }}</td></tr>{% endfor %}</tbody>
</table>
{{ segments_html|safe }}{% endif %}
{% if issues_html %}{{ issues_section|safe }}{{ issues_html|safe }}{% endif %}
{{ infos_note|safe }}
{{ all_section|safe }}
{% if summary_rows %}<table>
<thead><tr>{% for h in headers %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in summary_rows %}<tr><td>{{ r.name }}</td><td>{{ r.kind }}</td><td>{{ r.chip|safe }}</td>{% for c in r.cells %}<td class="num">{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table>{% else %}<p class="dg-note">Нет признаков для анализа.</p>{% endif %}
{{ dist_section|safe }}
{% for p in plots %}<details {% if p.open %}open{% endif %}><summary>{{ p.chip|safe }}<span>{{ p.column }}</span></summary>{{ p.html|safe }}</details>{% endfor %}
{% if plots_skipped %}<p class="dg-note">Показано {{ plots|length }} признаков; ещё {{ plots_skipped }} без изменений скрыто.</p>{% endif %}
{{ adv_section|safe }}
{{ adv_hero|safe }}
{{ importance_html|safe }}
{{ log_section|safe }}
<details><summary><span>Текст алертов для копирования</span></summary><pre class="dg-alerts">{{ alerts_text }}</pre></details>
"""
)

_TIMELINE_BODY = Template(
    """{{ topbar|safe }}
{{ hero|safe }}
{{ dyn_section|safe }}
<div class="dg-two"><div>{{ severity_html|safe }}</div><div>{{ psi_html|safe }}</div></div>
{% if previous_html %}{{ previous_section|safe }}{{ previous_note|safe }}<div class="dg-two"><div>{{ previous_html|safe }}</div><div>{{ previous_psi_html|safe }}</div></div>{% endif %}
{{ periods_section|safe }}
<table>
<thead><tr><th>Период</th><th>Строк</th><th>Статус</th><th>Таргет</th><th>Critical</th><th>Warning</th><th>Adversarial AUC</th><th>Алертов</th></tr></thead>
<tbody>{% for p in periods %}<tr><td>{{ p.label }}</td><td class="num">{{ p.rows }}</td><td>{{ p.chip|safe }}</td><td>{{ p.target_chip|safe }}</td><td class="num">{{ p.n_critical }}</td><td class="num">{{ p.n_warning }}</td><td class="num">{{ p.auc }}</td><td class="num">{{ p.alerts|length }}</td></tr>{% endfor %}</tbody>
</table>
{{ alerts_section|safe }}
{% for p in periods %}<details {% if p.overall_severity != 'ok' %}open{% endif %}><summary>{{ p.chip|safe }}<span>{{ p.label }}</span></summary>{% if p.alerts %}{{ p.list|safe }}{% else %}<p class="dg-note">Алертов нет.</p>{% endif %}</details>{% endfor %}
"""
)


def _distribution_figure(kind: str, reference: pd.Series, current: pd.Series) -> go.Figure:
    if kind == "numeric":
        return numeric_distribution_figure(reference, current, title=None)
    return categorical_distribution_figure(reference, current, title=None)


def _figures_to_html(figures: list[go.Figure], plotlyjs: str, heights: list[int] | None = None) -> list[str]:
    """Первая фигура несёт plotly.js (inline или cdn), остальные — только данные."""
    first = True if plotlyjs == "inline" else "cdn"
    heights = heights or [360] * len(figures)
    return [
        fig.to_html(
            full_html=False,
            include_plotlyjs=first if index == 0 else False,
            config={"displaylogo": False, "responsive": True},
            default_height=height,
        )
        for index, (fig, height) in enumerate(zip(figures, heights, strict=True))
    ]


def _footer(thresholds: dict, alpha_effective: float | None) -> str:
    credit = f"Data Drift Guardian · {PROJECT_LINE} · автор — {AUTHOR}"
    if not thresholds:
        return credit
    alpha = f" (с поправкой {alpha_effective:.3g})" if alpha_effective else ""
    return (
        f"Пороги: PSI {thresholds['psi_warning']} / {thresholds['psi_critical']} · "
        f"JS {thresholds['js_warning']} / {thresholds['js_critical']} · "
        f"Вассерштейн/σ {thresholds['wasserstein_warning']} / {thresholds['wasserstein_critical']} · "
        f"α = {thresholds['alpha']}{alpha} · adversarial ROC-AUC "
        f"{thresholds['adversarial_auc_warning']} / {thresholds['adversarial_auc_critical']}."
        f"<br>{credit}"
    )


def _split(html_list: list[str], sizes: list[int]) -> list[list[str]]:
    out, start = [], 0
    for size in sizes:
        out.append(html_list[start:start + size])
        start += size
    return out


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
    generated = (meta.get("generated_at") or datetime.now().isoformat(timespec="seconds")).replace("T", " ")[:16]

    def present(name: str) -> bool:
        return name in reference.columns and name in current.columns

    # 1. Карточки «что изменилось» (мини-графики).
    drifted = [c for c in drifted_columns(report) if present(c["column"])][:9]
    by_column = issues_by_column(report)
    card_figs = [
        compact_figure(_distribution_figure(c["kind"], reference[c["column"]], current[c["column"]]),
                       categorical=c["kind"] == "categorical")
        for c in drifted
    ]
    # 2. Целевая переменная / предсказания.
    special_meta, special_figs = [], []
    for key, block_title in SPECIAL_TITLES.items():
        block = report.get(key)
        if not block or not present(block["column"]):
            continue
        special_figs.append(_distribution_figure(block["kind"], reference[block["column"]], current[block["column"]]))
        special_meta.append((key, block_title, block))
    # 3. Распределения всех признаков.
    columns_sorted = sorted(
        (c for c in report.get("columns", []) if present(c["column"])),
        key=lambda c: (SEVERITY_RANK[c["severity"]], c["column"]),
    )
    plot_cols = columns_sorted[:max_plots]
    plot_figs = [_distribution_figure(c["kind"], reference[c["column"]], current[c["column"]]) for c in plot_cols]
    # 4. Adversarial validation.
    adversarial = report.get("adversarial")
    importance_figs = [feature_importance_figure(adversarial["top_features"], title=None)] \
        if adversarial and adversarial.get("top_features") else []

    segments_raw = report.get("segments") or []
    segment_figs = [segment_heatmap_figure(segments_raw)] if segments_raw else []

    figures = [*card_figs, *special_figs, *plot_figs, *importance_figs, *segment_figs]
    heights = (
        [150] * len(card_figs) + [300] * len(special_figs) + [340] * len(plot_figs)
        + [320] * len(importance_figs) + [max(180, 70 + 34 * len(segments_raw))] * len(segment_figs)
    )
    htmls = _figures_to_html(figures, plotlyjs, heights)
    card_html, special_html, plot_html, importance_html, segment_html = _split(
        htmls, [len(card_figs), len(special_figs), len(plot_figs), len(importance_figs), len(segment_figs)]
    )
    segments = [
        {
            **s, "chip": chip(s["overall_severity"]), "share": f"{s.get('share_current', 0):.0%}",
            "first_alert": clip(s["alerts"][0] if s["alerts"] else s.get("recommendation", ""), 120),
        }
        for s in segments_raw
    ]

    try:
        config = DriftConfig.from_dict(meta.get("config") or {})
    except (TypeError, ValueError):
        config = DriftConfig()
    cards = []
    for c, html in zip(drifted, card_html, strict=True):
        issues = by_column.get(c["column"], [])
        cards.append({"head": _card_head(c, issues, config.thresholds_for(c["column"])), "html": html})
    n_drifted = len(drifted_columns(report))
    changed_meta = f"{n_drifted} из {len(report.get('columns', []))} признаков" + (
        f", показаны {len(cards)}" if n_drifted > len(cards) else ""
    )

    special = []
    for (key, block_title, block), html in zip(special_meta, special_html, strict=True):
        show_note = key == "target_drift" and block["severity"] == "critical" and features_stable(report)
        special.append(
            {
                "section": section(f"{block_title} «{block['column']}»", STATUS_LABELS[block["severity"]]),
                "html": html,
                "list": issue_list(test_lines(block)),
                "note": note(CONCEPT_DRIFT_NOTE) if show_note else "",
            }
        )

    issues = [*report.get("schema", []), *report.get("data_quality", [])]
    alerts = [i for i in issues if i["severity"] != "ok"]
    infos = [i for i in issues if i["severity"] == "ok"]

    summary_rows = []
    for _, row in column_summary_frame(report).iterrows():
        sev = severity_from_label(row["статус"])
        summary_rows.append(
            {
                "name": row["признак"], "kind": row["тип"], "chip": chip(sev),
                "cells": [fmt_num(row["PSI"]), fmt_num(row["JS"]), fmt_num(row["Вассерштейн (норм.)"]),
                          fmt_num(row["KS / χ²"]), fmt_p(row["p-value"])],
            }
        )

    plots = [
        {"column": c["column"], "chip": chip(c["severity"]), "open": c["severity"] != "ok", "html": html}
        for c, html in zip(plot_cols, plot_html, strict=True)
    ]
    adv_hero = ""
    if adversarial:
        adv_hero = hero(
            adversarial["severity"], f"ROC-AUC {adversarial['roc_auc']:.3f}",
            "Классификатор учится отличать эталон от батча: около 0.5 — выборки неразличимы, "
            "чем выше, тем сильнее изменилась совместная структура признаков.",
            [(" бэкенд", adversarial["backend"]), (" строк использовано", fmt_int(adversarial["n_rows_used"]))],
        )
        if adversarial.get("note"):
            adv_hero += note(adversarial["note"])
    else:
        adv_hero = note("Adversarial validation не выполнялась.")
    engine_notes = "".join(note(text) for text in meta.get("notes", []))

    body = _REPORT_BODY.render(
        topbar=topbar("отчёт о дрейфе данных", f"сформирован {generated}"),
        hero=hero(severity, HEADLINES[severity], report.get("recommendation", ""), hero_facts(report)),
        changed_section=section("Что изменилось", changed_meta if cards else None),
        cards=cards,
        grid_cols=2 if len(cards) in (1, 2, 4) else 3,  # без «сироты»: 4 карточки — 2+2
        all_clear=all_clear(f"Все {len(report.get('columns', []))} признаков стабильны: распределения батча совпадают с эталоном."),
        special=special,
        issues_section=section("Замечания к данным", str(len(alerts))),
        issues_html=issue_list([(i["severity"], i["message"], CHECK_LABELS.get(i["check"], i["check"])) for i in alerts]) if alerts else "",
        infos_note=engine_notes + (note(" ".join(i["message"] for i in infos)) if infos else ""),
        all_section=section("Все признаки"),
        headers=["Признак", "Тип", "Статус", "PSI", "JS", "Вассерштейн (норм.)", "KS / χ²", "p-value"],
        summary_rows=summary_rows,
        dist_section=section("Распределения: эталон против батча"),
        plots=plots,
        plots_skipped=max(0, len(columns_sorted) - len(plots)),
        adv_section=section("Adversarial validation"),
        adv_hero=adv_hero,
        importance_html=importance_html[0] if importance_html else "",
        segments=segments,
        segments_section=section(
            f"По сегментам «{meta.get('segment_column', '')}»",
            f"показаны {len(segments)} из {meta['segment_values_total']}"
            if meta.get("segment_values_total") and meta["segment_values_total"] > len(segments)
            else plural(len(segments), "сегмент", "сегмента", "сегментов"),
        ),
        segments_html=segment_html[0] if segment_html else "",
        log_section=section("Журнал алертов", str(len(report.get("alerts", [])))),
        alerts_text="\n".join(report.get("alerts", [])) or "Алертов нет.",
    )
    return _PAGE.render(
        title=title, css=COMPONENT_CSS + PAGE_CSS, body=body,
        footer=_footer(thresholds, meta.get("alpha_effective")),
    )


def _card_head(col: dict, issues: list[dict], thresholds) -> str:
    from .theme import card_head

    label, value, warn, crit = card_metric(col, issues, thresholds)
    return card_head(col["column"], col["severity"], explain_column(col, issues), label, value, warn, crit)


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


def render_timeline_html(
    timeline: dict,
    *,
    title: str = "Data Drift Guardian — мониторинг во времени",
    plotlyjs: str = "inline",
) -> str:
    """HTML-отчёт по серии батчей: статус и PSI по периодам, сводная таблица, алерты."""
    thresholds = (timeline.get("meta", {}).get("config") or {}).get("thresholds") or {}
    periods_raw = timeline.get("periods", [])
    generated = (timeline.get("meta", {}).get("generated_at") or "").replace("T", " ")[:16]
    figures = [
        timeline_severity_figure(timeline, title=None),
        timeline_psi_figure(
            timeline, psi_warning=thresholds.get("psi_warning", 0.1),
            psi_critical=thresholds.get("psi_critical", 0.2), title=None,
        ),
    ]
    compare_previous = bool(timeline.get("meta", {}).get("compare_previous"))
    if compare_previous:
        figures += [
            timeline_severity_figure(timeline, title=None, source="previous"),
            timeline_psi_figure(
                timeline, psi_warning=thresholds.get("psi_warning", 0.1),
                psi_critical=thresholds.get("psi_critical", 0.2), title=None, source="previous",
            ),
        ]
    htmls = _figures_to_html(figures, plotlyjs, [320] * len(figures))
    severity_html, psi_html = htmls[0], htmls[1]
    previous_html, previous_psi_html = (htmls[2], htmls[3]) if compare_previous else ("", "")
    periods = [
        {
            **p,
            "chip": chip(p["overall_severity"]),
            "target_chip": chip(p["target_severity"]) if p.get("target_severity") else "—",
            "auc": fmt_num(p.get("adversarial_auc")),
            "list": issue_list([
                ("critical" if a.startswith("КРИТИЧНО") else "warning", a.split(": ", 1)[-1], "")
                for a in p["alerts"]
            ]),
        }
        for p in periods_raw
    ]
    if periods:
        last = periods[-1]
        first_bad = next((p["label"] for p in periods if p["overall_severity"] != "ok"), "нет")
        hero_html = hero(
            last["overall_severity"],
            f"{HEADLINES[last['overall_severity']]} в последнем периоде {last['label']}",
            last["recommendation"],
            [
                (" периодов", str(len(periods))),
                (" критичных", str(sum(p["overall_severity"] == "critical" for p in periods))),
                (" первый период с дрейфом", first_bad),
                (" строк в эталоне", fmt_int(timeline.get("reference_rows", 0))),
            ],
        )
    else:
        hero_html = note("В потоке не нашлось ни одного периода с данными.")
    body = _TIMELINE_BODY.render(
        topbar=topbar("мониторинг во времени", f"сформирован {generated}" if generated else ""),
        hero=hero_html,
        dyn_section=section("Динамика по периодам: относительно эталона"),
        severity_html=severity_html,
        psi_html=psi_html,
        previous_section=section("Относительно предыдущего периода"),
        previous_note=note(
            "Сравнение с эталоном показывает накопленный дрейф, сравнение с предыдущим периодом — "
            "скачки: медленно плывущий признак здесь остаётся в норме, резкое изменение видно "
            "в том периоде, где произошло."
        ),
        previous_html=previous_html,
        previous_psi_html=previous_psi_html,
        periods_section=section("Статус по периодам", f"{len(periods)}"),
        periods=periods,
        alerts_section=section("Алерты по периодам"),
    )
    return _PAGE.render(title=title, css=COMPONENT_CSS + PAGE_CSS, body=body, footer=_footer(thresholds, None))


def save_timeline_html(path: str | Path, timeline: dict, **kwargs: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_timeline_html(timeline, **kwargs), encoding="utf-8")
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
