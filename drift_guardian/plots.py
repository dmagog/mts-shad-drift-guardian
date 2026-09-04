"""Графики Plotly и таблицы для дашборда и HTML-отчёта.

Палитра: две серии «эталон / текущий батч» — синий и оранжевый в фиксированном
порядке (пара проходит проверку на различимость при дальтонизме). Статусы —
зарезервированные цвета ok / warning / critical, всегда вместе с иконкой
и подписью, никогда цветом в одиночку.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES_REFERENCE = "#2a78d6"
SERIES_CURRENT = "#eb6834"
STATUS_COLORS = {"ok": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
STATUS_TINTS = {"ok": "#e8f6e8", "warning": "#fff4d6", "critical": "#fbe4e4"}
STATUS_LABELS = {"ok": "✅ ok", "warning": "⚠️ warning", "critical": "🛑 critical"}
SEVERITY_RANK = {"critical": 0, "warning": 1, "ok": 2}

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color="#0b0b0b"),
    margin=dict(l=48, r=24, t=56, b=48),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    paper_bgcolor="#fcfcfb",
    plot_bgcolor="#fcfcfb",
)
_AXIS = dict(
    gridcolor="#e1e0d9",
    zerolinecolor="#c3c2b7",
    linecolor="#c3c2b7",
    tickfont=dict(color="#898781"),
    title_font=dict(color="#52514e"),
)


def _apply_layout(fig: go.Figure, title: str | None, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(title=dict(text=title or "", x=0, font=dict(size=15)), **_LAYOUT)
    fig.update_xaxes(title_text=x_title, **_AXIS)
    fig.update_yaxes(title_text=y_title, **_AXIS)
    return fig


def numeric_distribution_figure(
    reference: pd.Series,
    current: pd.Series,
    title: str | None = None,
    n_bins: int = 30,
) -> go.Figure:
    """Наложенные гистограммы долей (%) с общими бинами для эталона и батча."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    combined = np.concatenate([ref, cur]) if len(ref) + len(cur) else np.array([0.0, 1.0])
    lo, hi = np.quantile(combined, [0.005, 0.995])  # хвосты не должны схлопывать график
    if not hi > lo:
        lo, hi = float(combined.min()) - 0.5, float(combined.max()) + 0.5
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    fig = go.Figure()
    for values, name, color in (
        (ref, "Эталон", SERIES_REFERENCE),
        (cur, "Текущий батч", SERIES_CURRENT),
    ):
        counts, _ = np.histogram(np.clip(values, lo, hi), bins=edges)
        share = counts / counts.sum() * 100 if counts.sum() else counts.astype(float)
        fig.add_bar(
            x=centers,
            y=share,
            width=width,
            name=name,
            marker_color=color,
            opacity=0.65,
            hovertemplate="%{y:.1f}%<extra>" + name + "</extra>",
        )
    fig.update_layout(barmode="overlay", bargap=0.05)
    return _apply_layout(fig, title, "значение", "доля строк, %")


def categorical_distribution_figure(
    reference: pd.Series,
    current: pd.Series,
    title: str | None = None,
    max_categories: int = 15,
) -> go.Figure:
    """Сгруппированные столбцы долей (%) по категориям; новые категории батча видны."""
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    ref_share = ref.value_counts(normalize=True)
    cur_share = cur.value_counts(normalize=True)

    new_cats = [c for c in cur_share.index if c not in ref_share.index][:5]
    labels = list(ref_share.index[: max(1, max_categories - len(new_cats))]) + new_cats
    ref_vals = ref_share.reindex(labels, fill_value=0.0).tolist()
    cur_vals = cur_share.reindex(labels, fill_value=0.0).tolist()
    ref_rest, cur_rest = 1.0 - sum(ref_vals), 1.0 - sum(cur_vals)
    if ref_rest > 1e-9 or cur_rest > 1e-9:
        labels = [*labels, "прочее"]
        ref_vals.append(max(ref_rest, 0.0))
        cur_vals.append(max(cur_rest, 0.0))

    fig = go.Figure()
    for values, name, color in (
        (ref_vals, "Эталон", SERIES_REFERENCE),
        (cur_vals, "Текущий батч", SERIES_CURRENT),
    ):
        fig.add_bar(
            x=labels,
            y=[v * 100 for v in values],
            name=name,
            marker_color=color,
            hovertemplate="%{y:.1f}%<extra>" + name + "</extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    return _apply_layout(fig, title, "категория", "доля строк, %")


def feature_importance_figure(
    top_features: list[dict],
    title: str | None = "Какие признаки различают эталон и батч",
) -> go.Figure:
    """Горизонтальные столбцы важности признаков adversarial-модели."""
    features = [f["feature"] for f in top_features][::-1]
    values = [float(f["importance"]) * 100 for f in top_features][::-1]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=SERIES_REFERENCE,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    _apply_layout(fig, title, "доля важности, %", "")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def _test_value(tests: list[dict], name: str, field: str = "statistic"):
    for test in tests:
        if test["name"] == name:
            return test.get(field)
    return None


def column_summary_frame(report: dict) -> pd.DataFrame:
    """Плоская таблица по признакам: статус и ключевые метрики, худшие сверху."""
    rows = []
    for col in report.get("columns", []):
        tests = col["tests"]
        pvalue_test = "ks" if col["kind"] == "numeric" else "chi2"
        rows.append(
            {
                "признак": col["column"],
                "тип": "числовой" if col["kind"] == "numeric" else "категориальный",
                "статус": STATUS_LABELS[col["severity"]],
                "PSI": _test_value(tests, "psi"),
                "JS": _test_value(tests, "jensen_shannon"),
                "Вассерштейн (норм.)": _test_value(tests, "wasserstein_norm"),
                "KS / χ²": _test_value(tests, pvalue_test),
                "p-value": _test_value(tests, pvalue_test, "p_value"),
                "_rank": SEVERITY_RANK[col["severity"]],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["_rank", "PSI"], ascending=[True, False], na_position="last")
    return frame.drop(columns="_rank").reset_index(drop=True)


def issues_frame(issues: list[dict]) -> pd.DataFrame:
    """Таблица замечаний схемы и качества данных."""
    rows = [
        {
            "проверка": issue["check"],
            "признак": issue.get("column") or "—",
            "статус": STATUS_LABELS[issue["severity"]],
            "сообщение": issue["message"],
        }
        for issue in issues
    ]
    return pd.DataFrame(rows, columns=["проверка", "признак", "статус", "сообщение"])


def severity_from_label(label: str) -> str:
    for severity, text in STATUS_LABELS.items():
        if str(label) == text:
            return severity
    return "ok"


def style_severity(frame: pd.DataFrame, column: str = "статус"):
    """Подсветка строк: критичные — красным, предупреждения — жёлтым."""

    def _row_style(row):
        tint = STATUS_TINTS[severity_from_label(row.get(column, ""))]
        return [f"background-color: {tint}"] * len(row)

    return frame.style.apply(_row_style, axis=1)
