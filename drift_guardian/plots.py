"""Графики Plotly и таблицы для дашборда и HTML-отчёта.

Палитра: две серии «эталон / текущий батч» — синий и оранжевый в фиксированном
порядке (пара проходит проверку на различимость при дальтонизме). Статусы —
зарезервированные цвета ok / warning / critical, всегда вместе со словом
(«в норме», «внимание», «критично»), никогда цветом в одиночку.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES_REFERENCE = "#2a78d6"
SERIES_CURRENT = "#eb6834"
STATUS_COLORS = {"ok": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
STATUS_TINTS = {"ok": "#e8f6e8", "warning": "#fff4d6", "critical": "#fbe4e4"}
STATUS_LABELS = {"ok": "в норме", "warning": "внимание", "critical": "критично"}
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


def _test_value(tests: list[dict], name: str, field: str = "statistic") -> float:
    """Значение теста или NaN, если теста нет (в таблицах NaN отображается пустой ячейкой)."""
    for test in tests:
        if test["name"] == name:
            value = test.get(field)
            return float("nan") if value is None else value
    return float("nan")


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


# ---------- временной ряд ----------

SERIES_ORDER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_RANK_FOR_BAR = {"ok": 1, "warning": 2, "critical": 3}


def _period_view(period: dict, source: str) -> dict | None:
    """Сводка периода относительно эталона (``reference``) или предыдущего периода (``previous``)."""
    return period if source == "reference" else period.get("vs_previous")


def timeline_psi_figure(
    timeline: dict,
    top_n: int = 6,
    psi_warning: float = 0.1,
    psi_critical: float = 0.2,
    title: str | None = "PSI по периодам: признаки с наибольшим дрейфом",
    source: str = "reference",
) -> go.Figure:
    """Линии PSI по периодам для top-N признаков и пунктирные пороги."""
    periods = [p["label"] for p in timeline["periods"]]
    views = [_period_view(p, source) for p in timeline["periods"]]
    max_psi = {
        col: max((v["psi_by_column"].get(col, 0.0) for v in views if v), default=0.0)
        for col in timeline["columns"]
    }
    top = sorted(max_psi, key=max_psi.get, reverse=True)[:top_n]
    fig = go.Figure()
    for color, col in zip(SERIES_ORDER, top, strict=False):
        fig.add_scatter(
            x=periods,
            y=[v["psi_by_column"].get(col) if v else None for v in views],
            mode="lines+markers",
            name=col,
            line=dict(color=color, width=2),
            marker=dict(size=8),
            hovertemplate="%{y:.3f}<extra>" + col + "</extra>",
        )
    for level, label in ((psi_warning, STATUS_LABELS["warning"]), (psi_critical, STATUS_LABELS["critical"])):
        fig.add_hline(
            y=level, line=dict(color="#898781", dash="dash", width=1),
            annotation_text=label, annotation_position="top left",
            annotation_font_color="#898781",
        )
    return _apply_layout(fig, title, "период", "PSI")


def timeline_severity_figure(
    timeline: dict, title: str | None = "Статус по периодам", source: str = "reference"
) -> go.Figure:
    """Столбцы уровня серьёзности по периодам: цвет статуса, подпись в подсказке.

    ``source="previous"`` — статусы относительно предыдущего периода; первый период,
    у которого предыдущего нет, показывается пустым.
    """
    periods = [p["label"] for p in timeline["periods"]]
    views = [_period_view(p, source) for p in timeline["periods"]]
    fig = go.Figure(
        go.Bar(
            x=periods,
            y=[_RANK_FOR_BAR[v["overall_severity"]] if v else 0 for v in views],
            marker_color=[STATUS_COLORS[v["overall_severity"]] if v else "#e1e0d9" for v in views],
            customdata=[
                [STATUS_LABELS[v["overall_severity"]], v["n_critical"], v["n_warning"]]
                if v else ["нет предыдущего периода", 0, 0]
                for v in views
            ],
            hovertemplate="%{customdata[0]}<br>критичных признаков: %{customdata[1]}, "
                          "предупреждений: %{customdata[2]}<extra></extra>",
        )
    )
    _apply_layout(fig, title, "период", "")
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.35)
    fig.update_yaxes(
        tickvals=[1, 2, 3],
        ticktext=[STATUS_LABELS["ok"], STATUS_LABELS["warning"], STATUS_LABELS["critical"]],
        range=[0, 3.4],
    )
    return fig


def timeline_frame(timeline: dict) -> pd.DataFrame:
    """Таблица по периодам для дашборда и отчёта."""
    rows = []
    compare_previous = timeline.get("meta", {}).get("compare_previous", False)
    for p in timeline["periods"]:
        row = {
            "период": p["label"],
            "строк": p["rows"],
            "статус": STATUS_LABELS[p["overall_severity"]],
        }
        if compare_previous:
            previous = p.get("vs_previous")
            row["к предыдущему"] = STATUS_LABELS[previous["overall_severity"]] if previous else "—"
        row.update(
            {
                "таргет": STATUS_LABELS[p["target_severity"]] if p["target_severity"] else "—",
                "critical": p["n_critical"],
                "warning": p["n_warning"],
                "adversarial AUC": p["adversarial_auc"],
                "алертов": len(p["alerts"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compact_figure(fig: go.Figure, height: int = 150, categorical: bool = False) -> go.Figure:
    """Мини-график для карточки: без заголовка, легенды и подписей осей.

    У категориальных графиков подписи категорий скрываются: в карточке они не читаются,
    а что именно изменилось, объясняет текст над графиком.
    """
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=6, b=6 if categorical else 22),
        showlegend=False, title=None, hovermode="closest",
    )
    fig.update_xaxes(title_text=None, tickfont=dict(size=10), showgrid=False,
                     showticklabels=not categorical)
    fig.update_yaxes(title_text=None, visible=False)
    return fig


# ---------- сегменты ----------

_SEQUENTIAL_BLUE = [
    [0.0, "#f0efec"], [0.2, "#cde2fb"], [0.45, "#86b6ef"], [0.7, "#2a78d6"], [1.0, "#104281"],
]


def _short(label: str, limit: int = 18) -> str:
    """Подпись для оси: длинные имена усекаются, полное имя остаётся в подсказке."""
    return label if len(label) <= limit else label[: limit - 1] + "…"


def segment_heatmap_figure(
    segments: list[dict], max_columns: int = 12, title: str | None = None
) -> go.Figure:
    """Тепловая карта PSI «сегмент × признак»: один оттенок, светлее — ближе к нулю."""
    columns: dict[str, float] = {}
    for segment in segments:
        for column, value in segment.get("psi_by_column", {}).items():
            columns[column] = max(columns.get(column, 0.0), float(value))
    top = sorted(columns, key=columns.get, reverse=True)[:max_columns]
    labels = [_short(str(s["label"])) for s in segments]
    z = [[s.get("psi_by_column", {}).get(col) for col in top] for s in segments]
    zmax = max(0.3, max((v for row in z for v in row if v is not None), default=0.3))
    fig = go.Figure(
        go.Heatmap(
            z=z, x=[_short(str(c)) for c in top], y=labels,
            customdata=[[str(c) for c in top]] * len(labels),
            colorscale=_SEQUENTIAL_BLUE, zmin=0, zmax=zmax,
            text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z],
            texttemplate="%{text}", textfont=dict(size=11),
            xgap=2, ygap=2, colorbar=dict(title="PSI", thickness=12, len=0.9),
            hovertemplate="сегмент %{y}<br>%{customdata}: PSI %{z:.3f}<extra></extra>",
        )
    )
    _apply_layout(fig, title, "", "")
    fig.update_layout(
        height=max(180, 70 + 34 * len(labels)), hovermode="closest",
        margin=dict(l=8, r=8, t=30 if title else 8, b=8),
    )
    fig.update_xaxes(showgrid=False, side="top", tickfont=dict(size=11))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def segment_frame(segments: list[dict]) -> pd.DataFrame:
    """Таблица по сегментам для дашборда и отчёта."""
    rows = []
    for s in segments:
        rows.append(
            {
                "сегмент": s["label"],
                "строк (эталон / батч)": f"{s.get('rows_reference', 0)} / {s['rows']}",
                "доля батча": s.get("share_current", 0.0),
                "статус": STATUS_LABELS[s["overall_severity"]],
                "critical": s["n_critical"],
                "warning": s["n_warning"],
                "первый алерт": (s["alerts"][0] if s["alerts"] else s.get("recommendation", ""))[:120],
            }
        )
    return pd.DataFrame(rows)
