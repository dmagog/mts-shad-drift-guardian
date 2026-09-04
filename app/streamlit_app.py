"""Дашборд Data Drift Guardian.

Запуск:
    streamlit run app/streamlit_app.py

Два режима: «Два батча» (эталон против одного батча) и «Временной ряд» (эталон
против потока батчей по календарным периодам). Слева — источник данных, роли
колонок и пороги; справа — статусы, алерты, графики и детали. «Поплывшие»
признаки подсвечены красным (critical) и жёлтым (warning) — всегда вместе
с иконкой и подписью, чтобы статус читался и без цвета.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from drift_guardian import DriftConfig, Thresholds, analyze, run_timeline, split_by_period
from drift_guardian.demo import SCENARIOS, make_demo, make_timeline_demo
from drift_guardian.html_report import (
    render_html_report,
    render_timeline_html,
    report_to_json,
)
from drift_guardian.plots import (
    STATUS_LABELS,
    categorical_distribution_figure,
    column_summary_frame,
    feature_importance_figure,
    issues_frame,
    numeric_distribution_figure,
    style_severity,
    timeline_frame,
    timeline_psi_figure,
    timeline_severity_figure,
)
from drift_guardian.timeline import FREQ_LABELS

st.set_page_config(page_title="Data Drift Guardian", page_icon="🛡️", layout="wide")

BANNER = {"ok": st.success, "warning": st.warning, "critical": st.error}
SUMMARY_FORMATS = {
    "PSI": st.column_config.NumberColumn(format="%.3f"),
    "JS": st.column_config.NumberColumn(format="%.3f"),
    "Вассерштейн (норм.)": st.column_config.NumberColumn(format="%.3f"),
    "KS / χ²": st.column_config.NumberColumn(format="%.3f"),
    "p-value": st.column_config.NumberColumn(format="%.2e"),
}
TEST_FORMATS = {
    "статистика": st.column_config.NumberColumn(format="%.4f"),
    "p-value": st.column_config.NumberColumn(format="%.2e"),
}
TIMELINE_FORMATS = {"adversarial AUC": st.column_config.NumberColumn(format="%.3f")}
FILE_TYPES = ["csv", "parquet", "pq"]
NO_TARGET = "— нет —"
MODE_PAIR, MODE_STREAM = "Два батча", "Временной ряд"
SPECIAL_TITLES = {"target_drift": "🎯 Целевая переменная", "prediction_drift": "🔮 Предсказания модели"}


# ---------- данные и расчёт (кэшируются) ----------

@st.cache_data(show_spinner="Генерируем демо-данные…")
def load_demo(scenario: str, ref_rows: int, cur_rows: int, seed: int):
    return make_demo(scenario, ref_rows, cur_rows, seed)


@st.cache_data(show_spinner="Генерируем демо-поток…")
def load_timeline_demo(n_periods: int, rows_per_period: int, seed: int):
    return make_timeline_demo(n_periods, rows_per_period, seed)


@st.cache_data(show_spinner="Читаем файл…")
def read_table(data: bytes, name: str) -> pd.DataFrame:
    buffer = io.BytesIO(data)
    if name.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(buffer)
    return pd.read_csv(buffer)


@st.cache_data(show_spinner="Считаем метрики дрейфа…")
def run_analysis(reference: pd.DataFrame, current: pd.DataFrame, config_dict: dict) -> dict:
    return analyze(reference, current, DriftConfig.from_dict(config_dict))


@st.cache_data(show_spinner="Считаем метрики по периодам…")
def run_stream(reference: pd.DataFrame, stream: pd.DataFrame, date_column: str, freq: str,
               config_dict: dict) -> dict:
    batches = split_by_period(stream, date_column, freq)
    return run_timeline(reference, batches, DriftConfig.from_dict(config_dict))


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def distribution_figure(kind: str, reference: pd.Series, current: pd.Series, title: str):
    if kind == "numeric":
        return numeric_distribution_figure(reference, current, title=title)
    return categorical_distribution_figure(reference, current, title=title)


def tests_frame(tests: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "тест": t["name"],
                "статистика": t["statistic"],
                "p-value": t["p_value"],
                "статус": STATUS_LABELS[t["severity"]],
                "порог": t.get("threshold") or "",
            }
            for t in tests
        ]
    )


# ---------- боковая панель ----------

def sidebar():
    st.sidebar.title("Настройки")
    params = st.query_params  # deep-link: ?mode=stream&scenario=concept_drift
    mode = st.sidebar.radio(
        "Режим", [MODE_PAIR, MODE_STREAM], horizontal=True,
        index=1 if params.get("mode") == "stream" else 0,
    )
    st.sidebar.subheader("Данные")
    source = st.sidebar.radio(
        "Источник данных", ["Демо", "Свои файлы"], horizontal=True, label_visibility="collapsed"
    )
    is_demo = source == "Демо"
    reference = current = stream = None
    date_column = None
    freq = "M"
    label = ""

    if mode == MODE_PAIR:
        if is_demo:
            default_scenario = params.get("scenario", "mixed")
            if default_scenario not in SCENARIOS:
                default_scenario = "mixed"
            scenario = st.sidebar.selectbox(
                "Сценарий дрейфа",
                list(SCENARIOS),
                index=list(SCENARIOS).index(default_scenario),
                format_func=lambda s: f"{SCENARIOS[s].split(' — ')[0]} ({s})",
            )
            st.sidebar.caption(SCENARIOS[scenario])
            seed = int(st.sidebar.number_input("Seed генератора", min_value=0, value=42, step=1))
            reference, current = load_demo(scenario, 20_000, 5_000, seed)
            label = f"демо-сценарий «{scenario}», seed {seed}"
        else:
            ref_file = st.sidebar.file_uploader("Эталон (CSV / Parquet)", type=FILE_TYPES)
            cur_file = st.sidebar.file_uploader("Текущий батч (CSV / Parquet)", type=FILE_TYPES)
            if ref_file is not None and cur_file is not None:
                reference = read_table(ref_file.getvalue(), ref_file.name)
                current = read_table(cur_file.getvalue(), cur_file.name)
                label = f"{ref_file.name} → {cur_file.name}"
    else:
        if is_demo:
            n_periods = int(st.sidebar.slider("Периодов (месяцев)", 4, 12, 8))
            seed = int(st.sidebar.number_input("Seed генератора", min_value=0, value=42, step=1))
            reference, stream = load_timeline_demo(n_periods, 3_000, seed)
            date_column, freq = "date", "M"
            st.sidebar.caption(
                "Демо-поток: аудитория постепенно стареет и богатеет, с 5-го месяца растут "
                "пропуски, с 6-го появляется новый канал, с 7-го — концептуальный дрейф."
            )
            label = f"демо-поток на {n_periods} мес., seed {seed}"
        else:
            ref_file = st.sidebar.file_uploader("Эталон (CSV / Parquet)", type=FILE_TYPES)
            stream_file = st.sidebar.file_uploader("Поток с колонкой даты (CSV / Parquet)", type=FILE_TYPES)
            if ref_file is not None and stream_file is not None:
                reference = read_table(ref_file.getvalue(), ref_file.name)
                stream = read_table(stream_file.getvalue(), stream_file.name)
                columns = [str(c) for c in stream.columns]
                guess = next((c for c in columns if "date" in c.lower() or "dt" == c.lower()
                              or "time" in c.lower()), columns[0])
                date_column = st.sidebar.selectbox("Колонка даты", columns, index=columns.index(guess))
                freq = st.sidebar.selectbox(
                    "Период", list(FREQ_LABELS), index=2, format_func=lambda f: FREQ_LABELS[f]
                )
                label = f"{ref_file.name} → поток {stream_file.name} по {FREQ_LABELS[freq]}"

    target_column = None
    exclude_columns: list[str] = []
    if reference is not None:
        st.sidebar.subheader("Роли колонок")
        columns = [str(c) for c in reference.columns]
        default_index = columns.index("target") + 1 if is_demo and "target" in columns else 0
        chosen = st.sidebar.selectbox(
            "Целевая переменная (концептуальный дрейф)", [NO_TARGET, *columns], index=default_index
        )
        target_column = None if chosen == NO_TARGET else chosen
        exclude_columns = st.sidebar.multiselect(
            "Исключить из анализа (идентификаторы, даты)",
            [c for c in columns if c != target_column],
        )

    st.sidebar.subheader("Пороги алертов")
    psi_warning = st.sidebar.slider("PSI — warning", 0.01, 0.50, 0.10, 0.01)
    psi_critical = st.sidebar.slider("PSI — critical", 0.05, 1.00, 0.20, 0.01)
    alpha = st.sidebar.select_slider(
        "α для p-value-тестов (KS, χ²)", options=[0.001, 0.01, 0.05, 0.10], value=0.05
    )
    bonferroni = st.sidebar.checkbox("Поправка Бонферрони на число признаков", value=True)

    st.sidebar.subheader("Adversarial validation")
    adversarial_on = st.sidebar.checkbox("Включить (LightGBM)", value=True)
    auc_warning = st.sidebar.slider("ROC-AUC — warning", 0.50, 0.90, 0.55, 0.01)
    auc_critical = st.sidebar.slider("ROC-AUC — critical", 0.50, 0.95, 0.65, 0.01)

    thresholds = Thresholds(
        psi_warning=psi_warning,
        psi_critical=max(psi_critical, psi_warning),
        alpha=float(alpha),
        adversarial_auc_warning=auc_warning,
        adversarial_auc_critical=max(auc_critical, auc_warning),
    )
    config = DriftConfig(
        thresholds=thresholds,
        bonferroni=bonferroni,
        adversarial_enabled=adversarial_on,
        target_column=target_column,
        exclude_columns=exclude_columns or None,
    )
    st.sidebar.caption("Data Drift Guardian · итоговый проект 4.0 Школы аналитиков данных МТС")
    return {
        "mode": mode, "reference": reference, "current": current, "stream": stream,
        "date_column": date_column, "freq": freq, "config": config, "label": label,
    }


# ---------- блоки детального отчёта ----------

def render_special_blocks(report: dict, reference: pd.DataFrame, current: pd.DataFrame) -> None:
    feature_ok = all(c["severity"] != "critical" for c in report["columns"])
    for key, title in SPECIAL_TITLES.items():
        block = report.get(key)
        if not block:
            continue
        column = block["column"]
        st.subheader(f"{title} «{column}» — {STATUS_LABELS[block['severity']]}")
        left, right = st.columns([3, 2])
        left.plotly_chart(
            distribution_figure(block["kind"], reference[column], current[column], column),
            width="stretch", theme=None,
        )
        right.dataframe(
            style_severity(tests_frame(block["tests"])), width="stretch",
            hide_index=True, column_config=TEST_FORMATS,
        )
        if key == "target_drift" and block["severity"] == "critical" and feature_ok:
            right.warning(
                "Признаки стабильны, а целевая переменная изменилась — вероятен "
                "концептуальный дрейф: изменилась связь «признаки → цель»."
            )


def render_summary_tab(report: dict) -> None:
    frame = column_summary_frame(report)
    if frame.empty:
        st.info("Нет признаков для анализа.")
        return
    st.dataframe(
        style_severity(frame), width="stretch", hide_index=True, column_config=SUMMARY_FORMATS
    )
    st.caption(
        "🛑 critical — сильный сдвиг по метрикам размера эффекта · ⚠️ warning — умеренный сдвиг · "
        "✅ ok — стабильно. p-value-тесты (KS, χ²) засчитываются только вместе с заметным "
        "размером эффекта; на малых батчах порог warning поднимается до шумового уровня."
    )


def render_distributions_tab(report: dict, reference: pd.DataFrame, current: pd.DataFrame) -> None:
    frame = column_summary_frame(report)
    if frame.empty:
        st.info("Нет признаков для анализа.")
        return
    status_by_column = dict(zip(frame["признак"], frame["статус"], strict=True))
    column = st.selectbox(
        "Признак (худшие сверху)",
        frame["признак"].tolist(),
        format_func=lambda c: f"{status_by_column[c]}   {c}",
    )
    col_report = next(c for c in report["columns"] if c["column"] == column)
    st.plotly_chart(
        distribution_figure(col_report["kind"], reference[column], current[column], column),
        width="stretch", theme=None,
    )
    st.dataframe(
        style_severity(tests_frame(col_report["tests"])), width="stretch",
        hide_index=True, column_config=TEST_FORMATS,
    )


def render_quality_tab(report: dict) -> None:
    issues = [*report["schema"], *report["data_quality"]]
    if issues:
        st.dataframe(style_severity(issues_frame(issues)), width="stretch", hide_index=True)
    else:
        st.success("Замечаний к схеме и качеству данных нет.")
    meta = report["meta"]
    if meta.get("excluded_columns"):
        st.caption(f"Исключены по настройке: {', '.join(meta['excluded_columns'])}")


def render_adversarial_tab(report: dict) -> None:
    adversarial = report.get("adversarial")
    if not adversarial:
        st.info(
            "Adversarial validation отключена или данных слишком мало "
            "(нужно не меньше 50 строк в каждой выборке)."
        )
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", f"{adversarial['roc_auc']:.3f}")
    k2.metric("Статус", STATUS_LABELS[adversarial["severity"]])
    k3.metric("Бэкенд", adversarial["backend"])
    k4.metric("Строк использовано", fmt_int(adversarial["n_rows_used"]))
    if adversarial["top_features"]:
        st.plotly_chart(
            feature_importance_figure(adversarial["top_features"]), width="stretch", theme=None
        )
    st.markdown(
        "Классификатор учится отличать эталон от текущего батча. ROC-AUC около 0.5 — выборки "
        "неразличимы; чем выше, тем сильнее изменилась совместная структура признаков. "
        "Важности показывают, какие признаки изменились сильнее всего."
    )


def render_export_tab(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                      config: DriftConfig, timeline: dict | None) -> None:
    st.markdown(
        "HTML — самодостаточный отчёт с интерактивными графиками, "
        "JSON — полный словарь метрик и флагов (выходной контракт системы), "
        "YAML — текущие настройки, чтобы хранить их рядом с моделью."
    )
    inline = st.checkbox("Встроить Plotly.js в HTML (работает офлайн, файл ≈ 4 МБ)", value=False)
    plotlyjs = "inline" if inline else "cdn"
    report_key = report["meta"].get("generated_at", "")
    if st.button("Сформировать HTML-отчёт"):
        html = render_html_report(report, reference, current, plotlyjs=plotlyjs)
        st.session_state["html_report"] = (report_key, html)
    cached = st.session_state.get("html_report")
    if cached and cached[0] == report_key:
        st.download_button(
            "⬇️ Скачать HTML-отчёт", data=cached[1].encode("utf-8"),
            file_name="drift_report.html", mime="text/html",
        )
    st.download_button(
        "⬇️ Скачать JSON", data=report_to_json(report).encode("utf-8"),
        file_name="drift_report.json", mime="application/json",
    )
    if timeline is not None:
        st.download_button(
            "⬇️ Скачать HTML по временному ряду",
            data=render_timeline_html(timeline, plotlyjs=plotlyjs).encode("utf-8"),
            file_name="drift_timeline.html", mime="text/html",
        )
        st.download_button(
            "⬇️ Скачать JSON по временному ряду",
            data=report_to_json(timeline).encode("utf-8"),
            file_name="drift_timeline.json", mime="application/json",
        )
    import yaml

    st.download_button(
        "⬇️ Скачать конфиг YAML",
        data=yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False).encode("utf-8"),
        file_name="drift_config.yaml", mime="application/x-yaml",
    )
    with st.expander("Предпросмотр JSON"):
        st.json(report, expanded=False)


def render_detail(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                  config: DriftConfig, timeline: dict | None = None) -> None:
    severity = report["overall_severity"]
    BANNER[severity](f"**{STATUS_LABELS[severity].upper()}** — {report['recommendation']}")

    meta = report["meta"]
    adversarial = report.get("adversarial")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Строк в эталоне", fmt_int(meta["reference_rows"]))
    k2.metric("Строк в батче", fmt_int(meta["current_rows"]))
    k3.metric("Признаков проверено", len(report["columns"]))
    k4.metric("Алертов", len(report["alerts"]))
    k5.metric("Adversarial ROC-AUC", f"{adversarial['roc_auc']:.3f}" if adversarial else "—")

    if report["alerts"]:
        with st.expander(f"Алерты ({len(report['alerts'])})", expanded=True):
            for alert in report["alerts"]:
                st.markdown(f"- {alert}")
    if meta.get("underpowered_columns"):
        st.info(
            "Батч мал для надёжной оценки PSI/JS по колонкам: "
            f"{', '.join(meta['underpowered_columns'])}. Порог warning поднят до шумового уровня."
        )

    render_special_blocks(report, reference, current)

    tabs = st.tabs(
        ["Сводка по признакам", "Распределения", "Схема и качество", "Adversarial validation", "Экспорт"]
    )
    with tabs[0]:
        render_summary_tab(report)
    with tabs[1]:
        render_distributions_tab(report, reference, current)
    with tabs[2]:
        render_quality_tab(report)
    with tabs[3]:
        render_adversarial_tab(report)
    with tabs[4]:
        render_export_tab(report, reference, current, config, timeline)


def render_timeline_section(timeline: dict, config: DriftConfig) -> str:
    """Графики и таблица по периодам; возвращает выбранный период для детализации."""
    periods = timeline["periods"]
    last = periods[-1]
    BANNER[last["overall_severity"]](
        f"**Последний период {last['label']}: {STATUS_LABELS[last['overall_severity']].upper()}** — "
        f"{last['recommendation']}"
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Периодов", len(periods))
    k2.metric("Критичных периодов", sum(p["overall_severity"] == "critical" for p in periods))
    first_bad = next((p["label"] for p in periods if p["overall_severity"] != "ok"), "—")
    k3.metric("Первый период с дрейфом", first_bad)
    k4.metric("Строк в эталоне", fmt_int(timeline["reference_rows"]))

    left, right = st.columns([2, 3])
    left.plotly_chart(timeline_severity_figure(timeline), width="stretch", theme=None)
    right.plotly_chart(
        timeline_psi_figure(
            timeline, psi_warning=config.thresholds.psi_warning,
            psi_critical=config.thresholds.psi_critical,
        ),
        width="stretch", theme=None,
    )
    st.dataframe(
        style_severity(timeline_frame(timeline)), width="stretch", hide_index=True,
        column_config=TIMELINE_FORMATS,
    )
    labels = [p["label"] for p in periods]
    return st.selectbox("Период для детального разбора", labels, index=len(labels) - 1)


# ---------- страница ----------

def main() -> None:
    state = sidebar()
    reference, config = state["reference"], state["config"]
    st.title("🛡️ Data Drift Guardian")
    st.caption("Детекция дрейфа и контроль качества данных: эталон против текущего продакшн-батча")

    if state["mode"] == MODE_PAIR:
        current = state["current"]
        if reference is None or current is None:
            st.info("Выберите демо-сценарий или загрузите эталон и текущий батч в боковой панели.")
            return
        report = run_analysis(reference, current, config.to_dict())
        render_detail(report, reference, current, config)
    else:
        stream = state["stream"]
        if reference is None or stream is None:
            st.info("Выберите демо-поток или загрузите эталон и поток с колонкой даты.")
            return
        timeline = run_stream(reference, stream, state["date_column"], state["freq"], config.to_dict())
        if not timeline["periods"]:
            st.warning("В потоке не нашлось ни одного периода с данными.")
            return
        st.subheader("📈 Динамика по периодам")
        chosen = render_timeline_section(timeline, config)
        batches = dict(split_by_period(stream, state["date_column"], state["freq"]))
        current = batches[chosen]
        st.subheader(f"Период {chosen}: детальный разбор")
        report = run_analysis(reference, current, config.to_dict())
        render_detail(report, reference, current, config, timeline)

    st.caption(f"Источник: {state['label']} · сформировано {report['meta'].get('generated_at', '')}")


main()
