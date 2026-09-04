"""Дашборд Data Drift Guardian.

Запуск:
    streamlit run app/streamlit_app.py

Структура страницы отвечает на три вопроса в порядке важности: есть ли проблема
(вердикт наверху), где именно (ранжированные карточки «что изменилось»), насколько
и что делать (замечания, детали во вкладках, экспорт). Два режима: «Два батча»
и «Временной ряд». Статусы передаются цветом и словом одновременно.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402

from drift_guardian import (  # noqa: E402
    DriftConfig,
    Thresholds,
    analyze,
    run_timeline,
    split_by_period,
)
from drift_guardian.demo import SCENARIOS, make_demo, make_timeline_demo  # noqa: E402
from drift_guardian.html_report import (  # noqa: E402
    render_html_report,
    render_timeline_html,
    report_to_json,
)
from drift_guardian.plots import (  # noqa: E402
    SEVERITY_RANK,
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
from drift_guardian.timeline import FREQ_LABELS  # noqa: E402

st.set_page_config(page_title="Data Drift Guardian", page_icon="🛡️", layout="wide")
ui.inject_css()

SUMMARY_FORMATS = {
    "PSI": st.column_config.NumberColumn(format="%.3f"),
    "JS": st.column_config.NumberColumn(format="%.3f"),
}
TEST_FORMATS = {"статистика": st.column_config.NumberColumn(format="%.4f")}
TIMELINE_FORMATS = {"adversarial AUC": st.column_config.NumberColumn(format="%.3f")}
FILE_TYPES = ["csv", "parquet", "pq"]
NO_TARGET = "нет"
MODE_PAIR, MODE_STREAM = "Два батча", "Временной ряд"
HEADLINES = {"ok": "Дрейф не обнаружен", "warning": "Умеренный дрейф", "critical": "Критический дрейф"}
CHECK_LABELS = {
    "missing_values": "пропуски", "duplicates": "дубликаты", "out_of_range": "диапазон",
    "new_categories": "категории", "constant_column": "константа", "missing_column": "схема",
    "extra_column": "схема", "dtype_mismatch": "схема", "column_skipped": "пропущено",
    "small_batch": "малый батч", "target_column_missing": "таргет",
    "prediction_column_missing": "предсказания",
}
SPECIAL_TITLES = {"target_drift": "Целевая переменная", "prediction_drift": "Предсказания модели"}


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
    return run_timeline(reference, split_by_period(stream, date_column, freq),
                        DriftConfig.from_dict(config_dict))


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def distribution_figure(kind: str, reference: pd.Series, current: pd.Series, title: str | None):
    if kind == "numeric":
        return numeric_distribution_figure(reference, current, title=title)
    return categorical_distribution_figure(reference, current, title=title)


def fmt_p(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return "p < 1e-16" if value < 1e-16 else f"p = {value:.2g}"


def fmt_num(value, digits: int = 3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def tests_frame(tests: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "тест": t["name"],
                "статистика": t["statistic"],
                "p-value": fmt_p(t["p_value"]),
                "статус": STATUS_LABELS[t["severity"]],
                "порог": t.get("threshold") or "",
            }
            for t in tests
        ]
    )


def display_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица для экрана: пустые значения — тире, p-value — текстом."""
    out = frame.copy()
    out["Вассерштейн (норм.)"] = out["Вассерштейн (норм.)"].map(fmt_num)
    out["KS / χ²"] = out["KS / χ²"].map(fmt_num)
    out["p-value"] = out["p-value"].map(fmt_p)
    return out


# ---------- боковая панель ----------

def sidebar() -> dict:
    params = st.query_params  # deep-link: ?mode=stream&scenario=concept_drift
    st.sidebar.markdown("### Режим")
    mode = st.sidebar.radio(
        "Режим", [MODE_PAIR, MODE_STREAM], horizontal=True, label_visibility="collapsed",
        index=1 if params.get("mode") == "stream" else 0,
    )
    st.sidebar.markdown("### Данные")
    source = st.sidebar.radio(
        "Источник данных", ["Демо", "Свои файлы"], horizontal=True, label_visibility="collapsed"
    )
    is_demo = source == "Демо"
    reference = current = stream = None
    date_column, freq, label = None, "M", ""

    if mode == MODE_PAIR:
        if is_demo:
            default_scenario = params.get("scenario", "mixed")
            if default_scenario not in SCENARIOS:
                default_scenario = "mixed"
            scenario = st.sidebar.selectbox(
                "Сценарий", list(SCENARIOS), index=list(SCENARIOS).index(default_scenario),
                format_func=lambda s: SCENARIOS[s].split(" — ")[0],
            )
            st.sidebar.caption(SCENARIOS[scenario])
            seed = int(st.sidebar.number_input("Seed", min_value=0, value=42, step=1))
            reference, current = load_demo(scenario, 20_000, 5_000, seed)
            label = f"демо · {scenario} · seed {seed}"
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
            seed = int(st.sidebar.number_input("Seed", min_value=0, value=42, step=1))
            reference, stream = load_timeline_demo(n_periods, 3_000, seed)
            date_column, freq = "date", "M"
            st.sidebar.caption(
                "Аудитория постепенно стареет и богатеет; с 5-го месяца растут пропуски, "
                "с 6-го появляется новый канал, с 7-го — концептуальный дрейф."
            )
            label = f"демо-поток · {n_periods} мес. · seed {seed}"
        else:
            ref_file = st.sidebar.file_uploader("Эталон (CSV / Parquet)", type=FILE_TYPES)
            stream_file = st.sidebar.file_uploader("Поток с колонкой даты", type=FILE_TYPES)
            if ref_file is not None and stream_file is not None:
                reference = read_table(ref_file.getvalue(), ref_file.name)
                stream = read_table(stream_file.getvalue(), stream_file.name)
                columns = [str(c) for c in stream.columns]
                guess = next((c for c in columns if "date" in c.lower() or c.lower() == "dt"
                              or "time" in c.lower()), columns[0])
                date_column = st.sidebar.selectbox("Колонка даты", columns, index=columns.index(guess))
                freq = st.sidebar.selectbox(
                    "Период", list(FREQ_LABELS), index=2, format_func=lambda f: FREQ_LABELS[f]
                )
                label = f"{ref_file.name} → {stream_file.name} · по {FREQ_LABELS[freq]}"

    target_column = None
    exclude_columns: list[str] = []
    if reference is not None:
        st.sidebar.markdown("### Роли колонок")
        columns = [str(c) for c in reference.columns]
        default_index = columns.index("target") + 1 if is_demo and "target" in columns else 0
        chosen = st.sidebar.selectbox("Целевая переменная", [NO_TARGET, *columns], index=default_index)
        target_column = None if chosen == NO_TARGET else chosen
        exclude_columns = st.sidebar.multiselect(
            "Исключить из анализа", [c for c in columns if c != target_column],
            placeholder="идентификаторы, даты…",
        )

    with st.sidebar.expander("Пороги и параметры"):
        psi_warning = st.slider("PSI, внимание", 0.01, 0.50, 0.10, 0.01)
        psi_critical = st.slider("PSI, критично", 0.05, 1.00, 0.20, 0.01)
        alpha = st.select_slider("α для KS и χ²", options=[0.001, 0.01, 0.05, 0.10], value=0.05)
        bonferroni = st.checkbox("Поправка Бонферрони", value=True)
        adversarial_on = st.checkbox("Adversarial validation (LightGBM)", value=True)
        auc_warning = st.slider("ROC-AUC, внимание", 0.50, 0.90, 0.55, 0.01)
        auc_critical = st.slider("ROC-AUC, критично", 0.50, 0.95, 0.65, 0.01)
        guard = st.checkbox("Защита от малых выборок", value=True)

    thresholds = Thresholds(
        psi_warning=psi_warning, psi_critical=max(psi_critical, psi_warning), alpha=float(alpha),
        adversarial_auc_warning=auc_warning, adversarial_auc_critical=max(auc_critical, auc_warning),
    )
    config = DriftConfig(
        thresholds=thresholds, bonferroni=bonferroni, adversarial_enabled=adversarial_on,
        sample_size_guard=guard, target_column=target_column, exclude_columns=exclude_columns or None,
    )
    st.sidebar.caption("Итоговый проект 4.0 Школы аналитиков данных МТС")
    return {
        "mode": mode, "reference": reference, "current": current, "stream": stream,
        "date_column": date_column, "freq": freq, "config": config, "label": label,
    }


# ---------- вердикт и «что изменилось» ----------

def hero_facts(report: dict) -> list[tuple[str, str]]:
    meta = report["meta"]
    n_cols = len(report["columns"])
    n_drifted = sum(c["severity"] != "ok" for c in report["columns"])
    n_quality = sum(i["severity"] != "ok" for i in [*report["schema"], *report["data_quality"]])
    adversarial = report.get("adversarial")
    facts = [
        (" признаков с дрейфом", f"{n_drifted} из {n_cols}"),
        (" замечаний к данным", str(n_quality)),
        (" adversarial AUC", f"{adversarial['roc_auc']:.2f}" if adversarial else "выкл."),
        (" строк: эталон / батч", f"{fmt_int(meta['reference_rows'])} / {fmt_int(meta['current_rows'])}"),
    ]
    target = report.get("target_drift")
    if target:
        facts.insert(1, (" целевая переменная", STATUS_LABELS[target["severity"]]))
    return facts


def explain_column(col: dict, issues: list[dict]) -> str:
    tests = {t["name"]: t for t in col["tests"]}
    parts: list[str] = []
    if col["kind"] == "numeric":
        w = tests.get("wasserstein_norm", {}).get("statistic")
        if w is not None and w >= 0.1:
            parts.append(f"сдвиг на {w:.2f}σ")
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


def what_changed(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                 config: DriftConfig, key_prefix: str = "pair") -> None:
    drifted = sorted(
        (c for c in report["columns"] if c["severity"] != "ok"),
        key=lambda c: (SEVERITY_RANK[c["severity"]],
                       -next((t["statistic"] for t in c["tests"] if t["name"] == "psi"), 0.0)),
    )
    total = len(report["columns"])
    if not drifted:
        ui.section("Что изменилось")
        ui.all_clear(f"Все {total} признаков стабильны: распределения батча совпадают с эталоном.")
        return
    shown = drifted[:6]
    meta = f"{len(drifted)} из {total} признаков" + (f", показаны {len(shown)}" if len(drifted) > len(shown) else "")
    ui.section("Что изменилось", meta)
    issues_by_column: dict[str, list[dict]] = {}
    for issue in report["data_quality"]:
        if issue.get("column") and issue["severity"] != "ok":
            issues_by_column.setdefault(issue["column"], []).append(issue)
    for row_start in range(0, len(shown), 3):
        row = shown[row_start:row_start + 3]
        cells = st.columns(3)
        for cell, col in zip(cells, row, strict=False):
            name = col["column"]
            psi = next((t["statistic"] for t in col["tests"] if t["name"] == "psi"), 0.0)
            with cell, st.container(border=True):
                ui.feature_card_head(
                    name, col["severity"], explain_column(col, issues_by_column.get(name, [])),
                    "PSI", psi, config.thresholds.psi_warning, config.thresholds.psi_critical,
                )
                fig = ui.compact(
                    distribution_figure(col["kind"], reference[name], current[name], None),
                    categorical=col["kind"] == "categorical",
                )
                st.plotly_chart(fig, width="stretch", theme=None, key=f"{key_prefix}-mini-{name}")


def render_special_blocks(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                          key_prefix: str) -> None:
    feature_ok = all(c["severity"] != "critical" for c in report["columns"])
    for key, title in SPECIAL_TITLES.items():
        block = report.get(key)
        if not block:
            continue
        column = block["column"]
        ui.section(f"{title} «{column}»", STATUS_LABELS[block["severity"]])
        left, right = st.columns([3, 2])
        left.plotly_chart(
            distribution_figure(block["kind"], reference[column], current[column], None),
            width="stretch", theme=None, key=f"{key_prefix}-{key}",
        )
        with right:
            ui.issue_list([
                (
                    t["severity"],
                    f"{t['name']}: {t['statistic']:.4g}"
                    + (f", {fmt_p(t['p_value'])}" if t["p_value"] is not None else ""),
                    STATUS_LABELS[t["severity"]],
                )
                for t in block["tests"] if t["name"] != "skipped"
            ])
            if key == "target_drift" and block["severity"] == "critical" and feature_ok:
                ui.note(
                    "Признаки стабильны, а целевая переменная изменилась: вероятен концептуальный "
                    "дрейф, то есть изменилась связь между признаками и целью."
                )


def render_issues(report: dict) -> None:
    issues = [*report["schema"], *report["data_quality"]]
    alerts = [i for i in issues if i["severity"] != "ok"]
    infos = [i for i in issues if i["severity"] == "ok"]
    if alerts:
        ui.section("Замечания к данным", f"{len(alerts)}")
        ui.issue_list([(i["severity"], i["message"], CHECK_LABELS.get(i["check"], i["check"])) for i in alerts])
    if infos:
        ui.note(" ".join(i["message"] for i in infos))


# ---------- вкладки с деталями ----------

def render_summary_tab(report: dict) -> None:
    frame = column_summary_frame(report)
    if frame.empty:
        ui.note("Нет признаков для анализа.")
        return
    st.dataframe(style_severity(display_summary(frame)), width="stretch", hide_index=True,
                 column_config=SUMMARY_FORMATS)
    ui.note(
        "Критично — сильный сдвиг по метрикам размера эффекта, внимание — умеренный, в норме — стабильно. "
        "KS и χ² засчитываются только вместе с заметным размером эффекта; на малых батчах порог "
        "поднимается до шумового уровня."
    )


def render_distributions_tab(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                             key_prefix: str) -> None:
    frame = column_summary_frame(report)
    if frame.empty:
        ui.note("Нет признаков для анализа.")
        return
    status_by_column = dict(zip(frame["признак"], frame["статус"], strict=True))
    column = st.selectbox(
        "Признак", frame["признак"].tolist(),
        format_func=lambda c: f"{c}  ·  {status_by_column[c]}", key=f"{key_prefix}-dist-select",
    )
    col_report = next(c for c in report["columns"] if c["column"] == column)
    st.plotly_chart(
        distribution_figure(col_report["kind"], reference[column], current[column], None),
        width="stretch", theme=None, key=f"{key_prefix}-dist-{column}",
    )
    st.dataframe(style_severity(tests_frame(col_report["tests"])), width="stretch",
                 hide_index=True, column_config=TEST_FORMATS)


def render_quality_tab(report: dict) -> None:
    issues = [*report["schema"], *report["data_quality"]]
    if issues:
        st.dataframe(style_severity(issues_frame(issues)), width="stretch", hide_index=True)
    else:
        ui.all_clear("Замечаний к схеме и качеству данных нет.")
    meta = report["meta"]
    if meta.get("excluded_columns"):
        ui.note(f"Исключены по настройке: {', '.join(meta['excluded_columns'])}.")


def render_adversarial_tab(report: dict, key_prefix: str) -> None:
    adversarial = report.get("adversarial")
    if not adversarial:
        ui.note("Adversarial validation отключена или данных слишком мало (нужно не меньше 50 строк в каждой выборке).")
        return
    ui.hero(
        adversarial["severity"],
        f"ROC-AUC {adversarial['roc_auc']:.3f}",
        "Классификатор учится отличать эталон от батча: около 0.5 — выборки неразличимы, "
        "чем выше, тем сильнее изменилась совместная структура признаков.",
        [(" бэкенд", adversarial["backend"]), (" строк использовано", fmt_int(adversarial["n_rows_used"]))],
    )
    if adversarial["top_features"]:
        st.plotly_chart(feature_importance_figure(adversarial["top_features"], title=None),
                        width="stretch", theme=None, key=f"{key_prefix}-importance")


def render_export_tab(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                      config: DriftConfig, timeline: dict | None, key_prefix: str) -> None:
    ui.note(
        "HTML — самодостаточный отчёт с интерактивными графиками. JSON — полный словарь метрик "
        "и флагов, выходной контракт системы. YAML — текущие настройки, чтобы хранить их рядом с моделью."
    )
    inline = st.checkbox("Встроить Plotly.js в HTML (работает офлайн, файл около 4 МБ)", value=False,
                         key=f"{key_prefix}-inline")
    plotlyjs = "inline" if inline else "cdn"
    report_key = report["meta"].get("generated_at", "")
    col1, col2, col3 = st.columns(3)
    if col1.button("Сформировать HTML-отчёт", key=f"{key_prefix}-build"):
        st.session_state["html_report"] = (report_key, render_html_report(report, reference, current, plotlyjs=plotlyjs))
    cached = st.session_state.get("html_report")
    if cached and cached[0] == report_key:
        col1.download_button("Скачать HTML-отчёт", data=cached[1].encode("utf-8"),
                             file_name="drift_report.html", mime="text/html", key=f"{key_prefix}-dl-html")
    col2.download_button("Скачать JSON", data=report_to_json(report).encode("utf-8"),
                         file_name="drift_report.json", mime="application/json", key=f"{key_prefix}-dl-json")
    col3.download_button(
        "Скачать конфиг YAML",
        data=yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False).encode("utf-8"),
        file_name="drift_config.yaml", mime="application/x-yaml", key=f"{key_prefix}-dl-yaml",
    )
    if timeline is not None:
        t1, t2, _ = st.columns(3)
        t1.download_button("HTML по временному ряду", data=render_timeline_html(timeline, plotlyjs=plotlyjs).encode("utf-8"),
                           file_name="drift_timeline.html", mime="text/html", key=f"{key_prefix}-dl-thtml")
        t2.download_button("JSON по временному ряду", data=report_to_json(timeline).encode("utf-8"),
                           file_name="drift_timeline.json", mime="application/json", key=f"{key_prefix}-dl-tjson")
    with st.expander("Предпросмотр JSON"):
        st.json(report, expanded=False)


def render_details(report: dict, reference: pd.DataFrame, current: pd.DataFrame,
                   config: DriftConfig, timeline: dict | None, key_prefix: str) -> None:
    ui.section("Подробности")
    tabs = st.tabs(["Все признаки", "Распределения", "Схема и качество", "Adversarial validation", "Экспорт"])
    with tabs[0]:
        render_summary_tab(report)
    with tabs[1]:
        render_distributions_tab(report, reference, current, key_prefix)
    with tabs[2]:
        render_quality_tab(report)
    with tabs[3]:
        render_adversarial_tab(report, key_prefix)
    with tabs[4]:
        render_export_tab(report, reference, current, config, timeline, key_prefix)


def render_batch(report: dict, reference: pd.DataFrame, current: pd.DataFrame, config: DriftConfig,
                 headline: str, timeline: dict | None = None, key_prefix: str = "pair") -> None:
    ui.hero(report["overall_severity"], headline, report["recommendation"], hero_facts(report))
    what_changed(report, reference, current, config, key_prefix)
    render_special_blocks(report, reference, current, key_prefix)
    render_issues(report)
    render_details(report, reference, current, config, timeline, key_prefix)


# ---------- страница ----------

def main() -> None:
    state = sidebar()
    reference, config = state["reference"], state["config"]

    if state["mode"] == MODE_PAIR:
        current = state["current"]
        ui.topbar(state["label"] or "данные не выбраны", "")
        if reference is None or current is None:
            ui.empty_state("Нужны две выборки", [
                "Выберите демо-сценарий в боковой панели, чтобы посмотреть, как это работает.",
                "Или загрузите эталон и текущий батч в формате CSV или Parquet.",
                "Укажите целевую переменную, если она есть: включится проверка концептуального дрейфа.",
            ])
            return
        report = run_analysis(reference, current, config.to_dict())
        render_batch(report, reference, current, config, HEADLINES[report["overall_severity"]])
        return

    stream = state["stream"]
    ui.topbar(state["label"] or "данные не выбраны", "")
    if reference is None or stream is None:
        ui.empty_state("Нужен эталон и поток", [
            "Выберите демо-поток в боковой панели.",
            "Или загрузите эталон и поток с колонкой даты; поток будет разрезан на периоды.",
        ])
        return
    timeline = run_stream(reference, stream, state["date_column"], state["freq"], config.to_dict())
    periods = timeline["periods"]
    if not periods:
        ui.note("В потоке не нашлось ни одного периода с данными.")
        return
    last = periods[-1]
    first_bad = next((p["label"] for p in periods if p["overall_severity"] != "ok"), "нет")
    ui.hero(
        last["overall_severity"],
        f"{HEADLINES[last['overall_severity']]} в последнем периоде {last['label']}",
        last["recommendation"],
        [
            (" периодов", str(len(periods))),
            (" критичных", str(sum(p["overall_severity"] == "critical" for p in periods))),
            (" первый период с дрейфом", first_bad),
            (" строк в эталоне", fmt_int(timeline["reference_rows"])),
        ],
    )
    ui.section("Динамика по периодам")
    left, right = st.columns([2, 3])
    left.plotly_chart(timeline_severity_figure(timeline, title=None), width="stretch", theme=None, key="tl-sev")
    right.plotly_chart(
        timeline_psi_figure(timeline, psi_warning=config.thresholds.psi_warning,
                            psi_critical=config.thresholds.psi_critical, title=None),
        width="stretch", theme=None, key="tl-psi",
    )
    st.dataframe(style_severity(timeline_frame(timeline)), width="stretch", hide_index=True,
                 column_config=TIMELINE_FORMATS)

    labels = [p["label"] for p in periods]
    pick_col, _ = st.columns([1, 3])
    chosen = pick_col.selectbox("Период для разбора", labels, index=len(labels) - 1)
    ui.section(f"Период {chosen}")
    batches = dict(split_by_period(stream, state["date_column"], state["freq"]))
    current = batches[chosen]
    report = run_analysis(reference, current, config.to_dict())
    render_batch(report, reference, current, config, HEADLINES[report["overall_severity"]],
                 timeline=timeline, key_prefix=f"tl-{chosen}")


main()
