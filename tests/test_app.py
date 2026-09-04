"""Смоук-тесты дашборда через streamlit.testing (без браузера)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _run_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()
    return app


def test_dashboard_renders_default_scenario():
    app = _run_app()
    assert not app.exception, [e.value for e in app.exception]
    markdown = " ".join(m.value for m in app.markdown)
    assert "Data Drift Guardian" in markdown
    assert "Критический дрейф" in markdown, "сценарий mixed должен давать критический вердикт"
    assert len(app.dataframe) >= 1


def test_dashboard_no_drift_scenario_is_green():
    app = _run_app()
    app.sidebar.selectbox[0].set_value("no_drift").run()
    assert not app.exception, [e.value for e in app.exception]
    markdown = " ".join(m.value for m in app.markdown)
    assert "Дрейф не обнаружен" in markdown, "сценарий no_drift должен давать вердикт «в норме»"


def test_dashboard_timeline_mode_runs():
    app = _run_app()
    app.sidebar.radio[0].set_value("Временной ряд").run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Динамика по периодам" in m.value for m in app.markdown)
    assert len(app.get("plotly_chart")) >= 3  # статусы, PSI по периодам, детализация


def test_feature_card_opens_detail_panel():
    app = _run_app()
    open_buttons = [b for b in app.button if b.label == "Подробнее"]
    assert open_buttons, "у карточек признаков должна быть кнопка «Подробнее»"
    open_buttons[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    markdown = " ".join(m.value for m in app.markdown)
    assert "Признак «" in markdown


def test_wide_data_gets_show_more_and_table_filter(monkeypatch):
    """Много признаков: карточки раскрываются кнопкой, таблица получает поиск и фильтр."""
    import numpy as np
    import pandas as pd

    from drift_guardian import demo

    def wide_demo(scenario: str, ref_rows: int = 20000, cur_rows: int = 5000, seed: int = 42):
        rng = np.random.default_rng(seed)
        ref = pd.DataFrame({f"f{i:02d}": rng.normal(0, 1, 3000) for i in range(20)})
        cur = pd.DataFrame({f"f{i:02d}": rng.normal(0, 1, 1500) for i in range(20)})
        for i in range(8):
            cur[f"f{i:02d}"] += 1.0 + 0.2 * i
        ref["target"], cur["target"] = rng.integers(0, 2, 3000), rng.integers(0, 2, 1500)
        return ref, cur

    monkeypatch.setattr(demo, "make_demo", wide_demo)
    app = AppTest.from_file(str(APP_PATH), default_timeout=240)
    app.run()
    app.sidebar.number_input[0].set_value(7).run()
    assert not app.exception, [e.value for e in app.exception]
    more = [b for b in app.button if b.key == "pair-more"]
    assert more and more[0].label.startswith("Показать ещё")
    more[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert not [b for b in app.button if b.key == "pair-more"], "все 8 карточек показаны"
    assert app.text_input(key="pair-summary-query").value == ""
    app.checkbox(key="pair-summary-drift").check().run()
    assert not app.exception, [e.value for e in app.exception]
    captions = " ".join(c.value for c in app.caption)
    assert "Показано признаков: 8" in captions
