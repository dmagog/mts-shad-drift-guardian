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
