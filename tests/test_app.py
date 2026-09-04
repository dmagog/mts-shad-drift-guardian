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
    assert app.title[0].value.endswith("Data Drift Guardian")
    assert app.error, "сценарий mixed должен давать красный баннер"
    assert len(app.dataframe) >= 1


def test_dashboard_no_drift_scenario_is_green():
    app = _run_app()
    app.sidebar.selectbox[0].set_value("no_drift").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.success, "сценарий no_drift должен давать зелёный баннер"


def test_dashboard_timeline_mode_runs():
    app = _run_app()
    app.sidebar.radio[0].set_value("Временной ряд").run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Динамика по периодам" in s.value for s in app.subheader)
    assert len(app.get("plotly_chart")) >= 3  # статусы, PSI по периодам, детализация
