"""Streamlit-обёртка над общей визуальной системой ``drift_guardian.theme``.

Сниппеты собираются в theme (они же используются в HTML-отчёте), здесь они
оборачиваются в ``st.markdown`` и дополняются CSS, специфичным для Streamlit.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from drift_guardian import theme
from drift_guardian.plots import compact_figure

STREAMLIT_CSS = """
.block-container{padding-top:1.4rem;padding-bottom:40vh;max-width:1240px}
div[data-baseweb="tab-panel"]{min-height:620px}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--dg-ink2);margin:.9rem 0 .2rem}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{line-height:1.35}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--dg-surface)}
button[kind="tertiary"]{color:var(--dg-accent);padding-left:0;font-weight:600}
"""

compact = compact_figure


def inject_css() -> None:
    st.markdown(f"<style>{theme.COMPONENT_CSS}{STREAMLIT_CSS}</style>", unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def topbar(subtitle: str, meta: str) -> None:
    _html(theme.topbar(subtitle, meta))


def hero(severity: str, headline: str, recommendation: str, facts: list[tuple[str, str]]) -> None:
    _html(theme.hero(severity, headline, recommendation, facts))


def section(title: str, meta: str | None = None) -> None:
    _html(theme.section(title, meta))


def heading(title: str, severity: str) -> None:
    """Заголовок панели признака: имя и чип статуса в одну строку."""
    _html(
        '<div class="dg-card-head" style="margin:2px 0 10px">'
        f'<div class="dg-card-name" style="font-size:16px">{escape(title)}</div>{theme.chip(severity)}</div>'
    )


def feature_card_head(name: str, severity: str, why: str, metric_label: str, value: float,
                      warning: float, critical: float) -> None:
    _html(theme.card_head(name, severity, why, metric_label, value, warning, critical))


def issue_list(items: list[tuple[str, str, str]]) -> None:
    _html(theme.issue_list(items))


def empty_state(title: str, steps: list[str]) -> None:
    _html(theme.empty_state(title, steps))


def all_clear(text: str) -> None:
    _html(theme.all_clear(text))


def note(text: str) -> None:
    _html(theme.note(text))
