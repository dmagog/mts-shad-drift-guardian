"""Общая визуальная система для дашборда и HTML-отчётов.

Токены (цвета, линии), CSS компонентов и функции, собирающие HTML-сниппеты:
чип статуса, вердикт, шапка карточки признака, список замечаний. Дашборд
оборачивает сниппеты в ``st.markdown``, HTML-отчёт вставляет их в шаблон —
так оба интерфейса выглядят одинаково и правятся в одном месте.

Принципы: один вердикт наверху, ранжированное «что изменилось», статус —
цветом и словом одновременно, всё второстепенное приглушено или спрятано.
"""
from __future__ import annotations

from html import escape

from .plots import STATUS_COLORS, STATUS_LABELS, STATUS_TINTS

AUTHOR = "Георгий Мамарин"
PROJECT_LINE = "Итоговый проект 4.0 Школы аналитиков данных МТС"
REPO_URL = "https://github.com/dmagog/mts-shad-drift-guardian"

# Знак проекта: эталонное распределение (синее) и сдвинутый батч (оранжевый) над осью.
_MARK_BODY = (
    '<rect x="1" y="1" width="30" height="30" rx="8" fill="#fcfcfb" stroke="#e1e0d9"/>'
    '<path d="M5 23 C 9 23, 10 8, 14 8 S 19 23, 23 23" fill="none" stroke="#2a78d6" '
    'stroke-width="2.6" stroke-linecap="round"/>'
    '<path d="M9 23 C 13 23, 14 11, 18 11 S 23 23, 27 23" fill="none" stroke="#eb6834" '
    'stroke-width="2.6" stroke-linecap="round" opacity="0.92"/>'
    '<line x1="4" y1="25.5" x2="28" y2="25.5" stroke="#c3c2b7" stroke-width="1.4" stroke-linecap="round"/>'
)


def mark(size: int = 28) -> str:
    """Инлайн-SVG знака проекта заданного размера."""
    return (
        f'<svg class="dg-mark" width="{size}" height="{size}" viewBox="0 0 32 32" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">{_MARK_BODY}</svg>'
    )


COMPONENT_CSS = """
:root{--dg-ink:#0b0b0b;--dg-ink2:#52514e;--dg-muted:#898781;--dg-line:#e1e0d9;--dg-surface:#ffffff;--dg-page:#f9f9f7;--dg-accent:#2a78d6;--dg-track:#efeee9}
.dg-topbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin:0 0 .8rem}
.dg-wordmark{display:flex;align-items:center;gap:.6rem;font-weight:700;font-size:15px;letter-spacing:.02em;color:var(--dg-ink)}
.dg-wordmark span{color:var(--dg-muted);font-weight:500;letter-spacing:0}
.dg-page-title{font-size:15px;font-weight:650;color:var(--dg-ink);letter-spacing:-.005em}
.dg-meta{color:var(--dg-muted);font-size:12.5px}
.dg-brand{display:flex;align-items:center;gap:10px;margin:.1rem 0 1rem}
.dg-brand-name{font-weight:700;font-size:15px;letter-spacing:.01em;color:var(--dg-ink);line-height:1.2}
.dg-brand-sub{color:var(--dg-muted);font-size:12px;line-height:1.3}
.dg-footer{border-top:1px solid var(--dg-line);margin-top:1.4rem;padding-top:.8rem;color:var(--dg-muted);font-size:12px;line-height:1.5}
.dg-footer b{color:var(--dg-ink2);font-weight:600}
.dg-footer a{color:var(--dg-accent);text-decoration:none}
.dg-hero{border:1px solid var(--dg-line);border-left:6px solid var(--c);border-radius:10px;padding:18px 22px 16px;background:var(--dg-surface);margin:0 0 1.1rem}
.dg-verdict{font-size:22px;font-weight:650;margin:0 0 4px;display:flex;align-items:center;gap:.7rem;letter-spacing:-.01em;flex-wrap:wrap}
.dg-rec{font-size:15px;color:var(--dg-ink2);margin:0 0 12px;max-width:72ch}
.dg-facts{display:flex;flex-wrap:wrap;gap:6px 26px;color:var(--dg-ink2);font-size:13px}
.dg-facts b{color:var(--dg-ink);font-variant-numeric:tabular-nums;font-weight:600;margin-right:.3rem}
.dg-chip{display:inline-flex;align-items:center;gap:.45rem;padding:3px 11px 3px 9px;border-radius:999px;font-size:12.5px;font-weight:600;background:var(--tint);color:var(--dg-ink);white-space:nowrap;line-height:1.3}
.dg-chip::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--c);flex:0 0 8px}
.dg-chip.dg-lg{font-size:14px;padding:5px 14px 5px 11px}
.dg-section{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin:1.5rem 0 .55rem}
.dg-section h2{font-size:17px;font-weight:650;margin:0;letter-spacing:-.01em}
.dg-card{border:1px solid var(--dg-line);border-radius:10px;padding:14px 14px 6px;background:var(--dg-surface)}
.dg-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}
.dg-two{display:grid;grid-template-columns:3fr 2fr;gap:18px;align-items:start}
.dg-card-head{display:flex;justify-content:space-between;align-items:center;gap:.5rem;margin:0 0 4px}
.dg-card-name{font-weight:650;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dg-card-why{color:var(--dg-ink2);font-size:12.5px;min-height:2.7em;line-height:1.35;margin-bottom:6px}
.dg-metric{display:flex;justify-content:space-between;font-size:12.5px;color:var(--dg-ink2);font-variant-numeric:tabular-nums}
.dg-metric b{color:var(--dg-ink);font-weight:600}
.dg-bar{position:relative;height:6px;border-radius:3px;background:var(--dg-track);margin:5px 0 2px}
.dg-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--c)}
.dg-bar s{position:absolute;top:-3px;width:1px;height:12px;background:var(--dg-muted);text-decoration:none}
.dg-list{margin:0;padding:0;list-style:none}
.dg-list li{display:flex;gap:.65rem;align-items:flex-start;padding:7px 0;border-bottom:1px solid var(--dg-line);font-size:13.5px;line-height:1.4}
.dg-list li:last-child{border-bottom:0}
.dg-dot{flex:0 0 8px;width:8px;height:8px;border-radius:50%;margin-top:7px;background:var(--c)}
.dg-list .dg-tag{color:var(--dg-muted);font-size:12px;margin-left:auto;white-space:nowrap;padding-left:1rem}
.dg-empty{border:1px dashed var(--dg-line);border-radius:10px;padding:22px 24px;color:var(--dg-ink2);background:var(--dg-surface);max-width:60ch}
.dg-empty h3{margin:0 0 8px;font-size:16px;color:var(--dg-ink)}
.dg-empty ol{margin:0;padding-left:1.2rem}
.dg-note{color:var(--dg-ink2);font-size:13px;line-height:1.45;max-width:80ch}
.dg-ok{border:1px solid var(--dg-line);border-radius:10px;padding:14px 18px;background:var(--dg-surface);color:var(--dg-ink2);font-size:14px}
"""


def style_attr(severity: str) -> str:
    return f'style="--c:{STATUS_COLORS[severity]};--tint:{STATUS_TINTS[severity]}"'


def chip(severity: str, large: bool = False) -> str:
    cls = "dg-chip dg-lg" if large else "dg-chip"
    return f'<span class="{cls}" {style_attr(severity)}>{STATUS_LABELS[severity]}</span>'


def topbar(subtitle: str, meta: str, wordmark: bool = True) -> str:
    """Шапка страницы: знак и название (HTML-отчёт) или заголовок раздела (дашборд, где бренд в панели)."""
    if wordmark:
        left = f'<div class="dg-wordmark">{mark(24)}Data Drift Guardian<span>{escape(subtitle)}</span></div>'
    else:
        left = f'<div class="dg-page-title">{escape(subtitle)}</div>'
    return f'<div class="dg-topbar">{left}<div class="dg-meta">{escape(meta)}</div></div>'


def brand(subtitle: str = "мониторинг дрейфа данных") -> str:
    """Бренд-блок для боковой панели: знак, название, подпись."""
    return (
        f'<div class="dg-brand">{mark(30)}<div><div class="dg-brand-name">Data Drift Guardian</div>'
        f'<div class="dg-brand-sub">{escape(subtitle)}</div></div></div>'
    )


def footer() -> str:
    """Подвал: автор, проект, ссылка на репозиторий."""
    return (
        f'<div class="dg-footer"><b>Автор — {escape(AUTHOR)}</b><br>{escape(PROJECT_LINE)}<br>'
        f'<a href="{REPO_URL}" target="_blank" rel="noopener">github.com/dmagog/mts-shad-drift-guardian</a></div>'
    )


def hero(severity: str, headline: str, recommendation: str, facts: list[tuple[str, str]]) -> str:
    facts_html = "".join(
        f"<div><b>{escape(str(value))}</b>{escape(label)}</div>" for label, value in facts
    )
    return (
        f'<div class="dg-hero" {style_attr(severity)}>'
        f'<div class="dg-verdict">{chip(severity, large=True)}<span>{escape(headline)}</span></div>'
        f'<p class="dg-rec">{escape(recommendation)}</p>'
        f'<div class="dg-facts">{facts_html}</div></div>'
    )


def section(title: str, meta: str | None = None) -> str:
    right = f'<div class="dg-meta">{escape(meta)}</div>' if meta else ""
    return f'<div class="dg-section"><h2>{escape(title)}</h2>{right}</div>'


def card_head(name: str, severity: str, why: str, metric_label: str, value: float,
              warning: float, critical: float) -> str:
    """Шапка карточки признака: имя, чип, объяснение, метрика с полосой и засечками порогов."""
    cap = max(critical * 2.5, value * 1.15, 1e-9)
    fill = min(value / cap, 1.0) * 100
    marks = "".join(
        f'<s style="left:{min(level / cap, 1.0) * 100:.1f}%"></s>' for level in (warning, critical)
    )
    return (
        f'<div class="dg-card-head"><div class="dg-card-name" title="{escape(name)}">{escape(name)}</div>{chip(severity)}</div>'
        f'<div class="dg-card-why">{escape(why)}</div>'
        f'<div class="dg-metric"><span>{escape(metric_label)}</span><b>{value:.3f}</b></div>'
        f'<div class="dg-bar" {style_attr(severity)}><i style="width:{fill:.1f}%"></i>{marks}</div>'
    )


def issue_list(items: list[tuple[str, str, str]]) -> str:
    """Список замечаний: (severity, текст, ярлык справа)."""
    rows = "".join(
        f'<li><span class="dg-dot" {style_attr(sev)}></span><span>{escape(text)}</span>'
        f'<span class="dg-tag">{escape(tag)}</span></li>'
        for sev, text, tag in items
    )
    return f'<ul class="dg-list">{rows}</ul>'


def empty_state(title: str, steps: list[str]) -> str:
    items = "".join(f"<li>{escape(s)}</li>" for s in steps)
    return f'<div class="dg-empty"><h3>{escape(title)}</h3><ol>{items}</ol></div>'


def all_clear(text: str) -> str:
    return f'<div class="dg-ok">{escape(text)}</div>'


def note(text: str) -> str:
    return f'<p class="dg-note">{escape(text)}</p>'
