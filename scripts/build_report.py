"""Сборка итогового отчёта: report/report.md + report/experiments.md → report/report.html.

Запуск:
    python scripts/build_report.py

В report.md маркер ``<!-- EXPERIMENTS -->`` заменяется содержимым experiments.md
(без заголовка первого уровня, с понижением уровней заголовков), чтобы таблицы
экспериментов в отчёте всегда соответствовали последнему прогону.
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"

CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; background: #f9f9f7; margin: 0; padding: 32px; line-height: 1.5; }
main { max-width: 900px; margin: 0 auto; background: #fcfcfb; padding: 40px 48px; border: 1px solid #e1e0d9; border-radius: 12px; }
h1 { font-size: 28px; margin-top: 0; } h2 { font-size: 21px; margin-top: 36px; border-bottom: 1px solid #e1e0d9; padding-bottom: 6px; }
h3 { font-size: 17px; margin-top: 24px; } h4 { font-size: 15px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 12px 0; }
th, td { padding: 6px 10px; border-bottom: 1px solid #e1e0d9; text-align: left; vertical-align: top; }
th { color: #52514e; font-weight: 600; background: #f0efec; }
code { background: #f0efec; padding: 1px 5px; border-radius: 4px; font-size: 90%; }
pre { background: #f0efec; padding: 12px 14px; border-radius: 8px; overflow-x: auto; font-size: 13px; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #2a78d6; margin: 12px 0; padding: 4px 16px; color: #52514e; }
.muted { color: #52514e; font-size: 13px; }
"""


def _embed(text: str, marker: str, filename: str, demote: int = 1) -> str:
    """Вставляет содержимое markdown-файла вместо маркера, понижая уровни заголовков."""
    path = REPORT_DIR / filename
    if not path.exists():
        return text.replace(marker, f"_Файл {filename} не найден — запустите соответствующий скрипт._")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("# ")]
    body = "\n".join(re.sub(r"^(#+) ", lambda m: "#" * demote + m.group(1) + " ", line) for line in lines)
    return text.replace(marker, body)


def embed_experiments(text: str) -> str:
    text = _embed(text, "<!-- EXPERIMENTS -->", "experiments.md")
    text = _embed(text, "<!-- BENCHMARK -->", "benchmark_evidently.md")
    return _embed(text, "<!-- EDGE_CASES -->", "edge_cases.md")


def main() -> None:
    source = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    source = embed_experiments(source)
    html_body = markdown.markdown(
        source, extensions=["tables", "fenced_code", "toc", "sane_lists"], output_format="html5"
    )
    title = re.search(r"^# (.+)$", source, flags=re.M)
    html = (
        "<!doctype html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{title.group(1) if title else 'Отчёт'}</title>"
        f"<style>{CSS}</style></head><body><main>{html_body}</main></body></html>"
    )
    target = REPORT_DIR / "report.html"
    target.write_text(html, encoding="utf-8")
    print(f"Готово: {target} ({target.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
