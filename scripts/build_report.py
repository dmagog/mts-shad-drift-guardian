"""Сборка итогового отчёта: report/report.md + report/experiments.md → report/report.html.

Запуск:
    python scripts/build_report.py

В report.md маркер ``<!-- EXPERIMENTS -->`` заменяется содержимым experiments.md
(без заголовка первого уровня, с понижением уровней заголовков), чтобы таблицы
экспериментов в отчёте всегда соответствовали последнему прогону.
"""
from __future__ import annotations

import base64
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
img { display: block; max-width: 100%; height: auto; border: 1px solid #e1e0d9; border-radius: 8px; margin: 14px 0 6px; }
.toc { background: #f6f5f2; border: 1px solid #e1e0d9; border-radius: 10px; padding: 16px 22px 18px; margin: 24px 0 34px; }
.toc .toctitle { display: block; font-weight: 600; font-size: 15px; margin-bottom: 10px; }
.toc ul { list-style: none; margin: 0; padding-left: 0; }
.toc > ul > li { margin: 7px 0; }
.toc ul ul { padding-left: 20px; margin: 4px 0 8px; }
.toc ul ul:has(> li:nth-child(6)) { column-count: 2; column-gap: 28px; }
.toc ul ul li { margin: 2px 0; font-size: 13px; break-inside: avoid; }
.toc a { color: #0b0b0b; text-decoration: none; border-bottom: 1px solid transparent; }
.toc a:hover { border-bottom-color: #2a78d6; }
h2 a.headerlink, h3 a.headerlink { display: none; }
.math { font-family: "STIX Two Math", "Cambria Math", Georgia, "Times New Roman", serif; font-style: italic; white-space: nowrap; }
.math .op { font-style: normal; }
.math sub { font-style: normal; font-size: 75%; }
p.muted { margin-top: 0; }
"""


# --- Математика ------------------------------------------------------------------------
# В report.md формулы записаны LaTeX-ом: так их рендерит GitHub. Подключать MathJax из сети
# в HTML нельзя — отчёт должен открываться одним файлом без интернета, поэтому набор
# конструкций, который реально встречается в тексте, переводится в HTML с Unicode.
_MATH_OPS = ("ln", "max", "min", "log", "KL", "AUC")
_THIN_SPACE = "\u2009"
_MATH_LITERALS = (
    (r"\tfrac{1}{2}", "½"), (r"\tfrac12", "½"), (r"\frac{1}{2}", "½"),
    (r"\approx", "≈"), (r"\cdot", "·"), (r"\times", "×"),
    (r"\le", "≤"), (r"\ge", "≥"), (r"\neq", "≠"),
    (r"\mid", " | "), (r"\|", "‖"), (r"\,", " "),
)


def _sqrt(inner: str) -> str:
    """Корень: над одночленом обходимся без скобок, над выражением — со скобками."""
    return f"√{inner}" if re.fullmatch(r"[A-Za-z0-9]+", inner) else f"√({inner})"


def _math_to_html(source: str) -> str:
    """LaTeX-подмножество отчёта → HTML: индексы тегами, знаки Unicode, функции прямым шрифтом."""
    text = source
    for latex, symbol in _MATH_LITERALS:
        text = text.replace(latex, symbol)
    text = text.replace("-", "−")                          # знак минус, а не дефис
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\sqrt\{([^}]*)\}", lambda m: _sqrt(m.group(1)), text)
    text = re.sub(r"\\sum_([A-Za-z])", r"Σ<sub>\1</sub>", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)           # оставшиеся команды: \ln, \max…
    text = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text)
    text = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", text)
    text = re.sub(r" {2,}", " ", text)
    for op in _MATH_OPS:
        text = re.sub(rf"(?<![\w>]){op}(?![\w<])", f'<span class="op">{op}</span>', text)
    text = text.replace(")<span", ")" + _THIN_SPACE + "<span")   # (a − b) ln(...)
    return f'<span class="math">{text}</span>'


def stash_math(text: str) -> tuple[str, list[str]]:
    """Прячет формулы за плейсхолдеры до markdown: иначе он съедает подчёркивания индексов."""
    stash: list[str] = []

    def take(match: re.Match) -> str:
        stash.append(_math_to_html(" ".join(match.group(1).split())))
        return f"MATHSTASH{len(stash) - 1}ENDMATH"

    # Блоки кода не трогаем: `$(python -c …)` в инструкции запуска — не формула.
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i, part in enumerate(parts):
        if not part.startswith("```"):
            # Формула может быть разорвана переносом строки — переносы схлопываем в пробел.
            parts[i] = re.sub(r"\$([^$`]{1,300}?)\$", take, part, flags=re.S)
    return "".join(parts), stash


def restore_math(html: str, stash: list[str]) -> str:
    for index, rendered in enumerate(stash):
        html = html.replace(f"MATHSTASH{index}ENDMATH", rendered)
    return html


def _inline_images(html: str) -> str:
    """Встраивает картинки как data URI: отчёт остаётся одним самодостаточным файлом."""

    def replace(match: re.Match) -> str:
        path = (REPORT_DIR / match.group(1)).resolve()
        if not path.exists():
            return match.group(0)
        mime = "image/gif" if path.suffix == ".gif" else "image/png"
        return f'src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode("ascii")}"'

    return re.sub(r'src="([^"]+\.(?:png|gif|jpe?g))"', replace, html)


# --- Типографика ------------------------------------------------------------------------
# Неразрывные пробелы ставит сборщик, а не автор: правило накрывает и сгенерированные
# таблицы, а исходник остаётся удобным для правок и поиска.
NBSP = " "
NNBSP = " "          # узкий неразрывный — для разрядов чисел
_SHORT = "вкосуиаяВКОСУИАЯ"


def typography(text: str) -> str:
    """Расставляет неразрывные пробелы; блоки кода, код-спаны и HTML-теги не трогает."""
    parts = re.split(r"(```.*?```|`[^`\n]+`|<[^>]+>)", text, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith(("```", "`", "<")):
            continue
        part = re.sub(rf"(?<![^\s(«])([{_SHORT}])[ \t]+", rf"\1{NBSP}", part)   # предлоги и союзы
        part = re.sub(r"[ \t]+—", f"{NBSP}—", part)                             # перед тире
        part = re.sub(r"(\d)[ \t]+(\d{3})(?!\d)", rf"\1{NNBSP}\2", part)        # разряды
        part = re.sub(r"(\d)[ \t]+(%|с|мин|ч|ГБ|тыс)\b", rf"\1{NBSP}\2", part)  # единицы
        part = part.replace("№ ", f"№{NBSP}")
        part = part.replace("т. ч.", f"т.{NBSP}ч.").replace("п. п.", f"п.{NBSP}п.")
        parts[i] = part
    return "".join(parts)


def _embed_text(text: str, marker: str, body: str, demote: int = 1) -> str:
    """Вставляет готовый markdown вместо маркера, понижая уровни заголовков."""
    lines = [line for line in body.splitlines() if not line.startswith("# ")]
    shifted = "\n".join(re.sub(r"^(#+) ", lambda m: "#" * demote + m.group(1) + " ", line) for line in lines)
    return text.replace(marker, shifted)


def _embed(text: str, marker: str, filename: str, demote: int = 1) -> str:
    """Вставляет содержимое markdown-файла вместо маркера, понижая уровни заголовков."""
    path = REPORT_DIR / filename
    if not path.exists():
        return text.replace(marker, f"_Файл {filename} не найден — запустите соответствующий скрипт._")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("# ")]
    body = "\n".join(re.sub(r"^(#+) ", lambda m: "#" * demote + m.group(1) + " ", line) for line in lines)
    return text.replace(marker, body)


def _embed_benchmark(text: str) -> str:
    """Сверка с Evidently: сводка остаётся в разделе 7, поколоночные таблицы уходят в приложение.

    Файл ``benchmark_evidently.md`` при этом остаётся цельным самостоятельным документом —
    делится только его подача в отчёте.
    """
    path = REPORT_DIR / "benchmark_evidently.md"
    if not path.exists():
        return text.replace("<!-- BENCHMARK -->", "_Файл benchmark_evidently.md не найден._").replace(
            "<!-- BENCHMARK_TABLES -->", ""
        )
    body = path.read_text(encoding="utf-8")
    marker = "\n## По датасетам\n"
    head, _, tables = body.partition(marker)
    text = _embed_text(text, "<!-- BENCHMARK -->", head)
    return _embed_text(text, "<!-- BENCHMARK_TABLES -->", tables.strip())


def embed_experiments(text: str) -> str:
    text = _embed(text, "<!-- EXPERIMENTS -->", "experiments.md")
    text = _embed_benchmark(text)
    text = _embed(text, "<!-- EDGE_CASES -->", "edge_cases.md")
    return _embed(text, "<!-- REAL_DATA -->", "real_data.md", demote=2)


def main() -> None:
    source = (REPORT_DIR / "report.md").read_text(encoding="utf-8")
    source = typography(embed_experiments(source))
    source, math = stash_math(source)
    source = source.replace("<!-- TOC -->", "[TOC]")
    html_body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"title": "Содержание", "toc_depth": "2-3"}},
        output_format="html5",
    )
    html_body = restore_math(_inline_images(html_body), math)
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
