"""Граничные сценарии: как система ведёт себя на «неудобных» данных.

Запуск:
    python scripts/edge_cases.py --out report

Пишет report/edge_cases.md — таблицу «сценарий → вердикт → что сообщает система».
Каждый сценарий воспроизводим (seed фиксирован) и покрыт тестом в tests/.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from drift_guardian import DriftConfig, analyze, run_timeline_from_frame
from drift_guardian.demo import make_demo, make_timeline_demo
from drift_guardian.io import read_csv_any


def _verdict(report: dict) -> tuple[str, str]:
    adversarial = report.get("adversarial")
    extra = ""
    if adversarial:
        extra = f"; adversarial AUC {adversarial['roc_auc']:.2f}"
        if adversarial.get("overlap_share"):
            extra += f", совпадений строк {adversarial['overlap_share']:.0%}"
    first_alert = report["alerts"][0] if report["alerts"] else "—"
    return report["overall_severity"], f"{report['recommendation'][:90]}{extra}. Первый алерт: {first_alert[:110]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    ref, cur = make_demo("no_drift", 20_000, 5_000, 42)
    cfg = DriftConfig(target_column="target")
    rows: list[list[str]] = []

    def run(name: str, fn):
        started = time.perf_counter()
        try:
            verdict, note = fn()
        except Exception as exc:  # noqa: BLE001 — граничный сценарий может и упасть, это тоже результат
            verdict, note = "исключение", f"{type(exc).__name__}: {exc}"
        rows.append([name, verdict, note, f"{time.perf_counter() - started:.1f}"])
        print(f"  {name}: {verdict}")

    run("Батч из 10 строк", lambda: _verdict(analyze(ref, cur.head(10), cfg)))
    run("Батч из 30 строк", lambda: _verdict(analyze(ref, cur.head(30), cfg)))
    run("Эталон и батч идентичны", lambda: _verdict(analyze(ref, ref, cfg)))
    mixed = pd.concat([ref.sample(3000, random_state=1), cur.head(2000)], ignore_index=True)
    run("60 % строк батча совпадают с эталоном", lambda: _verdict(analyze(ref, mixed, cfg)))

    dup = cur.copy()
    dup.columns = ["age", "age", *list(cur.columns[2:])]
    ref_dup = ref.copy()
    ref_dup.columns = dup.columns
    run("Дубликаты имён колонок", lambda: _verdict(analyze(ref_dup, dup, cfg)))

    empty = cur.copy()
    empty["income"] = np.nan
    run("Колонка целиком пустая", lambda: _verdict(analyze(ref, empty, cfg)))

    inf = cur.copy()
    inf.loc[:50, "income"] = np.inf
    run("Бесконечности в числовой колонке", lambda: _verdict(analyze(ref, inf, cfg)))

    spaces = cur.copy()
    spaces["region"] = spaces["region"].map(lambda s: s + " ")
    run("Категории с лишними пробелами", lambda: _verdict(analyze(ref, spaces, cfg)))

    wide_ref = ref.copy()
    wide_ref["city"] = rng.integers(0, 500, len(ref)).astype(str)
    wide_cur = cur.copy()
    wide_cur["city"] = rng.integers(0, 500, len(cur)).astype(str)
    run("Категориальная колонка с 500 категориями", lambda: _verdict(analyze(wide_ref, wide_cur, cfg)))

    text_ref, text_cur = ref.copy(), cur.copy()
    text_ref["comment"] = [f"комментарий {i}" for i in range(len(ref))]
    text_cur["comment"] = [f"текст {i}" for i in range(len(cur))]
    run("Колонка со свободным текстом", lambda: (
        analyze(text_ref, text_cur, cfg)["overall_severity"],
        "пропущена: " + analyze(text_ref, text_cur, cfg)["meta"]["skipped_reasons"]["comment"],
    ))

    small_ref, big_cur = make_demo("no_drift", 500, 50_000, 1)
    run("Эталон 500 строк, батч 50 000", lambda: _verdict(analyze(small_ref, big_cur, cfg)))

    wide = pd.DataFrame(rng.normal(0, 1, (20_000, 300)), columns=[f"f{i}" for i in range(300)])
    wide_c = pd.DataFrame(rng.normal(0, 1, (5_000, 300)), columns=wide.columns)
    wide_c["f0"] += 1.0
    run("300 признаков, один сдвинут", lambda: _verdict(analyze(wide, wide_c, DriftConfig())))

    big_r = pd.DataFrame({"x": rng.normal(0, 1, 500_000), "c": rng.choice(list("abcd"), 500_000)})
    big_c = pd.DataFrame({"x": rng.normal(0.2, 1, 200_000), "c": rng.choice(list("abcd"), 200_000)})
    run("500 000 строк эталона, 200 000 батча", lambda: _verdict(analyze(big_r, big_c, DriftConfig())))

    raw = cur.head(200).to_csv(index=False).encode("cp1251")
    run("CSV в кодировке cp1251", lambda: ("ok", f"прочитано {read_csv_any(io.BytesIO(raw)).shape[0]} строк"))

    stream = cur.head(3000).copy()
    stream.insert(0, "date", ["2026-01-05"] * 1000 + ["мусор"] * 1000 + ["2026-02-10"] * 1000)
    run("Поток с мусором в колонке даты", lambda: (
        "ok",
        (lambda t: f"пропущено {t['meta']['dropped_rows']} из {t['meta']['total_rows']} строк, периодов: {len(t['periods'])}")(
            run_timeline_from_frame(ref, stream, "date", "M", DriftConfig(target_column="target", adversarial_enabled=False))
        ),
    ))

    _, tiny_stream = make_timeline_demo(n_periods=2, rows_per_period=300, seed=6)
    tiny_stream = pd.concat([
        tiny_stream[tiny_stream["date"] < "2026-02-01"], tiny_stream[tiny_stream["date"] >= "2026-02-01"].head(12),
    ])
    run("Поток: период из 12 строк", lambda: (
        (lambda p: (p["overall_severity"], f"insufficient={p['insufficient']}, строк {p['rows']}"))(
            run_timeline_from_frame(ref, tiny_stream, "date", "M", DriftConfig(target_column="target", adversarial_enabled=False))["periods"][-1]
        )
    ))

    header = "| Сценарий | Вердикт | Что сообщает система | Время, с |\n|---|---|---|---|\n"
    body = "\n".join("| " + " | ".join(str(c).replace("|", "¦") for c in r) + " |" for r in rows)
    text = (
        "# Граничные сценарии\n\n"
        "Сгенерировано скриптом `scripts/edge_cases.py`; каждый сценарий покрыт тестом в `tests/`.\n\n"
        + header + body + "\n"
    )
    (out / "edge_cases.md").write_text(text, encoding="utf-8")
    print(f"Готово: {out / 'edge_cases.md'}")


if __name__ == "__main__":
    main()
