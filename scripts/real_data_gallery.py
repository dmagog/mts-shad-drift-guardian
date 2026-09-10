"""Галерея отчётов на реальных открытых данных разной формы (OpenML).

Запуск:
    SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") \\
        python scripts/real_data_gallery.py --out examples/real

Датасеты (кэшируются scikit-learn в ~/scikit_learn_data):
  adult (id 1590, 48 842 строки, 14 признаков, смешанные типы) — случайный сплит (ожидается «в норме»)
      и смещённый сплит «батч = люди старше 50» (ожидается дрейф по возрасту и связанным признакам);
  credit-g (id 31, 1 000 строк, 20 признаков) — маленькие реальные данные, работа защиты от малых выборок;
  electricity (id 151, 45 312 строк, время упорядочено) — начало ряда против конца: настоящий временной дрейф.

Для каждого случая пишется HTML-отчёт и JSON, а также сводка в gallery.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from drift_guardian import DriftConfig, analyze
from drift_guardian.html_report import report_to_json, save_html_report


def load(data_id: int) -> tuple[pd.DataFrame, str]:
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    return frame, bunch.target.name


def cases() -> list[dict]:
    out: list[dict] = []

    adult, target = load(1590)
    adult = adult.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out.append({
        "name": "adult_random_split", "title": "adult: случайный сплит",
        "reference": adult.iloc[:30_000], "current": adult.iloc[30_000:],
        "config": DriftConfig(target_column=target, segment_column="sex"),
    })
    out.append({
        "name": "adult_age_over_50", "title": "adult: батч — люди старше 50",
        "reference": adult[adult["age"] <= 50].iloc[:30_000], "current": adult[adult["age"] > 50],
        "config": DriftConfig(target_column=target, segment_column="sex"),
    })

    credit, target = load(31)
    credit = credit.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out.append({
        "name": "credit_g_small", "title": "credit-g: 700 против 300 строк",
        "reference": credit.iloc[:700], "current": credit.iloc[700:],
        "config": DriftConfig(target_column=target),
    })

    electricity, target = load(151)
    out.append({
        "name": "electricity_begin_vs_end", "title": "electricity: начало ряда против конца",
        "reference": electricity.iloc[:15_000], "current": electricity.iloc[-15_000:],
        "config": DriftConfig(target_column=target, segment_column="day"),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/real")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases():
        started = time.perf_counter()
        report = analyze(case["reference"], case["current"], case["config"])
        elapsed = time.perf_counter() - started
        save_html_report(out / f"{case['name']}.html", report, case["reference"], case["current"], plotlyjs="cdn")
        (out / f"{case['name']}.json").write_text(report_to_json(report), encoding="utf-8")
        drifted = [c["column"] for c in report["columns"] if c["severity"] != "ok"]
        adversarial = report["adversarial"]
        rows.append([
            case["title"], f"{len(case['reference']):,} / {len(case['current']):,}".replace(",", "\u00a0"),
            len(report["columns"]),
            report["overall_severity"], ", ".join(map(str, drifted[:6])) + (" …" if len(drifted) > 6 else "") or "—",
            report["target_drift"]["severity"] if report["target_drift"] else "—",
            f"{adversarial['roc_auc']:.2f}" if adversarial else "—",
            len(report["segments"] or []), f"{elapsed:.1f}",
        ])
        print(f"{case['name']}: {report['overall_severity']} ({elapsed:.1f} с)")

    header = "| Кейс | Строк (эталон / батч) | Признаков | Итог | Признаки с дрейфом | Таргет | Adv. AUC | Сегментов | Время, с |\n|---|---|---|---|---|---|---|---|---|\n"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    (out / "gallery.md").write_text(
        "# Реальные данные: галерея отчётов\n\nСгенерировано `scripts/real_data_gallery.py`. "
        "Датасеты OpenML скачиваются скриптом: [adult](https://www.openml.org/d/1590), "
        "[credit-g](https://www.openml.org/d/31), [electricity](https://www.openml.org/d/151).\n\n"
        + header + body + "\n",
        encoding="utf-8",
    )
    report_copy = Path(__file__).resolve().parents[1] / "report" / "real_data.md"
    report_copy.write_text((out / "gallery.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Готово: {out / 'gallery.md'}")


if __name__ == "__main__":
    main()
