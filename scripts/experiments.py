"""Эксперименты для итогового отчёта.

1. Прогон всех демо-сценариев дрейфа.
2. Калибровка порогов: значения PSI / JS / Вассерштейна при известных сдвигах.
3. Частота ложных срабатываний на батчах без дрейфа (30 сидов) — двухключевое
   правило против «наивного» правила «p-value < α».
4. Реальные данные: OpenML bank-marketing (id 1461), эталон и батч — разные
   месяцы кампании (пропускается, если сети нет).

Запуск:
    python scripts/experiments.py --out report

Пишет report/experiments.md (таблицы для отчёта) и report/experiments.json.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
from scipy import stats

from drift_guardian import DriftConfig, analyze
from drift_guardian.demo import SCENARIOS, make_current, make_demo, make_reference
from drift_guardian.drift_engines import js_from_counts, psi_from_counts, reference_bin_edges

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------- 1. сценарии ----------

def run_scenarios(seed: int) -> tuple[list[dict], str]:
    rows = []
    for name in SCENARIOS:
        ref, cur = make_demo(name, 20_000, 5_000, seed)
        started = time.perf_counter()
        report = analyze(ref, cur, DriftConfig(target_column="target"))
        elapsed = time.perf_counter() - started
        by_severity = {"critical": [], "warning": []}
        for col in report["columns"]:
            if col["severity"] in by_severity:
                by_severity[col["severity"]].append(col["column"])
        rows.append(
            {
                "scenario": name,
                "overall": report["overall_severity"],
                "target": report["target_drift"]["severity"],
                "critical_features": by_severity["critical"],
                "warning_features": by_severity["warning"],
                "quality_alerts": len(report["data_quality"]),
                "adversarial_auc": report["adversarial"]["roc_auc"],
                "seconds": round(elapsed, 2),
                "recommendation": report["recommendation"],
            }
        )
    table = md_table(
        ["Сценарий", "Итог", "Таргет", "Признаки critical", "Признаки warning", "DQ-алерты", "Adv. ROC-AUC", "Время, с"],
        [
            [r["scenario"], r["overall"], r["target"], ", ".join(r["critical_features"]) or "—",
             ", ".join(r["warning_features"]) or "—", r["quality_alerts"], f"{r['adversarial_auc']:.3f}", r["seconds"]]
            for r in rows
        ],
    )
    return rows, table


# ---------- 2. калибровка ----------

def run_calibration(seed: int = 0, n: int = 200_000) -> tuple[dict, str]:
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 1, n)
    edges = reference_bin_edges(ref, 10)
    ref_counts, _ = np.histogram(ref, bins=edges)

    def metrics(cur: np.ndarray) -> tuple[float, float, float]:
        cur_counts, _ = np.histogram(cur, bins=edges)
        return (
            psi_from_counts(ref_counts, cur_counts),
            js_from_counts(ref_counts, cur_counts),
            float(stats.wasserstein_distance(ref, cur) / ref.std()),
        )

    shift_rows = [[d, *[f"{v:.3f}" for v in metrics(rng.normal(d, 1, n))]] for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]]
    scale_rows = [[k, *[f"{v:.3f}" for v in metrics(rng.normal(0, k, n))]] for k in [1.25, 1.5, 2.0, 0.5]]
    base = np.array([0.6, 0.3, 0.1, 0.0])
    new_cat_rows = []
    for s in [0.01, 0.03, 0.05, 0.10]:
        q = np.array([0.6 * (1 - s), 0.3 * (1 - s), 0.1 * (1 - s), s])
        new_cat_rows.append([f"{s:.0%}", f"{psi_from_counts(base * n, q * n):.3f}", f"{js_from_counts(base * n, q * n):.3f}"])
    move_rows = []
    for e in [0.05, 0.10, 0.15, 0.20]:
        q = np.array([0.6 - e, 0.3, 0.1 + e, 0.0])
        move_rows.append([f"{e:.0%}", f"{psi_from_counts(base * n, q * n):.3f}", f"{js_from_counts(base * n, q * n):.3f}"])

    md = "\n\n".join(
        [
            "**Сдвиг среднего нормального признака на δ·σ**\n\n" + md_table(["δ", "PSI", "JS", "W/σ"], shift_rows),
            "**Изменение разброса σ → k·σ**\n\n" + md_table(["k", "PSI", "JS", "W/σ"], scale_rows),
            "**Новая категория с долей s (эталон 0.6 / 0.3 / 0.1)**\n\n" + md_table(["s", "PSI", "JS"], new_cat_rows),
            "**Перенос доли ε из первой категории в третью**\n\n" + md_table(["ε", "PSI", "JS"], move_rows),
        ]
    )
    return {"shift": shift_rows, "scale": scale_rows, "new_category": new_cat_rows, "move": move_rows}, md


# ---------- 3. ложные срабатывания ----------

def run_false_positives(n_seeds: int) -> tuple[dict, str]:
    two_key_alerts = naive_alerts = 0
    two_key_columns = naive_columns = 0
    total_columns = 0
    for seed in range(n_seeds):
        ref = make_reference(20_000, seed)
        cur = make_current("no_drift", 5_000, seed)
        report = analyze(ref, cur, DriftConfig(target_column="target"))
        columns = [*report["columns"], report["target_drift"]]
        significant = [
            any(t.get("details", {}).get("significant") for t in c["tests"]) for c in columns
        ]
        flagged = [c["severity"] != "ok" for c in columns]
        total_columns += len(columns)
        naive_columns += sum(significant)
        two_key_columns += sum(flagged)
        naive_alerts += any(significant)
        two_key_alerts += report["overall_severity"] != "ok"
    result = {
        "n_seeds": n_seeds,
        "naive_batches_flagged": naive_alerts,
        "two_key_batches_flagged": two_key_alerts,
        "naive_columns_flagged": naive_columns,
        "two_key_columns_flagged": two_key_columns,
        "total_columns": total_columns,
    }
    md = md_table(
        ["Правило", "Батчей с ложным алертом", "Колонок с ложным алертом"],
        [
            ["Наивное: p-value < α (с поправкой Бонферрони)", f"{naive_alerts} из {n_seeds}", f"{naive_columns} из {total_columns}"],
            ["Двухключевое (значимость + размер эффекта)", f"{two_key_alerts} из {n_seeds}", f"{two_key_columns} из {total_columns}"],
        ],
    )
    return result, md


# ---------- 4. реальные данные ----------

class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def run_bank_marketing(timeout: int = 180) -> tuple[dict | None, str]:
    """OpenML bank-marketing: эталон — контакты в мае–августе, батч — в сентябре–декабре."""
    try:
        from sklearn.datasets import fetch_openml

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            bunch = fetch_openml(data_id=1461, as_frame=True, parser="auto")
        finally:
            signal.alarm(0)
    except Exception as exc:  # noqa: BLE001 — сеть может быть недоступна, это не ошибка отчёта
        return None, f"Датасет недоступен ({type(exc).__name__}: {exc}); раздел пропущен."

    frame = bunch.frame.copy()
    month_col = next(
        (c for c in frame.columns if set(frame[c].astype(str).str.lower().unique()) <= set(MONTHS)),
        None,
    )
    if month_col is None:
        return None, "Не удалось найти колонку месяца; раздел пропущен."
    month = frame[month_col].astype(str).str.lower()
    reference = frame[month.isin(["may", "jun", "jul", "aug"])].drop(columns=[month_col])
    current = frame[month.isin(["sep", "oct", "nov", "dec"])].drop(columns=[month_col])
    target = bunch.target.name if bunch.target is not None else "class"
    started = time.perf_counter()
    report = analyze(reference, current, DriftConfig(target_column=target))
    elapsed = time.perf_counter() - started
    rows = [
        [c["column"], c["kind"], c["severity"],
         next((f"{t['statistic']:.3f}" for t in c["tests"] if t["name"] == "psi"), "")]
        for c in sorted(report["columns"], key=lambda c: {"critical": 0, "warning": 1, "ok": 2}[c["severity"]])
    ]
    md = "\n\n".join(
        [
            f"Эталон: {len(reference)} строк (май–август), батч: {len(current)} строк (сентябрь–декабрь), "
            f"признаков: {len(report['columns'])}, время анализа: {elapsed:.1f} с.",
            f"Итог: **{report['overall_severity']}**, целевая переменная `{target}`: "
            f"**{report['target_drift']['severity']}**, adversarial ROC-AUC = {report['adversarial']['roc_auc']:.3f}.",
            md_table(["Признак", "Тип", "Статус", "PSI"], rows),
            "Алерты:\n\n" + "\n".join(f"- {a}" for a in report["alerts"]),
        ]
    )
    summary = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "overall": report["overall_severity"],
        "target": report["target_drift"]["severity"],
        "adversarial_auc": report["adversarial"]["roc_auc"],
        "seconds": round(elapsed, 1),
        "columns": {c["column"]: c["severity"] for c in report["columns"]},
        "alerts": report["alerts"],
    }
    return summary, md


# ---------- 5. размер батча: защита от малых выборок ----------

def run_batch_size(n_seeds: int) -> tuple[dict, str]:
    """Ложные срабатывания по колонкам на батчах без дрейфа разного размера, с защитой и без."""
    sizes = [100, 200, 500, 1000, 2000]
    result = {}
    rows = []
    for n in sizes:
        flagged = {True: 0, False: 0}
        batches_flagged = {True: 0, False: 0}
        floors = []
        total = 0
        for seed in range(n_seeds):
            reference = make_reference(20_000, seed)
            current = make_current("no_drift", n, seed)
            for guard in (True, False):
                config = DriftConfig(target_column="target", adversarial_enabled=False, sample_size_guard=guard)
                report = analyze(reference, current, config)
                columns = [*report["columns"], report["target_drift"]]
                bad = sum(c["severity"] != "ok" for c in columns)
                flagged[guard] += bad
                batches_flagged[guard] += bad > 0
                if guard:
                    total += len(columns)
                    floors += [
                        t["details"]["noise_floor"] for c in columns for t in c["tests"]
                        if t["name"] == "psi" and "noise_floor" in t.get("details", {})
                    ]
        result[n] = {
            "psi_noise_floor_median": float(np.median(floors)) if floors else None,
            "columns_flagged_no_guard": flagged[False], "columns_flagged_guard": flagged[True],
            "batches_flagged_no_guard": batches_flagged[False], "batches_flagged_guard": batches_flagged[True],
            "total_columns": total,
        }
        rows.append([
            n, f"{np.median(floors):.3f}" if floors else "—",
            f"{flagged[False]} из {total}", f"{flagged[True]} из {total}",
            f"{batches_flagged[False]} из {n_seeds}", f"{batches_flagged[True]} из {n_seeds}",
        ])
    md = md_table(
        ["Строк в батче", "Шумовой уровень PSI (99 %)", "Ложных колонок без защиты", "с защитой",
         "Батчей с ложным алертом без защиты", "с защитой"],
        rows,
    )
    return result, md


# ---------- 6. калибровка на негауссовых распределениях ----------

def run_nongaussian_calibration(seed: int = 1, n: int = 200_000) -> tuple[dict, str]:
    """PSI / JS / Вассерштейн при сдвиге на 0.3σ и 0.5σ и росте разброса в 1.5 раза
    для разных форм распределения (σ — стандартное отклонение эталона)."""
    rng = np.random.default_rng(seed)
    families = {
        "нормальное": lambda size: rng.normal(0, 1, size),
        "логнормальное (σ=0.5)": lambda size: rng.lognormal(0, 0.5, size),
        "равномерное": lambda size: rng.uniform(0, 1, size),
        "бимодальное": lambda size: np.where(rng.random(size) < 0.5, rng.normal(-1.5, 0.7, size), rng.normal(1.5, 0.7, size)),
        "экспоненциальное": lambda size: rng.exponential(1.0, size),
    }
    rows = []
    result = {}
    for name, sampler in families.items():
        ref = sampler(n)
        sigma = ref.std()
        edges = reference_bin_edges(ref, 10)
        ref_counts, _ = np.histogram(ref, bins=edges)

        def metrics(cur: np.ndarray, ref=ref, sigma=sigma, edges=edges,
                    ref_counts=ref_counts) -> tuple[float, float, float]:
            cur_counts, _ = np.histogram(cur, bins=edges)
            return (
                psi_from_counts(ref_counts, cur_counts),
                js_from_counts(ref_counts, cur_counts),
                float(stats.wasserstein_distance(ref, cur) / sigma),
            )

        variants = {
            "сдвиг 0.3σ": sampler(n) + 0.3 * sigma,
            "сдвиг 0.5σ": sampler(n) + 0.5 * sigma,
            "разброс ×1.5": (lambda x: x.mean() + (x - x.mean()) * 1.5)(sampler(n)),
        }
        result[name] = {}
        for variant, cur in variants.items():
            psi, js, w = metrics(cur)
            result[name][variant] = {"psi": psi, "js": js, "w": w}
            rows.append([name, variant, f"{psi:.3f}", f"{js:.3f}", f"{w:.3f}"])
    md = md_table(["Распределение", "Изменение", "PSI", "JS", "W/σ"], rows)
    return result, md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-seeds", type=int, default=30)
    parser.add_argument("--skip-openml", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1/6 сценарии…")
    scenarios, scenarios_md = run_scenarios(args.seed)
    print("2/6 калибровка…")
    calibration, calibration_md = run_calibration()
    print("3/6 ложные срабатывания…")
    false_positives, fp_md = run_false_positives(args.fp_seeds)
    print("4/6 реальные данные…")
    bank, bank_md = (None, "Пропущено по флагу --skip-openml.") if args.skip_openml else run_bank_marketing()
    print("5/6 размер батча…")
    batch_size, batch_md = run_batch_size(args.fp_seeds)
    print("6/6 негауссовы распределения…")
    nongaussian, nongaussian_md = run_nongaussian_calibration()

    md = "\n\n".join(
        [
            "# Эксперименты Data Drift Guardian",
            "Сгенерировано скриптом `scripts/experiments.py`; все числа воспроизводимы (seed фиксирован).",
            "## 1. Демо-сценарии дрейфа\n\n" + scenarios_md,
            "## 2. Калибровка порогов\n\n" + calibration_md,
            f"## 3. Ложные срабатывания на батчах без дрейфа ({args.fp_seeds} сидов)\n\n" + fp_md,
            "## 4. Реальные данные: OpenML bank-marketing\n\n" + bank_md,
            f"## 5. Размер батча и защита от малых выборок ({args.fp_seeds} сидов на размер)\n\n" + batch_md,
            "## 6. Калибровка на негауссовых распределениях\n\n" + nongaussian_md,
        ]
    )
    (out / "experiments.md").write_text(md + "\n", encoding="utf-8")
    (out / "experiments.json").write_text(
        json.dumps(
            {
                "scenarios": scenarios, "calibration": calibration, "false_positives": false_positives,
                "bank_marketing": bank, "batch_size": batch_size, "nongaussian": nongaussian,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Готово: {out / 'experiments.md'}")


if __name__ == "__main__":
    main()
