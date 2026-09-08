"""Сверка метрик Data Drift Guardian с библиотекой Evidently.

Запускается в отдельном окружении (у Evidently свои требования к зависимостям):
    uv venv .venv-bench && uv pip install --python .venv-bench/bin/python evidently scikit-learn pyyaml scipy
    .venv-bench/bin/python scripts/benchmark_evidently.py --out report

Evidently не обязана давать те же числа: у неё своё разбиение на бины для PSI и JS
и свои пороги по умолчанию (PSI 0.1, JS 0.1, Вассерштейн 0.1, KS/χ² p < 0.05).
Цель сверки — показать, что величины согласованы (KS p-value и Вассерштейн должны
совпасть точно, PSI/JS — по порядку и ранжированию), а решения о дрейфе совпадают
на явных случаях; расхождения объяснены.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from drift_guardian import DriftConfig, analyze  # noqa: E402
from drift_guardian.demo import make_demo  # noqa: E402

try:
    from evidently.legacy.calculations.stattests import (
        chi_stat_test,
        jensenshannon_stat_test,
        ks_stat_test,
        psi_stat_test,
        wasserstein_stat_test,
    )
except ImportError:  # pragma: no cover — старые версии Evidently
    from evidently.calculations.stattests import (  # type: ignore[no-redef]
        chi_stat_test,
        jensenshannon_stat_test,
        ks_stat_test,
        psi_stat_test,
        wasserstein_stat_test,
    )
try:
    from evidently.legacy.core import ColumnType

    NUM, CAT = ColumnType.Numerical, ColumnType.Categorical
except ImportError:  # pragma: no cover
    NUM, CAT = "num", "cat"

import evidently  # noqa: E402

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
NUMERIC_TESTS = {"psi": psi_stat_test, "js": jensenshannon_stat_test,
                 "wasserstein": wasserstein_stat_test, "ks": ks_stat_test}
CATEGORICAL_TESTS = {"psi": psi_stat_test, "js": jensenshannon_stat_test, "chi2": chi_stat_test}


def evidently_scores(ref: pd.Series, cur: pd.Series, kind: str) -> dict:
    tests, ftype = (NUMERIC_TESTS, NUM) if kind == "numeric" else (CATEGORICAL_TESTS, CAT)
    out = {}
    for name, test in tests.items():
        try:
            score, drifted = test.func(ref.dropna(), cur.dropna(), ftype, test.default_threshold)
            out[name] = {"score": float(score), "drifted": bool(drifted),
                         "threshold": float(test.default_threshold)}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return out


def our_scores(column: dict) -> dict:
    tests = {t["name"]: t for t in column["tests"]}
    return {
        "psi": tests.get("psi", {}).get("statistic"),
        "js": tests.get("jensen_shannon", {}).get("statistic"),
        "wasserstein": tests.get("wasserstein_norm", {}).get("statistic"),
        "p_value": (tests.get("ks") or tests.get("chi2") or {}).get("p_value"),
        "severity": column["severity"],
    }


def evidently_default_verdict(scores: dict, kind: str, n_ref: int) -> bool | None:
    """Решение Evidently по её методу по умолчанию: numeric — Вассерштейн (n>1000) или KS;
    categorical — JS (n>1000) или χ²."""
    if kind == "numeric":
        key = "wasserstein" if n_ref > 1000 else "ks"
    else:
        key = "js" if n_ref > 1000 else "chi2"
    return scores.get(key, {}).get("drifted")


def compare(name: str, reference: pd.DataFrame, current: pd.DataFrame, config: DriftConfig) -> dict:
    report = analyze(reference, current, config)
    rows = []
    for col in report["columns"]:
        column = col["column"]
        ours = our_scores(col)
        theirs = evidently_scores(reference[column], current[column], col["kind"])
        rows.append({
            "column": column, "kind": col["kind"], "ours": ours, "evidently": theirs,
            "evidently_default_drifted": evidently_default_verdict(theirs, col["kind"], len(reference)),
        })
    return {"dataset": name, "n_ref": int(len(reference)), "n_cur": int(len(current)), "rows": rows}


def _f(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def _p(value):
    if value is None:
        return "—"
    return "<1e-16" if value < 1e-16 else f"{value:.2g}"


def to_markdown(result: dict) -> str:
    lines = [
        f"### {result['dataset']} (эталон {result['n_ref']}, батч {result['n_cur']})", "",
        "| Признак | Тип | PSI наш | PSI Evid. | JS наш | JS Evid. | W/σ наш | W/σ Evid. | p наш | p Evid. | Вердикт наш | Evidently (метод по умолчанию) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        o, e = r["ours"], r["evidently"]
        p_key = "ks" if r["kind"] == "numeric" else "chi2"
        lines.append(
            f"| {r['column']} | {r['kind']} | {_f(o['psi'])} | {_f(e.get('psi', {}).get('score'))} | "
            f"{_f(o['js'])} | {_f(e.get('js', {}).get('score'))} | {_f(o['wasserstein'])} | "
            f"{_f(e.get('wasserstein', {}).get('score'))} | {_p(o['p_value'])} | "
            f"{_p(e.get(p_key, {}).get('score'))} | {o['severity']} | "
            f"{'дрейф' if r['evidently_default_drifted'] else 'нет' if r['evidently_default_drifted'] is not None else '—'} |"
        )
    return "\n".join(lines)


def summary(results: list[dict]) -> tuple[dict, str]:
    pairs = {"psi": [], "js": [], "wasserstein": [], "p": []}
    agree = total = 0
    for res in results:
        for r in res["rows"]:
            o, e = r["ours"], r["evidently"]
            for key in ("psi", "js", "wasserstein"):
                if o.get(key) is not None and "score" in e.get(key, {}):
                    pairs[key].append((o[key], e[key]["score"]))
            p_key = "ks" if r["kind"] == "numeric" else "chi2"
            if o.get("p_value") is not None and "score" in e.get(p_key, {}):
                pairs["p"].append((o["p_value"], e[p_key]["score"]))
            if r["evidently_default_drifted"] is not None:
                total += 1
                agree += (o["severity"] != "ok") == bool(r["evidently_default_drifted"])
    stats_out = {}
    lines = ["| Метрика | Пар | Spearman ρ | Медиана |наш − Evid.| | Макс. |наш − Evid.| |", "|---|---|---|---|---|"]
    names = {"psi": "PSI", "js": "JS", "wasserstein": "Вассерштейн/σ", "p": "p-value (KS / χ²)"}
    for key, values in pairs.items():
        if len(values) < 3:
            continue
        a, b = np.array(values).T
        rho = stats.spearmanr(a, b).statistic
        diff = np.abs(a - b)
        stats_out[key] = {"n": len(values), "spearman": float(rho), "median_abs_diff": float(np.median(diff)),
                          "max_abs_diff": float(diff.max())}
        lines.append(f"| {names[key]} | {len(values)} | {rho:.3f} | {np.median(diff):.4f} | {diff.max():.4f} |")
    stats_out["decision_agreement"] = {"agree": agree, "total": total}
    lines.append("")
    lines.append(f"Совпадение решений «дрейф / нет» с методом Evidently по умолчанию: **{agree} из {total}** колонок.")
    return stats_out, "\n".join(lines)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def load_bank_marketing(timeout: int = 180):
    from sklearn.datasets import fetch_openml

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        bunch = fetch_openml(data_id=1461, as_frame=True, parser="auto")
    finally:
        signal.alarm(0)
    frame = bunch.frame.copy()
    month_col = next(c for c in frame.columns if set(frame[c].astype(str).str.lower().unique()) <= set(MONTHS))
    month = frame[month_col].astype(str).str.lower()
    reference = frame[month.isin(["may", "jun", "jul", "aug"])].drop(columns=[month_col])
    current = frame[month.isin(["sep", "oct", "nov", "dec"])].drop(columns=[month_col])
    return reference, current, bunch.target.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report")
    parser.add_argument("--skip-openml", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    config = DriftConfig(target_column="target", adversarial_enabled=False)
    for scenario in ["no_drift", "mean_shift", "new_category", "mixed"]:
        print(f"сценарий {scenario}…")
        reference, current = make_demo(scenario, 20_000, 5_000, 42)
        results.append(compare(f"демо: {scenario}", reference, current, config))

    bank_note = ""
    if not args.skip_openml:
        try:
            print("bank-marketing…")
            reference, current, target = load_bank_marketing()
            results.append(compare("bank-marketing (май–август → сентябрь–декабрь)", reference, current,
                                   DriftConfig(target_column=target, adversarial_enabled=False)))
        except Exception as exc:  # noqa: BLE001
            bank_note = f"\n\nbank-marketing недоступен ({type(exc).__name__}: {exc})."

    stats_out, summary_md = summary(results)
    md = "\n\n".join([
        "# Сверка с Evidently",
        f"Evidently {evidently.__version__}, pandas {pd.__version__}. Скрипт: `scripts/benchmark_evidently.py`. Датасет bank-marketing: https://www.openml.org/d/1461 (скачивается скриптом через OpenML). "
        "Значения Evidently получены её собственными функциями статтестов (`psi_stat_test`, "
        "`jensenshannon_stat_test`, `wasserstein_stat_test`, `ks_stat_test`, `chi_stat_test`) "
        "с её порогами по умолчанию: PSI 0.1, JS 0.1, Вассерштейн/σ 0.1, p-value 0.05. "
        "Наш вердикт — итоговая серьёзность колонки (двухключевое правило, откалиброванные пороги).",
        "## Сводка\n\n" + summary_md + bank_note,
        "## По датасетам\n\n" + "\n\n".join(to_markdown(r) for r in results),
    ])
    (out / "benchmark_evidently.md").write_text(md + "\n", encoding="utf-8")
    (out / "benchmark_evidently.json").write_text(
        json.dumps({"summary": stats_out, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Готово: {out / 'benchmark_evidently.md'}")


if __name__ == "__main__":
    main()
