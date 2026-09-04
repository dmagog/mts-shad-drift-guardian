"""Генерация демо-данных кредитного скоринга с искусственно внесённым дрейфом.

Запуск:
    python scripts/generate_demo.py --out data/demo --seed 42

Создаёт reference.csv (эталон) и current_<сценарий>.csv для сценариев:
    no_drift       — контрольный батч из того же распределения;
    mean_shift     — аудитория постарела, доходы выросли;
    missing_surge  — резкий рост пропусков в доходе;
    new_category   — новый канал заявок, которого не было в эталоне;
    mixed          — всё вместе плюс рост разброса сумм кредита.

Seed фиксирован — данные полностью воспроизводимы.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EMPLOYMENT = ["наёмный", "ИП", "самозанятый", "безработный"]
EMPLOYMENT_P = [0.70, 0.12, 0.13, 0.05]
REGIONS = ["Москва", "Санкт-Петербург", "Миллионники", "Прочие"]
REGIONS_P = [0.25, 0.15, 0.30, 0.30]
CHANNELS = ["мобильное приложение", "сайт", "офис"]
CHANNELS_P = [0.50, 0.30, 0.20]


def make_base(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Базовое распределение заявок на кредит."""
    income = rng.lognormal(mean=11.0, sigma=0.5, size=n).round(0)
    frame = pd.DataFrame(
        {
            "age": np.clip(rng.normal(38, 10, n), 18, 75).round(0),
            "income": income,
            "loan_amount": (income * rng.uniform(0.5, 3.0, n)).round(0),
            "credit_history_years": np.clip(rng.normal(7, 4, n), 0, 40).round(1),
            "num_dependents": rng.integers(0, 4, n),
            "employment_type": rng.choice(EMPLOYMENT, size=n, p=EMPLOYMENT_P),
            "region": rng.choice(REGIONS, size=n, p=REGIONS_P),
            "channel": rng.choice(CHANNELS, size=n, p=CHANNELS_P),
        }
    )
    # Естественный фон пропусков в доходе (как в реальных заявках).
    frame.loc[rng.random(n) < 0.08, "income"] = np.nan
    return frame


def with_mean_shift(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    out["age"] = np.clip(out["age"] + 6, 18, 85)
    out["income"] = (out["income"] * 1.25).round(0)
    return out


def with_missing_surge(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    out.loc[rng.random(len(out)) < 0.12, "income"] = np.nan
    return out


def with_new_category(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    out.loc[rng.random(len(out)) < 0.12, "channel"] = "партнёрская сеть"
    return out


def with_variance_growth(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    center = out["loan_amount"].mean()
    out["loan_amount"] = (center + (out["loan_amount"] - center) * 1.6).round(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo", help="каталог для CSV-файлов")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-rows", type=int, default=20_000)
    parser.add_argument("--cur-rows", type=int, default=5_000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    reference = make_base(rng, args.ref_rows)
    reference.to_csv(out_dir / "reference.csv", index=False)

    scenarios = {
        "no_drift": lambda f: f,
        "mean_shift": lambda f: with_mean_shift(f, rng),
        "missing_surge": lambda f: with_missing_surge(f, rng),
        "new_category": lambda f: with_new_category(f, rng),
        "mixed": lambda f: with_variance_growth(
            with_new_category(with_missing_surge(with_mean_shift(f, rng), rng), rng), rng
        ),
    }

    print(f"reference.csv: {len(reference)} строк")
    for name, transform in scenarios.items():
        batch = transform(make_base(rng, args.cur_rows))
        path = out_dir / f"current_{name}.csv"
        batch.to_csv(path, index=False)
        print(f"{path.name}: {len(batch)} строк")
    print(f"\nГотово. Файлы в {out_dir.resolve()}")


if __name__ == "__main__":
    main()
