"""Демо-данные кредитного скоринга с искусственно внесённым дрейфом.

Используется CLI-скриптом ``scripts/generate_demo.py`` и дашбордом (сценарии
строятся прямо в памяти, файлы не нужны). Генерация детерминирована: одинаковые
seed и размеры всегда дают одинаковые данные.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMPLOYMENT = ["наёмный", "ИП", "самозанятый", "безработный"]
EMPLOYMENT_P = [0.70, 0.12, 0.13, 0.05]
REGIONS = ["Москва", "Санкт-Петербург", "Миллионники", "Прочие"]
REGIONS_P = [0.25, 0.15, 0.30, 0.30]
CHANNELS = ["мобильное приложение", "сайт", "офис"]
CHANNELS_P = [0.50, 0.30, 0.20]

SCENARIOS: dict[str, str] = {
    "no_drift": "Без дрейфа — контрольный батч из того же распределения",
    "mean_shift": "Сдвиг среднего — аудитория постарела, доходы выросли на 25%",
    "missing_surge": "Рост пропусков — доля пустого дохода выросла на ~11 п.п.",
    "new_category": "Новая категория — появился канал «партнёрская сеть»",
    "mixed": "Всё вместе — плюс рост разброса сумм кредита",
}


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


def apply_scenario(
    frame: pd.DataFrame, scenario: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Вносит в базовый батч дрейф заданного сценария."""
    if scenario == "no_drift":
        return frame
    if scenario == "mean_shift":
        return with_mean_shift(frame, rng)
    if scenario == "missing_surge":
        return with_missing_surge(frame, rng)
    if scenario == "new_category":
        return with_new_category(frame, rng)
    if scenario == "mixed":
        out = with_mean_shift(frame, rng)
        out = with_missing_surge(out, rng)
        out = with_new_category(out, rng)
        return with_variance_growth(out, rng)
    raise ValueError(f"Неизвестный сценарий: {scenario}. Доступны: {list(SCENARIOS)}")


def make_reference(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Эталонная выборка (на ней «обучалась модель»)."""
    return make_base(np.random.default_rng([seed, 0]), n_rows)


def make_current(scenario: str, n_rows: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Текущий продакшн-батч по сценарию дрейфа."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Неизвестный сценарий: {scenario}. Доступны: {list(SCENARIOS)}")
    stream = list(SCENARIOS).index(scenario) + 1
    rng = np.random.default_rng([seed, stream])
    return apply_scenario(make_base(rng, n_rows), scenario, rng)


def make_demo(
    scenario: str = "mixed",
    ref_rows: int = 20_000,
    cur_rows: int = 5_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пара (эталон, текущий батч) для сценария."""
    return make_reference(ref_rows, seed), make_current(scenario, cur_rows, seed)
