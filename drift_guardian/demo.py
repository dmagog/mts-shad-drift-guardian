"""Демо-данные кредитного скоринга с искусственно внесённым дрейфом.

Используется CLI-скриптом ``scripts/generate_demo.py``, дашбордом и notebook
(сценарии строятся прямо в памяти, файлы не нужны). Генерация детерминирована:
одинаковые seed и размеры всегда дают одинаковые данные.

Колонка ``target`` (1 — дефолт) порождается из признаков логистической моделью,
поэтому сценарий ``concept_drift`` меняет связь «признаки → дефолт», не трогая
распределение самих признаков: классический концептуальный дрейф.
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

BASE_INTERCEPT = -2.0     # ≈ 16 % дефолтов в эталоне
SHOCK_INTERCEPT = -0.5    # ≈ 40 % дефолтов при тех же заявках (экономический шок)

SCENARIOS: dict[str, str] = {
    "no_drift": "Без дрейфа — контрольный батч из того же распределения",
    "mean_shift": "Сдвиг среднего — аудитория постарела, доходы выросли на 25%",
    "missing_surge": "Рост пропусков — доля пустого дохода выросла на ~11 п.п.",
    "new_category": "Новая категория — появился канал «партнёрская сеть»",
    "concept_drift": "Концептуальный дрейф — признаки стабильны, но доля дефолтов выросла с ~16% до ~40%",
    "mixed": "Всё вместе — плюс рост разброса сумм кредита",
}


def default_probability(frame: pd.DataFrame, intercept: float = BASE_INTERCEPT) -> np.ndarray:
    """Вероятность дефолта как логистическая функция признаков."""
    income = frame["income"].fillna(frame["income"].median()).clip(lower=1.0)
    z_income = (np.log(income) - 11.0) / 0.5
    z_ratio = (frame["loan_amount"] / income - 1.75) / 0.7
    unemployed = (frame["employment_type"] == "безработный").astype(float)
    z_history = (frame["credit_history_years"] - 7.0) / 4.0
    logit = intercept - 0.6 * z_income + 0.5 * z_ratio + 1.0 * unemployed - 0.3 * z_history
    return 1.0 / (1.0 + np.exp(-logit))


def assign_target(
    frame: pd.DataFrame, rng: np.random.Generator, intercept: float = BASE_INTERCEPT
) -> pd.DataFrame:
    """Добавляет (или пересчитывает) колонку ``target`` по признакам."""
    out = frame.copy()
    out["target"] = (rng.random(len(out)) < default_probability(out, intercept)).astype(int)
    return out


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
    frame = assign_target(frame, rng)
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


def with_concept_drift(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Признаки те же, но дефолтов при тех же заявках заметно больше."""
    return assign_target(frame, rng, intercept=SHOCK_INTERCEPT)


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
    if scenario == "concept_drift":
        return with_concept_drift(frame, rng)
    if scenario == "mixed":
        out = with_mean_shift(frame, rng)
        out = with_missing_surge(out, rng)
        out = with_new_category(out, rng)
        out = with_variance_growth(out, rng)
        return with_concept_drift(out, rng)
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


def make_timeline_demo(
    n_periods: int = 8,
    rows_per_period: int = 3_000,
    seed: int = 42,
    start: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Эталон и поток заявок за ``n_periods`` месяцев с нарастающим дрейфом.

    С каждым месяцем аудитория чуть старше и богаче (плавный дрейф), с пятого
    месяца растут пропуски в доходе, с шестого появляется канал «партнёрская сеть»,
    с седьмого начинается концептуальный дрейф по дефолтам. Поток содержит
    колонку ``date``.
    """
    reference = make_reference(20_000, seed)
    frames = []
    ramp = max(n_periods - 6, 1)
    for i in range(n_periods):
        rng = np.random.default_rng([seed, 100 + i])
        batch = make_base(rng, rows_per_period)
        # Возраст остаётся целым: дробные значения тривиально выдавали бы батч adversarial-модели.
        batch["age"] = np.clip(batch["age"] + 0.9 * i, 18, 85).round(0)
        batch["income"] = (batch["income"] * (1 + 0.035 * i)).round(0)
        if i >= 4:
            batch = with_missing_surge(batch, rng)
        if i >= 5:
            batch.loc[rng.random(len(batch)) < 0.04 * (i - 4), "channel"] = "партнёрская сеть"
        if i >= 6:
            intercept = BASE_INTERCEPT + (SHOCK_INTERCEPT - BASE_INTERCEPT) * (i - 5) / ramp
            batch = assign_target(batch, rng, intercept=intercept)
        month_start = pd.Timestamp(start) + pd.DateOffset(months=i)
        batch.insert(
            0, "date", month_start + pd.to_timedelta(rng.integers(0, 28, len(batch)), unit="D")
        )
        frames.append(batch)
    return reference, pd.concat(frames, ignore_index=True)
