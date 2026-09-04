"""Мониторинг во времени: серия батчей против эталона и друг против друга.

Продакшн-мониторинг — это не два снимка, а поток: батч за днём, неделей, месяцем.
Модуль режет поток по колонке даты, прогоняет каждый период через ``DriftGuardian``
и собирает компактную сводку: статус периода, PSI по колонкам, число критичных
признаков, ROC-AUC adversarial validation — так видны тренды и момент, когда
дрейф начался.

Два взгляда на поток. Сравнение с **фиксированным эталоном** показывает накопленный
дрейф относительно данных, на которых училась модель. Сравнение с **предыдущим
периодом** (``compare_previous=True``) показывает скачки: медленно плывущий признак
здесь остаётся «в норме», а резкое изменение видно ровно в том периоде, где произошло.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from .config import DriftConfig
from .guardian import DriftGuardian
from .summaries import PeriodSummary, summarize_period

__all__ = [
    "FREQ_LABELS", "PeriodSummary", "summarize_period", "split_by_period",
    "run_timeline", "run_timeline_from_frame",
]

FREQ_LABELS = {"D": "день", "W": "неделя", "M": "месяц", "Q": "квартал"}


def split_by_period(
    frame: pd.DataFrame, date_column: str, freq: str = "M", stats: dict | None = None
) -> list[tuple[str, pd.DataFrame]]:
    """Режет поток на батчи по календарному периоду (D, W, M, Q); колонка даты убирается.

    Строки с нераспознанной датой пропускаются; их число попадает в ``stats``
    (``dropped_rows``, ``total_rows``), если словарь передан.
    """
    if date_column not in frame.columns:
        raise KeyError(f"Колонка даты '{date_column}' не найдена")
    # format="mixed": каждое значение разбирается отдельно, мусор становится NaT без предупреждений.
    dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    mask = dates.notna()
    if stats is not None:
        stats["dropped_rows"] = int((~mask).sum())
        stats["total_rows"] = int(len(frame))
    if not mask.any():
        raise ValueError(f"В колонке '{date_column}' нет распознаваемых дат")
    valid = frame.loc[mask].drop(columns=[date_column])
    periods = dates.loc[mask].dt.to_period(freq)
    return [(str(period), valid.loc[periods == period]) for period in sorted(periods.unique())]


def run_timeline(
    reference: pd.DataFrame,
    batches: list[tuple[str, pd.DataFrame]],
    config: DriftConfig | None = None,
    compare_previous: bool = False,
) -> dict:
    """Прогоняет каждый батч против эталона (и, по запросу, против предыдущего периода)."""
    guardian = DriftGuardian(config)
    summaries: list[PeriodSummary] = []
    previous: pd.DataFrame | None = None
    for label, batch in batches:
        summary = summarize_period(label, guardian.run(reference, batch), len(batch))
        if compare_previous:
            if previous is not None:
                summary.vs_previous = asdict(
                    summarize_period(label, guardian.run(previous, batch), len(batch))
                )
            previous = batch
        summaries.append(summary)
    columns: list[str] = []
    for summary in summaries:
        for column in summary.psi_by_column:
            if column not in columns:
                columns.append(column)
    return {
        "reference_rows": int(len(reference)),
        "periods": [asdict(s) for s in summaries],
        "columns": columns,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "compare_previous": compare_previous,
            "config": guardian.config.to_dict(),
        },
    }


def run_timeline_from_frame(
    reference: pd.DataFrame,
    stream: pd.DataFrame,
    date_column: str,
    freq: str = "M",
    config: DriftConfig | None = None,
    compare_previous: bool = False,
) -> dict:
    """Удобная обёртка: поток с колонкой даты → сводка по периодам."""
    stats: dict = {}
    timeline = run_timeline(
        reference, split_by_period(stream, date_column, freq, stats), config, compare_previous
    )
    timeline["meta"].update(stats)
    return timeline
