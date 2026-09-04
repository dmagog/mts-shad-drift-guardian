"""Data Ingestion Core: загрузка эталона и батча из CSV / Parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_SUFFIXES = (".csv", ".csv.gz", ".csv.zip", ".tsv")
PARQUET_SUFFIXES = (".parquet", ".pq")


def load_table(path: str | Path) -> pd.DataFrame:
    """Читает таблицу по расширению файла: CSV (в т. ч. сжатый, TSV) или Parquet."""
    source = Path(path)
    name = source.name.lower()
    if not source.exists():
        raise FileNotFoundError(f"Файл не найден: {source}")
    if name.endswith(PARQUET_SUFFIXES):
        return pd.read_parquet(source)
    if name.endswith(".tsv"):
        return pd.read_csv(source, sep="\t")
    if name.endswith(CSV_SUFFIXES):
        return pd.read_csv(source)
    raise ValueError(
        f"Неподдерживаемый формат '{source.suffix}': ожидается CSV или Parquet."
    )
