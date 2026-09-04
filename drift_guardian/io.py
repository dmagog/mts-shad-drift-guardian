"""Data Ingestion Core: загрузка эталона и батча из CSV / Parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_SUFFIXES = (".csv", ".csv.gz", ".csv.zip", ".tsv")
PARQUET_SUFFIXES = (".parquet", ".pq")
ENCODINGS = ("utf-8", "utf-8-sig", "cp1251")


def read_csv_any(source, **kwargs) -> pd.DataFrame:
    """CSV в UTF-8 (с BOM или без) или cp1251 — частый случай выгрузок из Windows-систем."""
    last_error: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(
        f"Не удалось прочитать CSV: ожидается кодировка UTF-8 или cp1251 ({last_error})"
    )


def load_table(path: str | Path) -> pd.DataFrame:
    """Читает таблицу по расширению файла: CSV (в т. ч. сжатый, TSV) или Parquet."""
    source = Path(path)
    name = source.name.lower()
    if not source.exists():
        raise FileNotFoundError(f"Файл не найден: {source}")
    if name.endswith(PARQUET_SUFFIXES):
        return pd.read_parquet(source)
    if name.endswith(".tsv"):
        return read_csv_any(source, sep="\t")
    if name.endswith(CSV_SUFFIXES):
        return read_csv_any(source)
    raise ValueError(
        f"Неподдерживаемый формат '{source.suffix}': ожидается CSV или Parquet."
    )
