"""Конфигурация Data Drift Guardian: пороги алертов, параметры анализа и контракт данных."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Thresholds:
    """Пороги срабатывания алертов.

    Пороги метрик размера эффекта откалиброваны друг относительно друга
    (README, раздел «Калибровка порогов»): при сдвиге среднего нормального признака
    примерно на 0.3σ все три метрики (PSI, JS, Вассерштейн) дают warning,
    примерно на 0.5σ — critical. PSI < 0.1 / 0.1–0.2 / > 0.2 — общепринятая шкала.
    """

    psi_warning: float = 0.1
    psi_critical: float = 0.2
    # Уровень значимости для KS и хи-квадрат (до поправки Бонферрони).
    alpha: float = 0.05
    # Дистанция Йенсена–Шеннона, диапазон [0, 1].
    js_warning: float = 0.1
    js_critical: float = 0.2
    # Расстояние Вассерштейна, нормированное на std эталона (= сдвиг среднего в σ).
    wasserstein_warning: float = 0.3
    wasserstein_critical: float = 0.5
    # ROC-AUC adversarial-валидации: 0.5 означает «выборки неразличимы».
    adversarial_auc_warning: float = 0.55
    adversarial_auc_critical: float = 0.65
    # Прирост доли пропусков в колонке (0.05 = +5 процентных пунктов).
    missing_delta_warning: float = 0.05
    missing_delta_critical: float = 0.15
    # Доля значений текущего батча вне допустимого диапазона.
    out_of_range_warning: float = 0.01
    out_of_range_critical: float = 0.05
    # Доля строк текущего батча с категориями вне допустимого множества.
    new_category_warning: float = 0.01
    new_category_critical: float = 0.05
    # Прирост доли полных дубликатов строк.
    duplicates_delta_warning: float = 0.05
    duplicates_delta_critical: float = 0.15


@dataclass
class DriftConfig:
    """Параметры анализа и контракт данных. Любое значение можно переопределить."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    # Число квантильных бинов (строятся по эталону) для PSI и JS.
    n_bins: int = 10
    # Поправка Бонферрони на множественные сравнения для p-value-тестов.
    bonferroni: bool = True
    # Категории сверх лимита (по частоте в эталоне) группируются в «прочее».
    max_categories: int = 20
    # Целочисленные колонки с nunique <= лимита трактуются как категориальные.
    categorical_int_unique_limit: int = 10
    # Минимум непустых значений в колонке для запуска статтестов.
    min_samples: int = 20
    # Защита от малых выборок: бутстреп шумового уровня PSI/JS, порог warning
    # поднимается до 95-го процентиля метрики при отсутствии дрейфа.
    sample_size_guard: bool = True
    bootstrap_samples: int = 200
    # Квантиль шума: порог warning поднимается до этого процентиля метрики без дрейфа,
    # т. е. доля ложных срабатываний на колонку не выше (1 − noise_quantile).
    noise_quantile: float = 0.99
    # Автодетекция идентификаторов: доля уникальных значений и минимум строк.
    identifier_unique_share: float = 0.98
    identifier_min_rows: int = 100
    # Включить adversarial-валидацию (можно отключить ради скорости).
    adversarial_enabled: bool = True
    # Сабсэмплинг для adversarial-валидации (скорость на больших данных).
    adversarial_max_rows: int = 50_000
    # Потоки LightGBM: None — автоматически (1 поток на небольших данных, где
    # многопоточность только мешает; 4 потока, когда строк × колонок ≥ 1 млн).
    adversarial_n_jobs: int | None = None
    random_state: int = 42
    # Явное переопределение типов колонок (имеет приоритет над эвристикой).
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None

    # --- роли колонок и контракт данных ---
    # Колонки, которые не анализируются вовсе: идентификаторы, timestamps и т. п.
    exclude_columns: list[str] | None = None
    # Целевая переменная: анализируется отдельным блоком (концептуальный дрейф),
    # в признаки и adversarial-валидацию не входит.
    target_column: str | None = None
    # Предсказания модели: отдельный блок «дрейф предсказаний».
    prediction_column: str | None = None
    # Допустимые диапазоны значений по контракту; если не заданы — берётся [min, max] эталона.
    value_bounds: dict[str, tuple[float, float]] | None = None
    # Допустимые категории по контракту; если не заданы — множество категорий эталона.
    allowed_categories: dict[str, list[Any]] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> DriftConfig:
        """Обратное к ``to_dict``: восстанавливает конфиг вместе с порогами."""
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Неизвестные параметры конфигурации: {unknown}")
        thresholds = data.pop("thresholds", None) or {}
        if data.get("value_bounds"):
            data["value_bounds"] = {
                str(col): (float(lo), float(hi)) for col, (lo, hi) in data["value_bounds"].items()
            }
        return cls(thresholds=Thresholds(**thresholds), **data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DriftConfig:
        """Загружает конфиг из YAML-файла (пример — ``examples/config.yaml``)."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_yaml(self, path: str | Path) -> Path:
        """Сохраняет конфиг в YAML — удобно хранить пороги рядом с моделью."""
        payload = self.to_dict()
        if payload.get("value_bounds"):
            payload["value_bounds"] = {k: list(v) for k, v in payload["value_bounds"].items()}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return target
