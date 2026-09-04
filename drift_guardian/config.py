"""Конфигурация Data Drift Guardian: пороги алертов и параметры анализа."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Thresholds:
    """Пороги срабатывания алертов.

    Значения по умолчанию — общепринятые в индустрии ориентиры:
    PSI < 0.1 — стабильно, 0.1–0.2 — умеренный сдвиг, > 0.2 — сильный сдвиг.
    """

    psi_warning: float = 0.1
    psi_critical: float = 0.2
    # Уровень значимости для KS и хи-квадрат (до поправки Бонферрони).
    alpha: float = 0.05
    # Дистанция Йенсена–Шеннона, диапазон [0, 1].
    js_warning: float = 0.1
    js_critical: float = 0.2
    # Расстояние Вассерштейна, нормированное на std эталона.
    wasserstein_warning: float = 0.1
    wasserstein_critical: float = 0.3
    # ROC-AUC adversarial-валидации: 0.5 означает «выборки неразличимы».
    adversarial_auc_warning: float = 0.55
    adversarial_auc_critical: float = 0.65
    # Прирост доли пропусков в колонке (0.05 = +5 процентных пунктов).
    missing_delta_warning: float = 0.05
    missing_delta_critical: float = 0.15
    # Доля значений текущего батча вне диапазона [min, max] эталона.
    out_of_range_warning: float = 0.01
    out_of_range_critical: float = 0.05
    # Доля строк текущего батча с категориями, которых не было в эталоне.
    new_category_warning: float = 0.01
    new_category_critical: float = 0.05
    # Прирост доли полных дубликатов строк.
    duplicates_delta_warning: float = 0.05
    duplicates_delta_critical: float = 0.15


@dataclass
class DriftConfig:
    """Параметры анализа. Любое значение можно переопределить при создании."""

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
    # Включить adversarial-валидацию (можно отключить ради скорости).
    adversarial_enabled: bool = True
    # Сабсэмплинг для adversarial-валидации (скорость на больших данных).
    adversarial_max_rows: int = 50_000
    # Потоки LightGBM. На небольших данных (и особенно на macOS) многопоточность
    # замедляет обучение в разы из-за накладных расходов, поэтому по умолчанию 1.
    adversarial_n_jobs: int = 1
    random_state: int = 42
    # Явное переопределение типов колонок (имеет приоритет над эвристикой).
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DriftConfig":
        """Обратное к ``to_dict``: восстанавливает конфиг вместе с порогами."""
        data = dict(data)
        thresholds = data.pop("thresholds", None) or {}
        return cls(thresholds=Thresholds(**thresholds), **data)
