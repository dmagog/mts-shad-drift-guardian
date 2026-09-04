# Data Drift Guardian

Система детекции дрейфа данных и контроля их качества для ML-моделей в продакшене.
Сравнивает эталонную выборку (на которой обучалась модель) с текущим продакшн-батчем
и выдаёт структурированный отчёт с метриками и алертами — без нейросетей, на строгой
математической статистике и классическом ML.

Итоговый проект «4.0 Школы аналитиков данных» МТС, задача №4. Заказчик — Бояджи Владислав.

## Что умеет

| Модуль | Что делает |
|---|---|
| **Schema Validation** (`schema.py`) | Сверяет набор колонок и семейства типов: пропавшие/новые колонки, смена типа. |
| **Data Quality** (`data_quality.py`) | Рост доли пропусков, дубликаты строк, выход за диапазон `[min, max]` эталона, новые категории, константные колонки. |
| **Drift Engines** (`drift_engines.py`) | Числовые: KS-тест, PSI, дистанция Йенсена–Шеннона, расстояние Вассерштейна. Категориальные: хи-квадрат, PSI, Йенсен–Шеннон. |
| **Adversarial Validation** (`adversarial.py`) | LightGBM учится отличать эталон от батча; ROC-AUC ≫ 0.5 — дрейф подтверждён, feature importance показывает, что изменилось сильнее всего. |
| **Оркестратор** (`guardian.py`) | Собирает всё в единый отчёт: сводная серьёзность, алерты на русском, рекомендация. |

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# демо-данные кредитного скоринга с искусственным дрейфом (seed фиксирован)
python scripts/generate_demo.py --out data/demo --seed 42
```

```python
import pandas as pd
from drift_guardian import analyze

reference = pd.read_csv("data/demo/reference.csv")
current = pd.read_csv("data/demo/current_mixed.csv")

report = analyze(reference, current)
print(report["overall_severity"])   # "critical"
print(report["recommendation"])
for alert in report["alerts"]:
    print(alert)
```

## Контракт интерфейса

Вход: два `pandas.DataFrame` — `reference` (эталон) и `current` (текущий батч).
Выход: словарь со следующей структурой.

```text
overall_severity   "ok" | "warning" | "critical"
recommendation     текстовая рекомендация (наблюдать / переобучать)
alerts             список человекочитаемых алертов
schema             замечания к схеме (пропавшие колонки, смена типов)
data_quality       замечания к качеству (пропуски, дубликаты, диапазоны, новые категории)
columns            по каждой колонке: kind, severity и список тестов
                   {name, statistic, p_value, severity, threshold, details}
adversarial        {roc_auc, severity, backend, n_rows_used, top_features}
meta               размеры выборок, списки колонок, alpha с поправкой, снимок конфига
```

Объектный API: `DriftGuardian(config).run(reference, current)` возвращает `DriftReport`
(dataclass), `.to_dict()` — тот же словарь.

## Пороги и логика алертов

Все пороги настраиваются через `DriftConfig` / `Thresholds`. Значения по умолчанию:

| Метрика | warning | critical |
|---|---|---|
| PSI | ≥ 0.10 | ≥ 0.20 |
| Дистанция Йенсена–Шеннона | ≥ 0.10 | ≥ 0.20 |
| Вассерштейн / std эталона | ≥ 0.10 | ≥ 0.30 |
| KS, хи-квадрат | p < α/k (Бонферрони) **и** размер эффекта ≥ warning | — |
| Adversarial ROC-AUC | ≥ 0.55 | ≥ 0.65 |
| Прирост доли пропусков | ≥ 5 п.п. | ≥ 15 п.п. |
| Доля значений вне диапазона | ≥ 1 % | ≥ 5 % |
| Доля строк с новыми категориями | ≥ 1 % | ≥ 5 % |

**Двухключевое правило для p-value-тестов.** KS и хи-квадрат фиксируют факт
статистически значимого сдвига, но на больших выборках значимым становится любой
микросдвиг, а на одинаковых распределениях они ложно срабатывают с частотой α.
Поэтому такой тест считается сработавшим, только если одновременно хотя бы одна
метрика размера эффекта (PSI, JS, Вассерштейн) превысила порог warning. Уровень
`critical` определяют только метрики размера эффекта. Значение p-value и флаг
`significant` при этом всегда есть в отчёте. Для p-value применяется поправка
Бонферрони на число проверяемых колонок.

## Структура репозитория

```text
drift_guardian/        пакет
  config.py            пороги и параметры анализа
  contracts.py         dataclass-контракт результатов
  columns.py           числовые vs категориальные колонки
  schema.py            Schema Validation
  data_quality.py      Data Quality
  drift_engines.py     статистические тесты дрейфа
  adversarial.py       Adversarial Validation (LightGBM)
  guardian.py          оркестратор и точка входа analyze()
scripts/generate_demo.py   генератор демо-данных с дрейфом
tests/                 unit-тесты (pytest)
notebooks/             демонстрационный notebook (в работе)
PLAN.md                план работ и распределение ролей
```

## Тесты

```bash
pytest -q
```

## Дорожная карта

- [x] Контракт интерфейса, четыре модуля анализа, оркестратор
- [x] Генератор демо-данных с фиксированным seed
- [x] Unit-тесты
- [ ] Интерактивный дашборд на Streamlit с графиками распределений (Plotly) и подсветкой «поплывших» признаков
- [ ] Экспорт HTML-отчёта
- [ ] Dockerfile и инструкция запуска
- [ ] Демонстрационный notebook
- [ ] Итоговый отчёт (PDF/HTML) и скринкаст
