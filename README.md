# Data Drift Guardian

[![tests](https://github.com/dmagog/mts-shad-drift-guardian/actions/workflows/tests.yml/badge.svg)](https://github.com/dmagog/mts-shad-drift-guardian/actions/workflows/tests.yml)

Система детекции дрейфа данных и контроля их качества для ML-моделей в продакшене.
Сравнивает эталонную выборку (на которой обучалась модель) с текущим продакшн-батчем
и выдаёт структурированный отчёт с метриками и алертами — без нейросетей, на строгой
математической статистике и классическом ML. Результат доступен как словарь Python,
интерактивный дашборд на Streamlit, самодостаточный HTML-отчёт, JSON и код возврата CLI.

Итоговый проект «4.0 Школы аналитиков данных» МТС, задача №4. Заказчик — Бояджи Владислав.

## Что умеет

| Модуль | Что делает |
|---|---|
| **Data Ingestion Core** (`io.py`, `schema.py`) | Загрузка CSV/Parquet; сверка набора колонок и семейств типов: пропавшие/новые колонки, смена типа. |
| **Data Quality** (`data_quality.py`) | Рост доли пропусков, дубликаты строк, значения вне допустимого диапазона, недопустимые категории, константные колонки. Границы и категории — из контракта данных или из эталона. |
| **Drift Engines** (`drift_engines.py`) | Числовые: KS-тест, PSI, дистанция Йенсена–Шеннона, расстояние Вассерштейна. Категориальные: хи-квадрат, PSI, Йенсен–Шеннон. Пороги откалиброваны между собой. |
| **Adversarial Validation** (`adversarial.py`) | LightGBM учится отличать эталон от батча; ROC-AUC ≫ 0.5 — многомерный сдвиг подтверждён, feature importance показывает, что изменилось сильнее всего. |
| **Концептуальный дрейф** (`guardian.py`) | Целевая переменная и предсказания модели анализируются отдельными блоками: признаки стабильны, а цель изменилась — отдельная рекомендация. |
| **Оркестратор** (`guardian.py`) | Единый отчёт: сводная серьёзность, алерты на русском, рекомендация «наблюдать / переобучать». |
| **Визуализация и экспорт** (`plots.py`, `html_report.py`, `app/`, `cli.py`) | Графики «эталон vs батч» (Plotly), дашборд Streamlit с подсветкой «поплывших» признаков, HTML-отчёт, JSON, командная строка с кодом возврата. |

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```python
from drift_guardian import DriftConfig, analyze
from drift_guardian.demo import make_demo

reference, current = make_demo("mixed")             # демо-данные кредитного скоринга с дрейфом
report = analyze(reference, current, DriftConfig(target_column="target"))

print(report["overall_severity"])                   # "critical"
print(report["recommendation"])
for alert in report["alerts"]:
    print(alert)
```

Из командной строки (код возврата 0 — ok, 1 — warning, 2 — critical, удобно для cron/CI):

```bash
drift-guardian --reference data/demo/reference.csv --current data/demo/current_mixed.csv \
    --target target --exclude customer_id --html report.html --json report.json
```

Демо-данные в виде CSV (эталон + шесть сценариев дрейфа, seed фиксирован):

```bash
python scripts/generate_demo.py --out data/demo --seed 42
```

## Дашборд

```bash
streamlit run app/streamlit_app.py
```

Откроется `http://localhost:8501`. В боковой панели — источник данных (демо-сценарий или свои
CSV/Parquet), роли колонок (целевая переменная, исключения) и пороги алертов. Справа —
сводный статус и рекомендация, алерты, блок целевой переменной и вкладки: сводка по признакам
с подсветкой красным/жёлтым, распределения по каждому признаку, замечания к схеме и качеству,
adversarial validation с важностями, экспорт HTML/JSON.

## Docker

```bash
docker build -t drift-guardian .
docker run --rm -p 8501:8501 drift-guardian
```

## Контракт данных (конфигурация)

Все параметры — в `DriftConfig`; их можно хранить в YAML рядом с моделью
(пример — [`examples/config.yaml`](examples/config.yaml)):

```yaml
target_column: target                 # целевая переменная → блок концептуального дрейфа
prediction_column: null               # предсказания модели → блок дрейфа предсказаний
exclude_columns: [customer_id, dt]    # идентификаторы и даты не анализируются
value_bounds: {age: [18, 90]}         # допустимые диапазоны по бизнес-правилам
allowed_categories: {employment_type: [наёмный, ИП, самозанятый, безработный]}
thresholds: {psi_warning: 0.1, psi_critical: 0.2}
```

```python
config = DriftConfig.from_yaml("examples/config.yaml")
```

Если границы и категории не заданы, они выводятся из эталона: `[min, max]` и множество
наблюдавшихся категорий.

## Контракт результата

Вход: два `pandas.DataFrame` — `reference` (эталон) и `current` (текущий батч).
Выход: словарь (пример — [`examples/drift_report_mixed.json`](examples/drift_report_mixed.json)).

```text
overall_severity   "ok" | "warning" | "critical"
recommendation     текстовая рекомендация (наблюдать / переобучать / концептуальный дрейф)
alerts             список человекочитаемых алертов
schema             замечания к схеме (пропавшие колонки, смена типов)
data_quality       замечания к качеству (пропуски, дубликаты, диапазоны, категории)
columns            по каждому признаку: kind, severity и список тестов
                   {name, statistic, p_value, severity, threshold, details}
target_drift       блок целевой переменной (та же структура, что у признака) или null
prediction_drift   блок предсказаний модели или null
adversarial        {roc_auc, severity, backend, n_rows_used, top_features}
meta               размеры выборок, списки колонок, alpha с поправкой, снимок конфига
```

Объектный API: `DriftGuardian(config).run(reference, current)` возвращает `DriftReport`
(dataclass), `.to_dict()` — тот же словарь.

## Пороги и логика алертов

| Метрика | warning | critical |
|---|---|---|
| PSI | ≥ 0.10 | ≥ 0.20 |
| Дистанция Йенсена–Шеннона | ≥ 0.10 | ≥ 0.20 |
| Вассерштейн / std эталона | ≥ 0.30 | ≥ 0.50 |
| KS, хи-квадрат | p < α/k (Бонферрони) **и** размер эффекта ≥ warning | — |
| Adversarial ROC-AUC | ≥ 0.55 | ≥ 0.65 |
| Прирост доли пропусков | ≥ 5 п.п. | ≥ 15 п.п. |
| Доля значений вне диапазона | ≥ 1 % | ≥ 5 % |
| Доля строк с недопустимыми категориями | ≥ 1 % | ≥ 5 % |

### Калибровка порогов

Пороги трёх метрик размера эффекта согласованы между собой, чтобы ни одна из них
не «перетягивала» вердикт. Значения при известных сдвигах нормального признака
(10 квантильных бинов, 200 000 наблюдений; полная таблица — `report/experiments.md`):

| Сдвиг среднего | PSI | JS | W/σ | Вердикт |
|---|---|---|---|---|
| 0.2σ | 0.040 | 0.084 | 0.204 | ok |
| 0.3σ | 0.086 | 0.124 | 0.299 | warning (JS) |
| 0.4σ | 0.153 | 0.165 | 0.400 | warning |
| 0.5σ | 0.239 | 0.205 | 0.499 | critical |

Для категорий доли ограничены снизу 10⁻⁴: без сглаживания PSI «взрывался» на любой
новой категории (1 % новой категории давал PSI > 0.2), теперь 1 % → 0.05, 5 % → 0.31.

### Двухключевое правило для p-value-тестов

KS и хи-квадрат фиксируют факт статистически значимого сдвига, но на больших выборках
значимым становится любой микросдвиг, а на одинаковых распределениях они ложно срабатывают
с частотой α. Поэтому такой тест считается сработавшим, только если одновременно хотя бы
одна метрика размера эффекта превысила порог warning. Уровень `critical` определяют только
метрики размера эффекта; значение p-value и флаг `significant` всегда есть в отчёте.
На 30 батчах без дрейфа наивное правило «p < α» дало ложные алерты в 2 батчах,
двухключевое — ни в одном (`report/experiments.md`).

### Концептуальный дрейф

Если указана `target_column`, целевая переменная не участвует в анализе признаков
и adversarial validation, а проверяется отдельно. Когда признаки стабильны, а цель
изменилась, система выдаёт отдельную рекомендацию: вероятен концептуальный дрейф,
нужны свежие размеченные данные. Демо-сценарий `concept_drift` показывает именно это.

## Структура репозитория

```text
drift_guardian/            пакет
  config.py                пороги, параметры, контракт данных, YAML
  contracts.py             dataclass-контракт результатов
  columns.py               числовые vs категориальные колонки
  io.py                    загрузка CSV / Parquet
  schema.py                Schema Validation
  data_quality.py          Data Quality
  drift_engines.py         статистические тесты дрейфа
  adversarial.py           Adversarial Validation (LightGBM)
  guardian.py              оркестратор, точка входа analyze()
  demo.py                  генератор демо-данных (6 сценариев, таргет)
  plots.py                 графики Plotly и таблицы
  html_report.py           HTML-отчёт и экспорт JSON
  cli.py                   командная строка (drift-guardian)
app/streamlit_app.py       дашборд Streamlit
scripts/generate_demo.py   демо-данные в CSV
scripts/experiments.py     эксперименты для отчёта
scripts/build_report.py    сборка итогового отчёта в HTML
notebooks/demo.ipynb       демонстрационный notebook (выполнен, с выводами)
report/                    итоговый отчёт (report.md → report.html) и эксперименты
examples/                  пример конфига, HTML- и JSON-отчёта
docs/screencast.md         сценарий скринкаста
tests/                     pytest: модули, отчёт, CLI, дашборд (streamlit.testing)
Dockerfile                 образ с дашбордом и CLI
```

## Тесты и качество

```bash
pytest -q          # 49 тестов, включая смоук-тесты дашборда
ruff check .       # линтер
```

CI (GitHub Actions) прогоняет линтер и тесты на каждый push. Точные версии
зависимостей, на которых проверялся проект, — в `requirements-lock.txt`.

## Ограничения

- Колонки типов datetime и вложенных структур не анализируются (попадают в `meta.skipped_columns`);
  дату лучше исключить или заранее превратить в признаки.
- Эвристика типов: целочисленная колонка с ≤ 10 уникальными значениями считается категориальной —
  переопределяется через `numeric_columns` / `categorical_columns`.
- Adversarial validation работает на подвыборке до 50 000 строк (настраивается).
- Пороги по умолчанию — разумные отраслевые ориентиры; для конкретной модели их стоит
  подобрать по историческим батчам.
