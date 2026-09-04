# Data Drift Guardian

Система детекции дрейфа данных и контроля их качества для ML-моделей в продакшене.
Сравнивает эталонную выборку (на которой обучалась модель) с текущим продакшн-батчем
и выдаёт структурированный отчёт с метриками и алертами — без нейросетей, на строгой
математической статистике и классическом ML. Результат доступен как словарь Python,
интерактивный дашборд на Streamlit и самодостаточный HTML-отчёт.

Итоговый проект «4.0 Школы аналитиков данных» МТС, задача №4. Заказчик — Бояджи Владислав.

## Что умеет

| Модуль | Что делает |
|---|---|
| **Schema Validation** (`schema.py`) | Сверяет набор колонок и семейства типов: пропавшие/новые колонки, смена типа. |
| **Data Quality** (`data_quality.py`) | Рост доли пропусков, дубликаты строк, выход за диапазон `[min, max]` эталона, новые категории, константные колонки. |
| **Drift Engines** (`drift_engines.py`) | Числовые: KS-тест, PSI, дистанция Йенсена–Шеннона, расстояние Вассерштейна. Категориальные: хи-квадрат, PSI, Йенсен–Шеннон. |
| **Adversarial Validation** (`adversarial.py`) | LightGBM учится отличать эталон от батча; ROC-AUC ≫ 0.5 — дрейф подтверждён, feature importance показывает, что изменилось сильнее всего. |
| **Оркестратор** (`guardian.py`) | Собирает всё в единый отчёт: сводная серьёзность, алерты на русском, рекомендация. |
| **Визуализация** (`plots.py`, `html_report.py`, `app/`) | Графики распределений «эталон vs батч» (Plotly), дашборд Streamlit с подсветкой «поплывших» признаков, экспорт HTML и JSON. |

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```python
import pandas as pd
from drift_guardian import analyze
from drift_guardian.demo import make_demo

reference, current = make_demo("mixed")      # демо-данные с искусственным дрейфом
report = analyze(reference, current)         # словарь с метриками и алертами

print(report["overall_severity"])            # "critical"
print(report["recommendation"])
for alert in report["alerts"]:
    print(alert)
```

Те же демо-данные в виде CSV (эталон + пять сценариев дрейфа, seed фиксирован):

```bash
python scripts/generate_demo.py --out data/demo --seed 42
```

## Дашборд

```bash
streamlit run app/streamlit_app.py
```

Откроется `http://localhost:8501`. В боковой панели — источник данных (демо-сценарий
или свои CSV/Parquet: эталон и текущий батч) и пороги алертов. Справа — сводный статус
и рекомендация, список алертов, вкладки: сводка по признакам с подсветкой красным/жёлтым,
распределения по каждому признаку, замечания к схеме и качеству, adversarial validation
с важностями признаков, экспорт HTML/JSON.

## HTML-отчёт и JSON

```python
from drift_guardian.html_report import save_html_report, report_to_json

save_html_report("report/drift_report.html", report, reference, current)  # plotlyjs="cdn" — лёгкий файл
open("report/drift_report.json", "w").write(report_to_json(report))
```

По умолчанию Plotly.js встраивается в файл (≈4 МБ, работает офлайн); с `plotlyjs="cdn"`
файл лёгкий, но графикам нужен интернет.

## Docker

```bash
docker build -t drift-guardian .
docker run --rm -p 8501:8501 drift-guardian
```

Дашборд будет доступен на `http://localhost:8501`.

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

Все пороги настраиваются через `DriftConfig` / `Thresholds` (и ползунками в дашборде).
Значения по умолчанию:

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
  demo.py              генератор демо-данных с пятью сценариями дрейфа
  plots.py             графики Plotly и таблицы для дашборда/отчёта
  html_report.py       HTML-отчёт и экспорт JSON
app/streamlit_app.py   дашборд Streamlit
scripts/generate_demo.py   демо-данные в CSV
tests/                 unit-тесты и смоук-тест дашборда (pytest)
notebooks/             демонстрационный notebook (в работе)
Dockerfile             образ с дашбордом
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
- [x] Дашборд на Streamlit с графиками распределений и подсветкой «поплывших» признаков
- [x] Экспорт HTML-отчёта и JSON
- [x] Dockerfile и инструкция запуска
- [ ] Демонстрационный notebook
- [ ] Итоговый отчёт (PDF/HTML) и скринкаст
