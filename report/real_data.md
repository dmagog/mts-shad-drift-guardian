# Реальные данные: галерея отчётов

Сгенерировано `scripts/real_data_gallery.py`.

| Кейс | Строк (эталон / батч) | Признаков | Итог | Признаки с дрейфом | Таргет | Adv. AUC | Сегментов | Время, с |
|---|---|---|---|---|---|---|---|---|
| adult: случайный сплит | 30000 / 18842 | 14 | ok | — | ok | 0.50 | 2 | 2.1 |
| adult: батч — люди старше 50 | 30000 / 9808 | 14 | critical | age, education-num, workclass, education, marital-status, occupation … | ok | 1.00 | 2 | 1.2 |
| credit-g: 700 против 300 строк | 700 / 300 | 20 | ok | — | ok | 0.52 | 0 | 0.2 |
| electricity: начало ряда против конца | 15000 / 15000 | 8 | critical | date, nswprice, nswdemand, vicprice, vicdemand, transfer | ok | 1.00 | 7 | 1.5 |
