# Реальные данные: галерея отчётов

Сгенерировано `scripts/real_data_gallery.py`. Датасеты OpenML скачиваются скриптом: [adult](https://www.openml.org/d/1590), [credit-g](https://www.openml.org/d/31), [electricity](https://www.openml.org/d/151).

| Кейс | Строк (эталон / батч) | Признаков | Итог | Признаки с дрейфом | Таргет | Adv. AUC | Сегментов | Время, с |
|---|---|---|---|---|---|---|---|---|
| adult: случайный сплит | 30000 / 18842 | 14 | ok | — | ok | 0.50 | 2 | 16.5 |
| adult: батч — люди старше 50 | 30000 / 9808 | 14 | critical | age, education-num, workclass, education, marital-status, occupation … | ok | 1.00 | 2 | 9.4 |
| credit-g: 700 против 300 строк | 700 / 300 | 20 | ok | — | ok | 0.52 | 0 | 1.6 |
| electricity: начало ряда против конца | 15000 / 15000 | 8 | critical | date, nswprice, nswdemand, vicprice, vicdemand, transfer | ok | 1.00 | 7 | 10.5 |
