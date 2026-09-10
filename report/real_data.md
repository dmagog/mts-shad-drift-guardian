# Реальные данные: галерея отчётов

Сгенерировано `scripts/real_data_gallery.py`. Датасеты OpenML скачиваются скриптом: [adult](https://www.openml.org/d/1590), [credit-g](https://www.openml.org/d/31), [electricity](https://www.openml.org/d/151).

| Кейс | Строк (эталон / батч) | Признаков | Итог | Признаки с дрейфом | Таргет | Adv. AUC | Сегментов | Время, с |
|---|---|---|---|---|---|---|---|---|
| adult: случайный сплит | 30 000 / 18 842 | 14 | ok | — | ok | 0.50 | 2 | 1.9 |
| adult: батч — люди старше 50 | 30 000 / 9 808 | 14 | critical | age, education-num, workclass, education, marital-status, occupation … | ok | 1.00 | 2 | 1.3 |
| credit-g: 700 против 300 строк | 700 / 300 | 20 | ok | — | ok | 0.52 | 0 | 0.2 |
| electricity: начало ряда против конца | 15 000 / 15 000 | 8 | critical | date, nswprice, nswdemand, vicprice, vicdemand, transfer | ok | 1.00 | 7 | 1.5 |
