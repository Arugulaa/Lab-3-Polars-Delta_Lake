# **Лабораторная работа №3. Lakehouse на Polars + Delta Lake**
|Выполнил: Пархоменко Николай, 25-МАГ-ИАД|

## 1. Описание задачи

Проект реализует пайплайн Bronze → Silver → Gold для прогнозирования
задержек авиарейсов США на данных 2018–2024 годов. Данные обрабатываются
через Polars Lazy API, хранятся в Delta Lake, модели и метрики логируются в MLflow.

Датасет: https://www.kaggle.com/code/peymanradmanesh/flight-delay-analysis-2018-2024

## 2. Архитектура проекта
```
3_lab/
├── data/
│   └── flight_data_2018_2024.csv   # исходный CSV-датасет
│
├── delta/                          # Delta Lake — хранилище всех слоёв
│   ├── bronze/                     # сырые данные, как есть из CSV
│   ├── silver/                     # очищенные данные с производными признаками
│   └── gold/                       # витрины: аналитика и feature table для ML
│
├── logs/
│   └── pipeline.log                # лог выполнения пайплайна
│
├── notebooks/
│   └── explain_query.ipynb         # анализ планов запросов (.explain())
│
├── src/
│   └── delta/                      # основные модули пайплайна
│       ├── __init__.py             # делает папку Python-пакетом
│       ├── bronze.py               # загрузка CSV -> Bronze Delta-таблица
│       ├── silver.py               # очистка, MERGE в Silver
│       ├── gold.py                 # формирование витрин Gold
│       ├── ml.py                   # обучение моделей, логирование в MLflow
│       └── delta_ops.py            # утилиты Delta: OPTIMIZE, VACUUM, Z-ORDER, time travel
│
├── main.py                         # точка входа: запускает bronze->silver->gold->ml
├── docker-compose.yml             
├── Dockerfile                      
├── requirements.txt                
└── README.md                       
```

## 3. Руководство по запуску

*1. Клонировать репозиторий:*
```
git clone https://github.com/https://github.com/Arugulaa/Lab-3-Polars-Delta_Lake.git/3_lab.git
cd 3_lab
```
*2. Скачать датасет:*
```
data/flight_data_2018_2024.csv — скачать с Kaggle вручную
```

*3. Собрать и запустить:*
```
docker-compose up --build
```
## 4. Выполнение запроса с .explain() и выбор партиционирования
Реализация данного пункта представлена в `notebooks/explain_query.ipynb`.

Код:

```
import polars as pl
import os

SILVER_PATH = os.path.join("..", "delta", "silver")

query = (
    pl.scan_delta(SILVER_PATH)
    .filter(pl.col("Origin") == "ATL")
    .select(["FlightDate", "Origin", "Dest", "ArrDelay", "season"])
    .group_by("season")
    .agg(pl.col("ArrDelay").mean().alias("avg_delay"))
)

print(query.explain())
```

Результат:

```
AGGREGATE[maintain_order: false]
  [col("ArrDelay").mean().alias("avg_delay")] BY [col("season")]
  FROM
  simple π 2/2 ["ArrDelay", "season"]
    Parquet SCAN [C:/projects/3_lab/delta/silver/FlightDate=2024-01-31/part-00000-2c9d9c8d-4634-4b73-9149-29e411908bdb-c000.zstd.parquet, ... 30 other sources]
    PROJECT 3/16 COLUMNS
    SELECTION: [(col("Origin")) == ("ATL")]
```

* PROJECT 3/16 COLUMNS показывает PROJECT pushdown: из 16 колонок в файлах читаются только 3 нужные (для фильтра и агрегации), остальные вообще не грузятся.

* SELECTION: [(col("Origin")) == ("ATL")] показывает SELECTION pushdown: фильтр по аэропорту ATL выполняется прямо при чтении parquet‑файлов, строки с другими аэропортами отбрасываются сразу

Silver-слой партиционируется по ["year", "month", "FlightDate"]. В нашем подмножестве данных фактически присутствует один год и один месяц, поэтому разбивка только по year/month не даёт реального эффекта. Адекватный способ сплита здесь — по дням, то есть по полю FlightDate.

## 5. Результаты

1. После запуска Docker и выполнения пайплайна откройте браузер и перейдите по адресу `http://localhost:5000`.
2. В интерфейсе MLflow выберите эксперимент `Flight_Delay_Prediction`.
3. На вкладке **Model Metrics** можно изучить метрики моделей (RMSE, MAE, Accuracy и др.).
