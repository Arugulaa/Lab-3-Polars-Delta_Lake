import polars as pl
from deltalake import DeltaTable, write_deltalake
import os
import logging

# Получаем логгер с именем модуля — в логах будет видно "[src.bronze]"
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flight_data_2018_2024.csv")
BRONZE_PATH = os.path.join(BASE_DIR, "delta", "bronze")

def run_bronze():
    logger.info("=== BRONZE: Загрузка CSV в Delta ===")
    os.makedirs(BRONZE_PATH, exist_ok=True)

    df = pl.read_csv(DATA_PATH, infer_schema_length=10000, ignore_errors=True)
    logger.info(f"Загружено строк: {df.shape[0]}, колонок: {df.shape[1]}")

    df = df.with_columns(
        pl.col("FlightDate").cast(pl.Utf8).str.slice(0, 10).alias("batch_date")
    )

    days = df["batch_date"].unique().sort()

    for day in days:
        batch = df.filter(pl.col("batch_date") == day).drop("batch_date")
        write_deltalake(BRONZE_PATH, batch, mode="append")
        logger.info(f"Записан день: {day} | строк: {batch.shape[0]}")

    dt = DeltaTable(BRONZE_PATH)
    logger.info(f"Итого версий Delta: {dt.version()}")
    logger.info("=== BRONZE: Готово ===")

if __name__ == "__main__":
    run_bronze()