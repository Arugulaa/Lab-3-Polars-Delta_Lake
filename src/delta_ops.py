import polars as pl
from deltalake import DeltaTable, write_deltalake
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_PATH = os.path.join(BASE_DIR, "delta", "silver")
GOLD_FEATURES_PATH = os.path.join(BASE_DIR, "delta", "gold", "features")


def run_delta_ops():
    logger.info("=== DELTA OPS: Обслуживание таблиц ===")

    # Открываем Silver-таблицу Delta Lake
    dt_silver = DeltaTable(SILVER_PATH)

    # Получаем список уникальных дат в Silver-слое
    # Это нужно, чтобы выполнять optimize/z-order по дневным партициям
    df_dates = (
        pl.scan_delta(SILVER_PATH)
        .select("FlightDate")
        .unique()
        .sort("FlightDate")
        .collect()
    )

    date_list = df_dates["FlightDate"].to_list()

    # --- OPTIMIZE (compaction) ---
    # Объединяем мелкие файлы внутри каждой дневной партиции
    for day_val in date_list:
        try:
            dt_silver.optimize.compact(
                partition_filters=[("FlightDate", "=", day_val)]
            )
            logger.info(f"OPTIMIZE выполнен: FlightDate={day_val}")
        except Exception as e:
            logger.warning(f"OPTIMIZE пропущен для {day_val}: {e}")

    logger.info("OPTIMIZE (compaction) выполнен на Silver.")

    # --- Z-ORDER ---
    # Переупорядочиваем данные внутри партиции по часто используемым колонкам
    # Это может ускорить фильтрацию по аэропорту и дню недели
    for day_val in date_list:
        try:
            dt_silver.optimize.z_order(
                ["Origin", "DayOfWeek"],
                partition_filters=[("FlightDate", "=", day_val)]
            )
            logger.info(f"Z-ORDER выполнен: FlightDate={day_val}")
        except Exception as e:
            logger.warning(f"Z-ORDER пропущен для {day_val}: {e}")

    logger.info("Z-ORDER по [Origin, DayOfWeek] выполнен на Silver.")

    # --- VACUUM ---
    # Удаляем старые неиспользуемые файлы старше 7 дней
    dt_silver.vacuum(
        retention_hours=168,
        dry_run=False,
        enforce_retention_duration=False
    )
    logger.info("VACUUM выполнен на Silver.")

    # --- TIME TRAVEL ---
    # Читаем самую первую версию таблицы Silver
    logger.info("[Time Travel] Читаем версию 0 таблицы Silver:")
    df_v0 = (
        pl.scan_delta(SILVER_PATH, version=0)
        .select(["FlightDate", "Origin", "Dest", "ArrDelay"])
        .limit(5)
        .collect()
    )
    logger.info(f"\n{df_v0}")
    logger.info(f"Текущая версия Silver: {dt_silver.version()}")

    # --- SCHEMA EVOLUTION ---
    # Читаем всю Gold feature table, чтобы не потерять существующие колонки
    logger.info("[Schema Evolution] Добавляем колонку is_long_flight в Gold features:")

    df_gold = pl.scan_delta(GOLD_FEATURES_PATH).collect()

    # Добавляем новый бинарный признак:
    # 1 — дальний рейс, если расстояние больше 1000
    # 0 — иначе
    df_new = df_gold.with_columns(
        (pl.col("Distance") > 1000).cast(pl.Int8).alias("is_long_flight")
    )

    # Записываем обновлённую схему в ту же таблицу
    write_deltalake(
        GOLD_FEATURES_PATH,
        df_new,
        mode="overwrite",
        schema_mode="merge"
    )
    logger.info("Schema evolution выполнен: добавлена колонка is_long_flight.")

    logger.info("=== DELTA OPS: Готово ===")


if __name__ == "__main__":
    run_delta_ops()