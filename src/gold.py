import polars as pl
from deltalake import DeltaTable, write_deltalake
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_PATH = os.path.join(BASE_DIR, "delta", "silver")
GOLD_ANALYTICS_PATH = os.path.join(BASE_DIR, "delta", "gold", "analytics")
GOLD_FEATURES_PATH = os.path.join(BASE_DIR, "delta", "gold", "features")

def run_gold():
    logger.info("=== GOLD: Создание витрин ===")

    lf = pl.scan_delta(SILVER_PATH)

    # --- Витрина 1: Аналитические агрегаты ---
    agg_df = (
        lf
        .group_by(["year", "month", "IATA_Code_Marketing_Airline", "Origin", "Dest", "hour", "season"])
        .agg([
            pl.col("ArrDelay").mean().alias("avg_arr_delay"),
            pl.col("ArrDelay").std().alias("std_arr_delay"),
            pl.col("DepDelay").mean().alias("avg_dep_delay"),
            pl.len().alias("num_flights"),
            (pl.col("ArrDelay") > 15).sum().alias("delayed_flights"),
        ])
        .with_columns(
            (pl.col("delayed_flights") / pl.col("num_flights") * 100).alias("delay_pct")
        )
        .collect()
    )
    logger.info(f"Аналитика: {agg_df.shape[0]} строк")

    os.makedirs(GOLD_ANALYTICS_PATH, exist_ok=True)
    write_deltalake(GOLD_ANALYTICS_PATH, agg_df, mode="overwrite")
    logger.info("Витрина analytics записана.")

    # --- Витрина 2: Feature table для ML ---
    feature_df = (
        lf
        .with_columns([
            (pl.col("ArrDelay") > 15).cast(pl.Int8).alias("is_delayed"),
            pl.col("ArrDelay").cast(pl.Float32).alias("target_regression"),
        ])
        .select([
            "IATA_Code_Marketing_Airline",
            "Origin", "Dest", "route",
            "month", "DayOfWeek", "hour", "season",
            "Distance", "DepDelay",
            "target_regression", "is_delayed"
        ])
        .collect()
    )
    logger.info(f"Feature table: {feature_df.shape[0]} строк")

    os.makedirs(GOLD_FEATURES_PATH, exist_ok=True)
    write_deltalake(GOLD_FEATURES_PATH, feature_df, mode="overwrite")
    logger.info("Витрина features записана.")

    logger.info(f"Версия analytics: {DeltaTable(GOLD_ANALYTICS_PATH).version()}")
    logger.info(f"Версия features: {DeltaTable(GOLD_FEATURES_PATH).version()}")
    logger.info("=== GOLD: Готово ===")

if __name__ == "__main__":
    run_gold()