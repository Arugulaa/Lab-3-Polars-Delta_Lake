import polars as pl
from deltalake import DeltaTable, write_deltalake
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_DIR, "delta", "bronze")
SILVER_PATH = os.path.join(BASE_DIR, "delta", "silver")

MERGE_KEYS = ["FlightDate", "IATA_Code_Marketing_Airline", "Flight_Number_Marketing_Airline", "Origin", "Dest"]

def run_silver():
    logger.info("=== SILVER: Очистка и трансформация ===")

    lf = pl.scan_delta(BRONZE_PATH)

    lf = (
        lf
        .filter(
            (pl.col("Cancelled") == 0) &
            (pl.col("ArrDelay").is_not_null()) &
            (pl.col("ArrDelay").abs() < 500)
        )
        .with_columns([
            (pl.col("CRSDepTime") // 100).cast(pl.Int32).alias("hour"),
            pl.when(pl.col("Month").is_in([12, 1, 2])).then(pl.lit("Winter"))
              .when(pl.col("Month").is_in([3, 4, 5])).then(pl.lit("Spring"))
              .when(pl.col("Month").is_in([6, 7, 8])).then(pl.lit("Summer"))
              .otherwise(pl.lit("Autumn")).alias("season"),
            pl.concat_str([pl.col("Origin"), pl.lit("_"), pl.col("Dest")]).alias("route"),
        ])
        .with_columns([
            pl.col("IATA_Code_Marketing_Airline").str.to_uppercase(),
            pl.col("Origin").str.to_uppercase(),
            pl.col("Dest").str.to_uppercase(),
        ])
        .rename({"Year": "year", "Month": "month"})
        .select([
            "FlightDate", "year", "month", "DayOfWeek",
            "IATA_Code_Marketing_Airline", "Flight_Number_Marketing_Airline",
            "Origin", "Dest",
            "CRSDepTime", "DepDelay",
            "CRSArrTime", "ArrDelay",
            "Distance",
            "hour", "season", "route"
        ])
    )

    df = lf.collect()
    logger.info(f"Строк после очистки: {df.shape[0]}")

    df = df.unique(subset=MERGE_KEYS, keep="first")
    logger.info(f"Строк после дедупликации: {df.shape[0]}")

    os.makedirs(SILVER_PATH, exist_ok=True)

    days = sorted(df["FlightDate"].unique().to_list())

    if DeltaTable.is_deltatable(SILVER_PATH):
        dt = DeltaTable(SILVER_PATH)
        predicate = " AND ".join([f"target.{k} = source.{k}" for k in MERGE_KEYS])

        for day_val in days:
            batch = df.filter(pl.col("FlightDate") == day_val)
            dt.merge(
                source=batch,
                predicate=predicate,
                source_alias="source",
                target_alias="target"
            ).when_matched_update_all().when_not_matched_insert_all().execute()
            logger.info(f"MERGE выполнен: date={day_val}, строк={batch.shape[0]}")

    else:
        for i, day_val in enumerate(days):
            batch = df.filter(pl.col("FlightDate") == day_val)
            write_mode = "overwrite" if i == 0 else "append"
            write_deltalake(
                SILVER_PATH,
                batch,
                partition_by=["year", "month", "FlightDate"],
                mode=write_mode
            )
            logger.info(f"Записан день: {day_val}, строк={batch.shape[0]}")

    logger.info(f"Версия Silver: {DeltaTable(SILVER_PATH).version()}")
    logger.info("=== SILVER: Готово ===")

if __name__ == "__main__":
    run_silver()