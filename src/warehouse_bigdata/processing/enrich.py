"""J03: calendar/price enrichment and analytical aggregates."""

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from .normalize import SALES_DAILY_COLUMNS, ProcessingError

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

SALES_ENRICHED_COLUMNS = (
    *SALES_DAILY_COLUMNS,
    "sell_price",
    "price_missing",
    "wday",
    "month",
    "year",
    "snap_applicable",
    "is_zero_sale",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
)


def prepare_sell_prices(sell_prices: "DataFrame") -> "DataFrame":
    """Type the weekly price table and require one positive price per store-item-week."""

    typed = sell_prices.select(
        "store_id",
        "item_id",
        F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),
        F.col("sell_price").cast("double").alias("sell_price"),
    )
    stats = typed.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("store_id", "item_id", "wm_yr_wk").alias("keys"),
        F.sum(
            F.when(F.col("sell_price").isNotNull() & (F.col("sell_price") <= 0), 1).otherwise(0)
        ).alias("non_positive"),
    ).first()
    if stats["rows"] != stats["keys"]:
        raise ProcessingError("sell_prices must have one row per (store_id, item_id, wm_yr_wk).")
    if stats["non_positive"]:
        raise ProcessingError("sell_prices contains non-positive prices.")
    return typed


def enrich_sales(
    sales_daily: "DataFrame", calendar: "DataFrame", sell_prices: "DataFrame"
) -> "DataFrame":
    """Return ``sales_enriched`` with the same row grain as ``sales_daily``.

    Prices are joined on ``(store_id, item_id, wm_yr_wk)`` with a left join; a missing
    price stays null and is flagged, never replaced with zero. ``calendar`` must come
    from ``prepare_calendar`` and ``sell_prices`` from ``prepare_sell_prices``.
    """

    calendar_columns = calendar.select(
        F.col("d").alias("day_key"),
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    )
    enriched = (
        sales_daily.join(F.broadcast(calendar_columns), "day_key", "left")
        .join(sell_prices, ["store_id", "item_id", "wm_yr_wk"], "left")
        .withColumn("price_missing", F.col("sell_price").isNull())
        .withColumn("is_zero_sale", F.col("unit_sales") == 0)
        .withColumn(
            "snap_applicable",
            F.when(F.col("state_id") == "CA", F.col("snap_CA") == 1)
            .when(F.col("state_id") == "TX", F.col("snap_TX") == 1)
            .when(F.col("state_id") == "WI", F.col("snap_WI") == 1)
            .otherwise(F.lit(False)),
        )
    )
    return enriched.select(*SALES_ENRICHED_COLUMNS)


def store_daily_sales(sales_daily: "DataFrame", source_snapshot_id: str) -> "DataFrame":
    """Aggregate observed units per snapshot-store-date."""

    return (
        sales_daily.groupBy("store_id", "date")
        .agg(
            F.sum("unit_sales").cast("long").alias("total_units"),
            F.count(F.lit(1)).cast("long").alias("observed_series_count"),
        )
        .select(
            F.lit(source_snapshot_id).alias("source_snapshot_id"),
            "store_id",
            "date",
            "total_units",
            "observed_series_count",
        )
    )


def item_store_weekly_sales(sales_daily: "DataFrame") -> "DataFrame":
    """Aggregate units and observed-day coverage per item-store-retail week."""

    return sales_daily.groupBy("item_id", "store_id", "wm_yr_wk").agg(
        F.sum("unit_sales").cast("long").alias("total_units"),
        F.count(F.lit(1)).cast("long").alias("observed_days"),
    )


def category_monthly_sales(sales_daily: "DataFrame") -> "DataFrame":
    """Aggregate units per category and month start."""

    return (
        sales_daily.withColumn("month_start", F.trunc("date", "month"))
        .groupBy("cat_id", "month_start")
        .agg(F.sum("unit_sales").cast("long").alias("total_units"))
    )
