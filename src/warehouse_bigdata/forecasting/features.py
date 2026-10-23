"""Point-in-time-correct feature construction (docs/06 section 8, docs/07 section 7)."""

from typing import TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F

from .protocol import FEATURE_VERSION, LOOKBACK_DAYS, M5_PROTOCOL, Protocol

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

MODEL_NUMERIC_FEATURES = (
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "is_weekend",
    "event_flag",
    "price_missing",
)
MODEL_CATEGORICAL_FEATURES = ("store_id", "dept_id")

FORECAST_FEATURE_COLUMNS = (
    "source_snapshot_id",
    "feature_version",
    "feature_role",
    "item_id",
    "store_id",
    "dept_id",
    "target_date",
    "as_of_date",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "is_weekend",
    "event_flag",
    "price_missing",
    "last_known_price",
    "label",
)


class FeatureError(ValueError):
    """Raised when inputs violate a feature-construction precondition."""


def add_calendar_features(
    df: "DataFrame", date_col: str, event_cols: tuple[str, str] = ("event_name_1", "event_name_2")
) -> "DataFrame":
    """Add deterministic target-date features known at issue time.

    ``day_of_week`` follows Spark ``dayofweek`` (1 = Sunday). ``event_flag`` assumes the
    calendar's scheduled events were known in advance (documented assumption).
    """

    dow = F.dayofweek(F.col(date_col))
    return (
        df.withColumn("day_of_week", dow)
        .withColumn("month", F.month(F.col(date_col)))
        .withColumn("is_weekend", dow.isin(1, 7))
        .withColumn(
            "event_flag", F.col(event_cols[0]).isNotNull() | F.col(event_cols[1]).isNotNull()
        )
    )


def assert_contiguous_daily(sales: "DataFrame") -> None:
    """Require one row per calendar day per series, as row-offset windows assume."""

    bad = (
        sales.groupBy("item_id", "store_id")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("day_index").alias("distinct_days"),
            (F.max("day_index") - F.min("day_index") + 1).alias("span"),
        )
        .filter((F.col("rows") != F.col("distinct_days")) | (F.col("rows") != F.col("span")))
        .limit(1)
        .count()
    )
    if bad:
        raise FeatureError("Every series must have a unique, contiguous daily history.")


def build_forecast_features(
    sales_enriched: "DataFrame",
    source_snapshot_id: str,
    *,
    protocol: Protocol = M5_PROTOCOL,
    feature_version: str = FEATURE_VERSION,
) -> "DataFrame":
    """Build one-step supervised rows: label at ``t`` from information up to ``t-1``.

    Lags and rolling means exclude the target day. Only rows with a full 28-day prior
    window are produced; missing lags are never filled with zero. ``last_known_price``
    is the latest non-null price strictly before the target day.
    """

    ordered = Window.partitionBy("item_id", "store_id").orderBy("day_index")
    prior_price = ordered.rowsBetween(Window.unboundedPreceding, -1)
    role = (
        F.when(F.col("day_index") <= protocol.train_last_day, "train")
        .when(F.col("day_index") <= protocol.validation_last_day, "validation")
        .otherwise("test")
    )

    framed = (
        sales_enriched.withColumn("row_number", F.row_number().over(ordered))
        .withColumn("as_of_date", F.lag("date", 1).over(ordered))
        .withColumn("lag_1", F.lag("unit_sales", 1).over(ordered))
        .withColumn("lag_7", F.lag("unit_sales", 7).over(ordered))
        .withColumn("lag_28", F.lag("unit_sales", LOOKBACK_DAYS).over(ordered))
        .withColumn("rolling_mean_7", F.avg("unit_sales").over(ordered.rowsBetween(-7, -1)))
        .withColumn(
            "rolling_mean_28", F.avg("unit_sales").over(ordered.rowsBetween(-LOOKBACK_DAYS, -1))
        )
        .withColumn("last_known_price", F.last("sell_price", ignorenulls=True).over(prior_price))
        .filter(F.col("row_number") > LOOKBACK_DAYS)
    )
    featured = add_calendar_features(framed, "date").withColumn(
        "price_missing", F.col("last_known_price").isNull()
    )
    return featured.select(
        F.lit(source_snapshot_id).alias("source_snapshot_id"),
        F.lit(feature_version).alias("feature_version"),
        role.alias("feature_role"),
        "item_id",
        "store_id",
        "dept_id",
        F.col("date").alias("target_date"),
        "as_of_date",
        "lag_1",
        "lag_7",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "day_of_week",
        "month",
        "is_weekend",
        "event_flag",
        "price_missing",
        "last_known_price",
        F.col("unit_sales").alias("label"),
    )


def to_model_input(
    df: "DataFrame", *, with_label: bool = False, keys: tuple[str, ...] = ()
) -> "DataFrame":
    """Select model inputs with numeric/boolean features cast to double.

    ``keys`` are extra columns passed through untouched (for example identifiers).
    """

    cast = [F.col(c).cast("double").alias(c) for c in MODEL_NUMERIC_FEATURES]
    extra = [F.col("last_known_price").cast("double").alias("last_known_price")]
    keep = [F.col(c) for c in (*keys, *MODEL_CATEGORICAL_FEATURES)]
    label = [F.col("label").cast("double").alias("label")] if with_label else []
    return df.select(*keep, *cast, *extra, *label)


def sample_training_rows(features: "DataFrame", *, target_rows: int, seed: int = 42) -> "DataFrame":
    """Deterministically sample about ``target_rows`` training-role rows.

    Selection hashes ``(item_id, store_id, target_date, seed)`` so reruns pick the same
    keys regardless of partitioning. Sampling happens after feature construction, so lag
    distances are never distorted. The sample is approximate, not exact.
    """

    training = features.filter(F.col("feature_role") == "train")
    total = training.count()
    if total == 0:
        raise FeatureError("No training rows are available.")
    fraction = min(1.0, target_rows / total)
    bucket = F.abs(
        F.xxhash64(F.col("item_id"), F.col("store_id"), F.col("target_date"), F.lit(seed))
    ) % F.lit(1_000_000)
    return training.filter(bucket < F.lit(int(fraction * 1_000_000)))
