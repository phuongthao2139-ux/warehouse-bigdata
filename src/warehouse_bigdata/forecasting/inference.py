"""Seven-day recursive scoring and the seasonal-naive baseline (docs/07 sections 9-10)."""

from datetime import date, timedelta
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from .features import FeatureError, add_calendar_features, to_model_input
from .protocol import HORIZON, LOOKBACK_DAYS

if TYPE_CHECKING:
    from pyspark.ml import PipelineModel
    from pyspark.sql import DataFrame

FORECAST_COLUMNS = (
    "item_id",
    "store_id",
    "horizon",
    "forecast_date",
    "raw_prediction",
    "predicted_units",
)


def _target_calendar(calendar: "DataFrame", target: date) -> "DataFrame":
    row = calendar.filter(F.col("date") == F.lit(target)).select(
        F.col("date").alias("forecast_date"), "event_name_1", "event_name_2"
    )
    if row.count() != 1:
        raise FeatureError(f"Calendar must contain exactly one row for {target}.")
    return row


def recursive_forecast(
    model: "PipelineModel",
    state: "DataFrame",
    calendar: "DataFrame",
    issue_date: date,
    *,
    horizon: int = HORIZON,
) -> "DataFrame":
    """Score ``horizon`` days ahead with one-step recursion.

    At step ``h`` the lag/rolling features are rebuilt from actual history through
    ``issue_date`` plus the model's own earlier predictions; no actual after the issue
    date is read. Predictions are clipped at zero (``raw_prediction`` is kept) and a
    non-finite prediction fails the run instead of becoming zero.
    """

    hist = state.select("item_id", "store_id", "dept_id", "hist", "last_known_price")
    outputs: list[DataFrame] = []
    for step in range(1, horizon + 1):
        target = issue_date + timedelta(days=step)
        stepped = add_calendar_features(
            hist.crossJoin(F.broadcast(_target_calendar(calendar, target))), "forecast_date"
        )
        stepped = (
            stepped.withColumn("lag_1", F.element_at("hist", -1))
            .withColumn("lag_7", F.element_at("hist", -7))
            .withColumn("lag_28", F.element_at("hist", -LOOKBACK_DAYS))
            .withColumn(
                "rolling_mean_7",
                F.aggregate(F.slice("hist", -7, 7), F.lit(0.0), lambda a, x: a + x) / 7,
            )
            .withColumn(
                "rolling_mean_28",
                F.aggregate("hist", F.lit(0.0), lambda a, x: a + x) / LOOKBACK_DAYS,
            )
            .withColumn("price_missing", F.col("last_known_price").isNull())
        )
        scored = model.transform(to_model_input(stepped, keys=("item_id", "forecast_date")))
        step_out = scored.select(
            "item_id",
            "store_id",
            F.lit(step).alias("horizon"),
            "forecast_date",
            F.col("raw_prediction").cast("double").alias("raw_prediction"),
        ).withColumn("predicted_units", F.greatest(F.col("raw_prediction"), F.lit(0.0)))
        # Materialize each step so the lineage does not grow and earlier steps are not rescored.
        step_out = step_out.localCheckpoint()
        outputs.append(step_out)
        # Append this step's prediction to the temporary state; actual history is untouched.
        hist = (
            hist.join(
                step_out.select("item_id", "store_id", F.col("predicted_units").alias("_next")),
                ["item_id", "store_id"],
            )
            .withColumn(
                "hist",
                F.slice(F.concat("hist", F.array("_next")), -LOOKBACK_DAYS, LOOKBACK_DAYS),
            )
            .drop("_next")
            .localCheckpoint()
        )
    result = outputs[0]
    for frame in outputs[1:]:
        result = result.unionByName(frame)
    result = result.select(*FORECAST_COLUMNS)
    invalid = (
        result.filter(
            F.col("raw_prediction").isNull()
            | F.isnan("raw_prediction")
            | (F.abs("raw_prediction") == F.lit(float("inf")))
        )
        .limit(1)
        .count()
    )
    if invalid:
        raise FeatureError("Model produced a null or non-finite prediction.")
    return result


def seasonal_naive_forecast(
    state: "DataFrame", issue_date: date, *, horizon: int = HORIZON
) -> "DataFrame":
    """Forecast ``T+h`` as the actual of ``T+h-7`` (same weekday last week)."""

    frames = [
        state.select(
            "item_id",
            "store_id",
            F.lit(step).alias("horizon"),
            F.lit(issue_date + timedelta(days=step)).alias("forecast_date"),
            F.element_at("hist", -(8 - step)).alias("raw_prediction"),
        ).withColumn("predicted_units", F.col("raw_prediction"))
        for step in range(1, horizon + 1)
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.unionByName(frame)
    return result.select(*FORECAST_COLUMNS)


def to_forecast_daily(
    forecast: "DataFrame",
    *,
    forecast_run_id: str,
    model_run_id: str,
    source_snapshot_id: str,
    feature_version: str,
    issue_date: date,
    run_type: str,
) -> "DataFrame":
    """Attach lineage and return rows in the ``forecast_daily`` contract order."""

    return forecast.select(
        F.lit(forecast_run_id).alias("forecast_run_id"),
        F.lit(model_run_id).alias("model_run_id"),
        F.lit(source_snapshot_id).alias("source_snapshot_id"),
        F.lit(feature_version).alias("feature_version"),
        F.lit(issue_date).alias("issue_date"),
        "forecast_date",
        "horizon",
        "item_id",
        "store_id",
        "raw_prediction",
        "predicted_units",
        F.lit(run_type).alias("run_type"),
        F.current_timestamp().alias("created_at"),
    )
