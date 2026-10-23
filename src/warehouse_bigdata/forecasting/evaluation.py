"""Rolling-origin evaluation and forecast metrics (docs/07 sections 8, 10, 11)."""

from datetime import date, timedelta
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from .history import build_history_state
from .inference import recursive_forecast, seasonal_naive_forecast
from .protocol import HORIZON, METHOD_RANDOM_FOREST, METHOD_SEASONAL_NAIVE

if TYPE_CHECKING:
    from pyspark.ml import PipelineModel
    from pyspark.sql import DataFrame

METRIC_COLUMNS = (
    "evaluation_run_id",
    "model_run_id",
    "forecast_run_id",
    "holdout_id",
    "method_id",
    "metric_name",
    "horizon",
    "metric_value",
    "observation_count",
    "sum_absolute_error",
    "sum_actual_units",
    "evaluation_start",
    "evaluation_end",
    "run_type",
)


def score_origin(
    model: "PipelineModel",
    sales_enriched: "DataFrame",
    calendar: "DataFrame",
    issue_date: date,
    *,
    window_id: str,
) -> "DataFrame":
    """Score one issue date with both methods and align to observed outcomes.

    The history state reads only data on or before ``issue_date``; observed outcomes
    for ``T+1..T+7`` are joined afterwards, purely for scoring. The cohort contains only
    series that have a full history window and all seven observed outcomes, so both
    methods are scored on identical rows.
    """

    state = build_history_state(sales_enriched, issue_date).cache()
    rf = recursive_forecast(model, state, calendar, issue_date).select(
        "item_id",
        "store_id",
        "horizon",
        "forecast_date",
        F.col("predicted_units").alias(METHOD_RANDOM_FOREST),
    )
    naive = seasonal_naive_forecast(state, issue_date).select(
        "item_id", "store_id", "horizon", F.col("predicted_units").alias(METHOD_SEASONAL_NAIVE)
    )
    actuals = sales_enriched.filter(
        (F.col("date") > F.lit(issue_date))
        & (F.col("date") <= F.lit(issue_date + timedelta(days=HORIZON)))
    ).select(
        "item_id",
        "store_id",
        F.col("date").alias("forecast_date"),
        F.col("unit_sales").alias("actual"),
    )

    aligned = rf.join(naive, ["item_id", "store_id", "horizon"]).join(
        actuals, ["item_id", "store_id", "forecast_date"]
    )
    complete = aligned.groupBy("item_id", "store_id").agg(F.count(F.lit(1)).alias("n"))
    return (
        aligned.join(complete.filter(F.col("n") == HORIZON).drop("n"), ["item_id", "store_id"])
        .withColumn("window_id", F.lit(window_id))
        .withColumn("issue_date", F.lit(issue_date))
    )


def compute_forecast_metrics(
    scored: "DataFrame",
    *,
    evaluation_run_id: str,
    model_run_id: str,
    holdout_id: str,
    run_type: str,
) -> "DataFrame":
    """Return ``forecast_metrics`` rows (MAE, RMSE, WAPE) by method and horizon.

    Horizon ``0`` is the pooled result across horizons 1-7. Sums are kept next to each
    metric; WAPE is null (not zero) when the actual total is zero.
    """

    long = scored.select(
        "horizon",
        "forecast_date",
        F.col("actual").cast("double").alias("actual"),
        F.expr(
            f"stack(2, '{METHOD_RANDOM_FOREST}', {METHOD_RANDOM_FOREST}, "
            f"'{METHOD_SEASONAL_NAIVE}', {METHOD_SEASONAL_NAIVE}) as (method_id, prediction)"
        ),
    ).withColumn("error", F.col("prediction") - F.col("actual"))
    pooled = long.withColumn("horizon", F.lit(0))
    grouped = (
        long.unionByName(pooled)
        .groupBy("method_id", "horizon")
        .agg(
            F.count(F.lit(1)).alias("observation_count"),
            F.sum(F.abs("error")).alias("sum_absolute_error"),
            F.sum(F.col("error") * F.col("error")).alias("sum_squared_error"),
            F.sum("actual").alias("sum_actual_units"),
            F.min("forecast_date").alias("evaluation_start"),
            F.max("forecast_date").alias("evaluation_end"),
        )
    )
    metrics = grouped.select(
        "*",
        F.expr(
            "stack(3, "
            "'MAE', sum_absolute_error / observation_count, "
            "'RMSE', sqrt(sum_squared_error / observation_count), "
            "'WAPE', CASE WHEN sum_actual_units > 0 "
            "THEN sum_absolute_error / sum_actual_units * 100 ELSE NULL END"
            ") as (metric_name, metric_value)"
        ),
    )
    return metrics.select(
        F.lit(evaluation_run_id).alias("evaluation_run_id"),
        F.lit(model_run_id).alias("model_run_id"),
        F.lit(None).cast("string").alias("forecast_run_id"),
        F.lit(holdout_id).alias("holdout_id"),
        "method_id",
        "metric_name",
        "horizon",
        F.col("metric_value").cast("double"),
        F.col("observation_count").cast("long"),
        F.col("sum_absolute_error").cast("double"),
        F.col("sum_actual_units").cast("double"),
        "evaluation_start",
        "evaluation_end",
        F.lit(run_type).alias("run_type"),
    )
