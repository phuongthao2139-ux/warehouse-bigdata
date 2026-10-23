"""Historical policy evaluation against observed sales as a demand proxy (docs/08 s11-12)."""

from datetime import date, timedelta
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from .decisions import HORIZON

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

EVALUATION_COLUMNS = (
    "evaluation_run_id",
    "source_snapshot_id",
    "forecast_run_id",
    "scenario_id",
    "decision_run_id",
    "policy_id",
    "policy_version",
    "issue_date",
    "item_id",
    "store_id",
    "actual_sales_proxy_7d_units",
    "simulated_fulfilled_units",
    "simulated_unmet_units",
    "ending_stock_units",
    "excess_over_safety_units",
    "simulated_stockout_days",
    "run_type",
    "observed_label_window_start",
    "observed_label_window_end",
)

SCORED_RUN_TYPES = ("validation", "test")


def weekly_actuals(sales: "DataFrame", issue_date: date) -> "DataFrame":
    """Return ``y1..y7`` observed units per series for ``T+1..T+7``.

    Series missing any of the seven days are dropped so they surface as exclusions
    rather than as zero-loss outcomes.
    """

    window = sales.filter(
        (F.col("date") > F.lit(issue_date))
        & (F.col("date") <= F.lit(issue_date + timedelta(days=HORIZON)))
    ).withColumn("h", F.datediff("date", F.lit(issue_date)))
    pivoted = (
        window.groupBy("item_id", "store_id")
        .pivot("h", list(range(1, HORIZON + 1)))
        .agg(F.first("unit_sales"))
    )
    renamed = pivoted.select(
        "item_id", "store_id", *[F.col(str(h)).alias(f"y{h}") for h in range(1, HORIZON + 1)]
    )
    complete = F.lit(True)
    for h in range(1, HORIZON + 1):
        complete = complete & F.col(f"y{h}").isNotNull()
    return renamed.filter(complete)


def evaluate_policies(
    allocation: "DataFrame",
    replenishment: "DataFrame",
    sales: "DataFrame",
    *,
    evaluation_run_id: str,
    run_type: str,
) -> "DataFrame":
    """Replay each policy's allocation against the revealed holdout week.

    ``stock_0 = opening + allocated``; each day sells ``min(stock, observed)``; unmet
    units are lost. Decisions must already be frozen: observed sales are read only here.
    """

    if run_type not in SCORED_RUN_TYPES:
        raise ValueError(f"run_type must be one of {SCORED_RUN_TYPES}, got {run_type!r}.")
    issue_dates = [r[0] for r in allocation.select("issue_date").distinct().limit(2).collect()]
    if len(issue_dates) != 1:
        raise ValueError("Allocation must contain exactly one issue_date.")
    issue_date = issue_dates[0]

    base = (
        allocation.join(
            replenishment.select(
                "item_id", "store_id", "opening_stock_units", "safety_stock_units"
            ),
            ["item_id", "store_id"],
        )
        .join(weekly_actuals(sales, issue_date), ["item_id", "store_id"])
        .withColumn("stock", F.col("opening_stock_units") + F.col("allocated_units"))
        .withColumn("fulfilled", F.lit(0))
        .withColumn("unmet", F.lit(0))
        .withColumn("stockout_days", F.lit(0))
    )
    for h in range(1, HORIZON + 1):
        sold = F.col(f"y{h}")
        served = F.least(F.col("stock"), sold)
        base = (
            base.withColumn("fulfilled", F.col("fulfilled") + served)
            .withColumn("unmet", F.col("unmet") + (sold - served))
            .withColumn(
                "stockout_days",
                F.col("stockout_days") + F.when(sold > F.col("stock"), 1).otherwise(0),
            )
            .withColumn("stock", F.col("stock") - served)
        )
    total = sum((F.col(f"y{h}") for h in range(1, HORIZON + 1)), F.lit(0))
    return base.select(
        F.lit(evaluation_run_id).alias("evaluation_run_id"),
        "source_snapshot_id",
        "forecast_run_id",
        "scenario_id",
        "decision_run_id",
        "policy_id",
        "policy_version",
        "issue_date",
        "item_id",
        "store_id",
        total.cast("int").alias("actual_sales_proxy_7d_units"),
        F.col("fulfilled").cast("int").alias("simulated_fulfilled_units"),
        F.col("unmet").cast("int").alias("simulated_unmet_units"),
        F.col("stock").cast("int").alias("ending_stock_units"),
        F.greatest(F.lit(0), F.col("stock") - F.col("safety_stock_units"))
        .cast("int")
        .alias("excess_over_safety_units"),
        F.col("stockout_days").cast("int").alias("simulated_stockout_days"),
        F.lit(run_type).alias("run_type"),
        F.date_add("issue_date", 1).alias("observed_label_window_start"),
        F.date_add("issue_date", HORIZON).alias("observed_label_window_end"),
    ).select(*EVALUATION_COLUMNS)


def summarize_policies(evaluation: "DataFrame", allocation: "DataFrame") -> "DataFrame":
    """Aggregate policy outcomes; sums come before ratios and zero denominators give null.

    Central supply is counted once per item, never once per store.
    """

    outcome = evaluation.groupBy("policy_id").agg(
        F.count(F.lit(1)).alias("evaluated_series"),
        F.sum("actual_sales_proxy_7d_units").alias("actual_units"),
        F.sum("simulated_fulfilled_units").alias("simulated_fulfilled_units"),
        F.sum("simulated_unmet_units").alias("simulated_unmet_units"),
        F.sum("ending_stock_units").alias("ending_stock_units"),
        F.sum("excess_over_safety_units").alias("excess_over_safety_units"),
        F.sum(F.when(F.col("simulated_unmet_units") > 0, 1).otherwise(0)).alias(
            "store_horizon_stockout_count"
        ),
        F.sum("simulated_stockout_days").alias("simulated_stockout_days"),
    )
    per_item = allocation.groupBy("policy_id", "item_id").agg(
        F.sum("allocated_units").alias("allocated"),
        F.first("central_available_units").alias("central"),
        F.sum("unmet_request_units").alias("unmet_request"),
    )
    supply = per_item.groupBy("policy_id").agg(
        F.sum("allocated").alias("allocated_units"),
        F.sum("central").alias("central_available_units"),
        F.sum("unmet_request").alias("unmet_request_units"),
    )
    return (
        outcome.join(supply, "policy_id")
        .withColumn(
            "demand_fulfillment_rate",
            F.when(
                F.col("actual_units") > 0,
                F.col("simulated_fulfilled_units") / F.col("actual_units"),
            ),
        )
        .withColumn(
            "central_utilization_rate",
            F.when(
                F.col("central_available_units") > 0,
                F.col("allocated_units") / F.col("central_available_units"),
            ),
        )
    )
