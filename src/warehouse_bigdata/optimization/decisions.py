"""Spark replenishment and allocation decisions (docs/08 sections 7-10)."""

from datetime import date
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

from warehouse_bigdata.common.identity import stable_id

from .allocation import (
    POLICY_EQUAL_SHARE,
    POLICY_SHORTAGE_FIRST,
    POLICY_VERSION,
    allocate_equal_share,
    allocate_shortage_first,
)
from .scenario import ORIGIN_LABEL

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

REPLENISHMENT_POLICY_VERSION = "replenishment_v1"
HORIZON = 7

REPLENISHMENT_COLUMNS = (
    "decision_run_id",
    "source_snapshot_id",
    "forecast_run_id",
    "model_run_id",
    "scenario_id",
    "policy_version",
    "issue_date",
    "item_id",
    "store_id",
    "forecast_7d_units",
    "planning_demand_units",
    "opening_stock_units",
    "safety_stock_units",
    "target_stock_units",
    "critical_need_units",
    "safety_need_units",
    "request_units",
    "forecast_shortage_flag",
    "origin_label",
)

ALLOCATION_COLUMNS = (
    "decision_run_id",
    "source_snapshot_id",
    "forecast_run_id",
    "model_run_id",
    "scenario_id",
    "policy_id",
    "policy_version",
    "issue_date",
    "item_id",
    "store_id",
    "source_warehouse_id",
    "requested_units",
    "critical_allocated_units",
    "safety_allocated_units",
    "allocated_units",
    "unmet_request_units",
    "central_available_units",
)


class DecisionError(ValueError):
    """Raised when forecast and scenario inputs cannot form a consistent decision batch."""


def decision_run_id_for(
    forecast_run_id: str, scenario_id: str, issue_date: date, policy_ids: list[str]
) -> str:
    """Return the identity of one decision batch bound to its forecast, scenario, and policies."""

    return stable_id(
        "decision",
        forecast_run_id=forecast_run_id,
        scenario_id=scenario_id,
        issue_date=issue_date.isoformat(),
        policies=sorted(policy_ids),
        replenishment=REPLENISHMENT_POLICY_VERSION,
        allocation=POLICY_VERSION,
    )


def _single_value(df: "DataFrame", column: str, what: str) -> str:
    values = [r[0] for r in df.select(column).distinct().limit(2).collect()]
    if len(values) != 1:
        raise DecisionError(f"Expected exactly one {what}, found {len(values)}.")
    return values[0]


def build_replenishment(
    forecast_daily: "DataFrame",
    scenario: "DataFrame",
    *,
    decision_run_id: str,
) -> tuple["DataFrame", "DataFrame"]:
    """Return ``(replenishment_recommendations, rejected)``.

    The forecast must belong to one forecast run and issue date, and the scenario to one
    scenario with the same issue date. Series without exactly seven valid, finite,
    nonnegative predictions for horizons 1-7, or without a scenario row, are rejected
    with a reason instead of being defaulted to zero. The seven unrounded predictions are
    summed exactly (decimal) before one ceiling.
    """

    forecast_run_id = _single_value(forecast_daily, "forecast_run_id", "forecast_run_id")
    issue_date = _single_value(forecast_daily, "issue_date", "issue_date")
    _single_value(forecast_daily, "model_run_id", "model_run_id")
    scenario_id = _single_value(scenario, "scenario_id", "scenario_id")
    if _single_value(scenario, "issue_date", "scenario issue_date") != issue_date:
        raise DecisionError("Scenario issue_date does not match the forecast issue_date.")
    _single_value(scenario, "source_snapshot_id", "scenario source_snapshot_id")

    valid_prediction = (
        F.col("predicted_units").isNotNull()
        & ~F.isnan("predicted_units")
        & (F.col("predicted_units") >= 0)
        & (F.col("predicted_units") < F.lit(float("inf")))
    )
    per_series = forecast_daily.groupBy("item_id", "store_id").agg(
        F.first("model_run_id").alias("model_run_id"),
        F.first("source_snapshot_id").alias("source_snapshot_id"),
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("horizon").alias("horizons"),
        F.min("horizon").alias("min_h"),
        F.max("horizon").alias("max_h"),
        F.min(valid_prediction.cast("int")).alias("all_valid"),
        F.sum(F.col("predicted_units").cast("decimal(38,12)")).alias("forecast_sum"),
    )
    complete = (
        (F.col("rows") == HORIZON)
        & (F.col("horizons") == HORIZON)
        & (F.col("min_h") == 1)
        & (F.col("max_h") == HORIZON)
        & (F.col("all_valid") == 1)
    )
    joined = per_series.join(
        scenario.select(
            "item_id", "store_id", "opening_stock_units", "safety_stock_units", "source_snapshot_id"
        ).withColumnRenamed("source_snapshot_id", "scenario_snapshot"),
        ["item_id", "store_id"],
        "left",
    )
    rejected = joined.filter(~complete | F.col("opening_stock_units").isNull()).select(
        "item_id",
        "store_id",
        F.lit(forecast_run_id).alias("forecast_run_id"),
        F.when(~complete, "INVALID_FORECAST").otherwise("MISSING_SCENARIO").alias("reason_code"),
    )

    ok = joined.filter(complete & F.col("opening_stock_units").isNotNull())
    demand = F.ceil(F.col("forecast_sum")).cast("int")
    request = F.greatest(
        F.lit(0), demand + F.col("safety_stock_units") - F.col("opening_stock_units")
    )
    critical = F.greatest(F.lit(0), demand - F.col("opening_stock_units"))
    recommendations = ok.select(
        F.lit(decision_run_id).alias("decision_run_id"),
        "source_snapshot_id",
        F.lit(forecast_run_id).alias("forecast_run_id"),
        "model_run_id",
        F.lit(scenario_id).alias("scenario_id"),
        F.lit(REPLENISHMENT_POLICY_VERSION).alias("policy_version"),
        F.lit(issue_date).alias("issue_date"),
        "item_id",
        "store_id",
        F.col("forecast_sum").cast("double").alias("forecast_7d_units"),
        demand.alias("planning_demand_units"),
        "opening_stock_units",
        "safety_stock_units",
        (demand + F.col("safety_stock_units")).alias("target_stock_units"),
        critical.alias("critical_need_units"),
        (request - critical).alias("safety_need_units"),
        request.alias("request_units"),
        (F.col("opening_stock_units") < demand).alias("forecast_shortage_flag"),
        F.lit(ORIGIN_LABEL).alias("origin_label"),
    )
    return recommendations.select(*REPLENISHMENT_COLUMNS), rejected


_ALLOCATION_RESULT = ArrayType(
    StructType(
        [
            StructField("store_id", StringType()),
            StructField("critical_allocated_units", IntegerType()),
            StructField("safety_allocated_units", IntegerType()),
        ]
    )
)


def _shortage_first_udf(stores, central_available):
    critical = {s["store_id"]: s["critical"] for s in stores}
    safety = {s["store_id"]: s["safety"] for s in stores}
    first, second = allocate_shortage_first(critical, safety, central_available)
    return [(s, first[s], second[s]) for s in sorted(first)]


def _equal_share_rows(stores, central_available):
    requests = {s["store_id"]: s["critical"] + s["safety"] for s in stores}
    shares = allocate_equal_share(requests, central_available)
    return [(s, shares[s], 0) for s in sorted(shares)]


def allocate(
    replenishment: "DataFrame",
    central_supply: "DataFrame",
    *,
    policy_id: str,
) -> "DataFrame":
    """Allocate each item's central supply across its stores under ``policy_id``.

    Items are independent; a request, supply, and result stay within one ``item_id``.
    Every store of an item keeps a row (zero allocation when it requests nothing).
    The equal-share baseline reports its units in ``allocated_units`` and leaves the
    stage split null because it has no critical/safety stages.
    """

    if policy_id not in (POLICY_SHORTAGE_FIRST, POLICY_EQUAL_SHARE):
        raise DecisionError(f"Unknown policy_id: {policy_id}")

    if policy_id == POLICY_SHORTAGE_FIRST:
        allocator = F.udf(_shortage_first_udf, _ALLOCATION_RESULT)
    else:
        allocator = F.udf(_equal_share_rows, _ALLOCATION_RESULT)

    items = replenishment.join(
        central_supply.select("item_id", "source_warehouse_id", "central_available_units"),
        "item_id",
    )
    grouped = items.groupBy("item_id").agg(
        F.first("central_available_units").alias("central_available_units"),
        F.first("source_warehouse_id").alias("source_warehouse_id"),
        F.collect_list(
            F.struct(
                F.col("store_id"),
                F.col("critical_need_units").alias("critical"),
                F.col("safety_need_units").alias("safety"),
            )
        ).alias("stores"),
    )
    solved = grouped.withColumn(
        "result", F.explode(allocator(F.col("stores"), F.col("central_available_units")))
    ).select(
        "item_id",
        "central_available_units",
        "source_warehouse_id",
        F.col("result.store_id").alias("store_id"),
        F.col("result.critical_allocated_units").alias("c_alloc"),
        F.col("result.safety_allocated_units").alias("s_alloc"),
    )
    joined = replenishment.join(solved, ["item_id", "store_id"])
    shortage_first = policy_id == POLICY_SHORTAGE_FIRST
    critical_alloc = F.col("c_alloc") if shortage_first else F.lit(None).cast("int")
    safety_alloc = F.col("s_alloc") if shortage_first else F.lit(None).cast("int")
    allocated = F.col("c_alloc") + F.col("s_alloc") if shortage_first else F.col("c_alloc")
    return joined.select(
        "decision_run_id",
        "source_snapshot_id",
        "forecast_run_id",
        "model_run_id",
        "scenario_id",
        F.lit(policy_id).alias("policy_id"),
        F.lit(POLICY_VERSION).alias("policy_version"),
        "issue_date",
        "item_id",
        "store_id",
        "source_warehouse_id",
        F.col("request_units").alias("requested_units"),
        critical_alloc.alias("critical_allocated_units"),
        safety_alloc.alias("safety_allocated_units"),
        allocated.cast("int").alias("allocated_units"),
        (F.col("request_units") - allocated).cast("int").alias("unmet_request_units"),
        "central_available_units",
    ).select(*ALLOCATION_COLUMNS)
