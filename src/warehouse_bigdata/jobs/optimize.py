"""Stage 4 job: simulated scenario, replenishment, allocation, and policy evaluation."""

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from warehouse_bigdata.common.identity import stable_id
from warehouse_bigdata.optimization.allocation import POLICY_EQUAL_SHARE, POLICY_SHORTAGE_FIRST
from warehouse_bigdata.optimization.decisions import (
    allocate,
    build_replenishment,
    decision_run_id_for,
)
from warehouse_bigdata.optimization.evaluation import evaluate_policies, summarize_policies
from warehouse_bigdata.optimization.scenario import PRESETS, build_scenario

from .paths import curated_path, decision_path, forecast_path, scenario_path

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def run_optimization(
    spark: "SparkSession",
    *,
    output_root: str,
    source_snapshot_id: str,
    forecast_run_id: str,
    preset_name: str,
    evaluate: bool,
) -> dict[str, object]:
    """Decide and, for historical forecasts, evaluate both allocation policies.

    The scenario is built from sales up to the issue date only. Decisions are written
    before observed outcomes are read; ``forward_demo`` forecasts are never evaluated.
    """

    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}; choose from {sorted(PRESETS)}.")
    preset = PRESETS[preset_name]

    forecast = spark.read.parquet(forecast_path(output_root, forecast_run_id)).cache()
    run_types = [r[0] for r in forecast.select("run_type").distinct().collect()]
    if len(run_types) != 1:
        raise ValueError("Forecast must contain exactly one run_type.")
    run_type = run_types[0]
    if evaluate and run_type == "forward_demo":
        raise ValueError("forward_demo forecasts have no observed outcomes to evaluate.")

    issue_date = forecast.select("issue_date").first()[0]
    sales = spark.read.parquet(curated_path(output_root, "sales_daily", source_snapshot_id))
    eligible = forecast.select("item_id", "store_id").distinct()
    scenario, central = build_scenario(sales, eligible, issue_date, preset, source_snapshot_id)
    scenario_id = scenario.select("scenario_id").first()[0]

    policies = [POLICY_SHORTAGE_FIRST, POLICY_EQUAL_SHARE]
    decision_run_id = decision_run_id_for(forecast_run_id, scenario_id, issue_date, policies)
    replenishment, rejected = build_replenishment(
        forecast, scenario, decision_run_id=decision_run_id
    )
    replenishment = replenishment.cache()
    allocation = (
        allocate(replenishment, central, policy_id=POLICY_SHORTAGE_FIRST)
        .unionByName(allocate(replenishment, central, policy_id=POLICY_EQUAL_SHARE))
        .cache()
    )

    base = decision_path(output_root, decision_run_id)
    scenario.write.mode("overwrite").parquet(
        scenario_path(output_root, scenario_id) + "/inventory_scenario"
    )
    central.write.mode("overwrite").parquet(
        scenario_path(output_root, scenario_id) + "/central_supply_scenario"
    )
    replenishment.write.mode("overwrite").parquet(base + "/replenishment_recommendations")
    allocation.write.mode("overwrite").parquet(base + "/allocation_recommendations")
    rejected.write.mode("overwrite").parquet(base + "/rejected_series")

    result: dict[str, object] = {
        "decision_run_id": decision_run_id,
        "scenario_id": scenario_id,
        "run_type": run_type,
        "recommended_series": replenishment.count(),
        "rejected_series": rejected.count(),
    }
    if evaluate:
        evaluation_run_id = stable_id(
            "evaluation", decision_run_id=decision_run_id, run_type=run_type
        )
        evaluation = evaluate_policies(
            allocation, replenishment, sales, evaluation_run_id=evaluation_run_id, run_type=run_type
        ).cache()
        evaluation.write.mode("overwrite").parquet(base + "/inventory_policy_evaluation")
        summary = summarize_policies(evaluation, allocation)
        summary.write.mode("overwrite").parquet(base + "/policy_summary")
        result["evaluation_run_id"] = evaluation_run_id
        result["evaluated_series"] = evaluation.select("item_id", "store_id").distinct().count()
        result["excluded_series"] = (
            replenishment.select("item_id", "store_id").count() - result["evaluated_series"]
        )
        result["summary"] = [r.asDict() for r in summary.orderBy("policy_id").collect()]
        evaluation.unpersist()
    forecast.unpersist()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulated replenishment and allocation.")
    parser.add_argument("--master", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-snapshot-id", required=True)
    parser.add_argument("--forecast-run-id", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Replay holdout sales (validation/test forecasts only).",
    )
    args = parser.parse_args(argv)

    from warehouse_bigdata.common.spark import build_spark_session

    spark = build_spark_session(args.master, "warehouse-optimize")
    try:
        result = run_optimization(
            spark,
            output_root=args.output_root,
            source_snapshot_id=args.source_snapshot_id,
            forecast_run_id=args.forecast_run_id,
            preset_name=args.preset,
            evaluate=args.evaluate,
        )
    finally:
        spark.stop()
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
