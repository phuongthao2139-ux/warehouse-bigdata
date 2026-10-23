"""Stage 3 jobs: features, training, evaluation, and forecast scoring."""

import argparse
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from warehouse_bigdata.common.identity import stable_id
from warehouse_bigdata.forecasting.evaluation import compute_forecast_metrics, score_origin
from warehouse_bigdata.forecasting.features import (
    assert_contiguous_daily,
    build_forecast_features,
    sample_training_rows,
    to_model_input,
)
from warehouse_bigdata.forecasting.history import build_history_state, require_eligible_series
from warehouse_bigdata.forecasting.inference import recursive_forecast, to_forecast_daily
from warehouse_bigdata.forecasting.model import ForestParams, model_run_id_for, train_model
from warehouse_bigdata.forecasting.protocol import FEATURE_VERSION, M5_PROTOCOL, Protocol
from warehouse_bigdata.processing.normalize import prepare_calendar

from .paths import curated_path, evaluation_path, features_path, forecast_path, model_path

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def date_for_day(sales: "DataFrame", day_index: int) -> date:
    """Return the calendar date of ``day_index`` from curated sales."""

    row = sales.filter(F.col("day_index") == day_index).select("date").limit(1).first()
    if row is None:
        raise ValueError(f"day_index {day_index} is not present in the curated sales.")
    return row["date"]


def run_features(
    spark: "SparkSession",
    *,
    output_root: str,
    source_snapshot_id: str,
    protocol: Protocol = M5_PROTOCOL,
) -> int:
    """Build and write the historical one-step feature table; return its row count."""

    sales = spark.read.parquet(curated_path(output_root, "sales_enriched", source_snapshot_id))
    assert_contiguous_daily(sales)
    features = build_forecast_features(sales, source_snapshot_id, protocol=protocol)
    target = features_path(output_root, source_snapshot_id, FEATURE_VERSION)
    features.write.mode("overwrite").partitionBy("feature_role").parquet(target)
    return spark.read.parquet(target).count()


def run_train(
    spark: "SparkSession",
    *,
    output_root: str,
    source_snapshot_id: str,
    target_rows: int,
    sample_seed: int = 42,
    params: ForestParams | None = None,
    protocol: Protocol = M5_PROTOCOL,
) -> dict[str, object]:
    """Fit on a deterministic sample of training-role rows only and save the model."""

    params = params or ForestParams()
    features = spark.read.parquet(features_path(output_root, source_snapshot_id, FEATURE_VERSION))
    sample = sample_training_rows(features, target_rows=target_rows, seed=sample_seed).cache()
    training_rows = sample.count()
    model_run_id = model_run_id_for(
        params,
        source_snapshot_id=source_snapshot_id,
        feature_version=FEATURE_VERSION,
        train_last_day=protocol.train_last_day,
        training_rows=training_rows,
        sample_seed=sample_seed,
    )
    model = train_model(to_model_input(sample, with_label=True), params)
    model.write().overwrite().save(model_path(output_root, model_run_id))
    sample.select("item_id", "store_id", "target_date").write.mode("overwrite").parquet(
        model_path(output_root, model_run_id) + "_training_keys"
    )
    summary = {
        "model_run_id": model_run_id,
        "feature_eligible_rows": features.count(),
        "training_rows": training_rows,
        "requested_sample_rows": target_rows,
    }
    sample.unpersist()
    return summary


def _load_calendar(spark: "SparkSession", calendar_path: str) -> "DataFrame":
    return prepare_calendar(spark.read.option("header", True).csv(calendar_path))


def run_evaluate(
    spark: "SparkSession",
    *,
    output_root: str,
    source_snapshot_id: str,
    model_run_id: str,
    calendar_path: str,
    run_type: str,
    protocol: Protocol = M5_PROTOCOL,
) -> str:
    """Score every origin of ``run_type`` and write pooled and per-window metrics."""

    sales = spark.read.parquet(curated_path(output_root, "sales_enriched", source_snapshot_id))
    calendar = _load_calendar(spark, calendar_path)
    model = PipelineModel.load(model_path(output_root, model_run_id))
    origins = protocol.origins_for(run_type)
    evaluation_run_id = stable_id(
        "evaluation",
        model_run_id=model_run_id,
        run_type=run_type,
        origins=list(origins),
        source_snapshot_id=source_snapshot_id,
    )

    scored: DataFrame | None = None
    for position, day_index in enumerate(origins, start=1):
        window = score_origin(
            model,
            sales,
            calendar,
            date_for_day(sales, day_index),
            window_id=f"{run_type}_w{position}",
        )
        scored = window if scored is None else scored.unionByName(window)
    assert scored is not None
    scored = scored.cache()

    def metrics_for(frame: "DataFrame", holdout_id: str) -> "DataFrame":
        return compute_forecast_metrics(
            frame,
            evaluation_run_id=evaluation_run_id,
            model_run_id=model_run_id,
            holdout_id=holdout_id,
            run_type=run_type,
        )

    result = metrics_for(scored, run_type)
    for position in range(1, len(origins) + 1):
        window_id = f"{run_type}_w{position}"
        result = result.unionByName(
            metrics_for(scored.filter(F.col("window_id") == window_id), window_id)
        )
    result.write.mode("overwrite").parquet(
        evaluation_path(output_root, evaluation_run_id) + "/forecast_metrics"
    )
    scored.unpersist()
    return evaluation_run_id


def run_forecast(
    spark: "SparkSession",
    *,
    output_root: str,
    source_snapshot_id: str,
    model_run_id: str,
    calendar_path: str,
    run_type: str,
    issue_day: int | None = None,
) -> tuple[str, int]:
    """Write seven-day forecasts for eligible series; return ``(forecast_run_id, series)``.

    ``forward_demo`` always issues on the last observed day and has no observed outcome.
    ``validation``/``test`` require an explicit historical ``issue_day``.
    """

    sales = spark.read.parquet(curated_path(output_root, "sales_enriched", source_snapshot_id))
    calendar = _load_calendar(spark, calendar_path)
    if run_type == "forward_demo":
        if issue_day is not None:
            raise ValueError("forward_demo issues on the last observed day; do not pass issue_day.")
        issue_date = sales.agg(F.max("date")).first()[0]
    else:
        if issue_day is None:
            raise ValueError("validation/test forecasts require issue_day.")
        issue_date = date_for_day(sales, issue_day)

    model = PipelineModel.load(model_path(output_root, model_run_id))
    state = build_history_state(sales, issue_date).cache()
    eligible = require_eligible_series(state)
    forecast_run_id = stable_id(
        "forecast",
        model_run_id=model_run_id,
        issue_date=issue_date.isoformat(),
        run_type=run_type,
        source_snapshot_id=source_snapshot_id,
    )
    daily = to_forecast_daily(
        recursive_forecast(model, state, calendar, issue_date),
        forecast_run_id=forecast_run_id,
        model_run_id=model_run_id,
        source_snapshot_id=source_snapshot_id,
        feature_version=FEATURE_VERSION,
        issue_date=issue_date,
        run_type=run_type,
    )
    daily.write.mode("overwrite").parquet(forecast_path(output_root, forecast_run_id))
    state.unpersist()
    return forecast_run_id, eligible


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forecasting jobs (Spark MLlib).")
    parser.add_argument("--master", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-snapshot-id", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("features")
    train = sub.add_parser("train")
    train.add_argument("--target-rows", type=int, default=200_000)
    train.add_argument("--sample-seed", type=int, default=42)
    train.add_argument("--num-trees", type=int, default=20)
    train.add_argument("--max-depth", type=int, default=6)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model-run-id", required=True)
    evaluate.add_argument("--calendar-path", required=True)
    evaluate.add_argument("--run-type", choices=["validation", "test"], required=True)
    forecast = sub.add_parser("forecast")
    forecast.add_argument("--model-run-id", required=True)
    forecast.add_argument("--calendar-path", required=True)
    forecast.add_argument(
        "--run-type", choices=["validation", "test", "forward_demo"], required=True
    )
    forecast.add_argument("--issue-day", type=int)
    args = parser.parse_args(argv)

    from warehouse_bigdata.common.spark import build_spark_session

    spark = build_spark_session(args.master, f"warehouse-forecast-{args.command}")
    try:
        common = {"output_root": args.output_root, "source_snapshot_id": args.source_snapshot_id}
        if args.command == "features":
            print(f"feature_rows={run_features(spark, **common)}")
        elif args.command == "train":
            params = ForestParams(
                num_trees=args.num_trees, max_depth=args.max_depth, seed=args.sample_seed
            )
            print(
                run_train(
                    spark,
                    **common,
                    target_rows=args.target_rows,
                    sample_seed=args.sample_seed,
                    params=params,
                )
            )
        elif args.command == "evaluate":
            print(
                "evaluation_run_id="
                + run_evaluate(
                    spark,
                    **common,
                    model_run_id=args.model_run_id,
                    calendar_path=args.calendar_path,
                    run_type=args.run_type,
                )
            )
        else:
            forecast_run_id, series = run_forecast(
                spark,
                **common,
                model_run_id=args.model_run_id,
                calendar_path=args.calendar_path,
                run_type=args.run_type,
                issue_day=args.issue_day,
            )
            print(f"forecast_run_id={forecast_run_id} scored_series={series}")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
