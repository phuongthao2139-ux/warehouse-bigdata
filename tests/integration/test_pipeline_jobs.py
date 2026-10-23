"""Local end-to-end wiring of the stage 2-4 jobs on synthetic M5-shaped CSV files.

This verifies that artifacts flow between jobs with consistent identities. It is a
single-machine smoke test, not evidence of distributed execution or model quality.
"""

import csv
from pathlib import Path

import pytest

from tests.data import spark_fixtures as fx
from tests.data.forecast_fixtures import SMALL_PROTOCOL
from warehouse_bigdata.forecasting.model import ForestParams
from warehouse_bigdata.jobs.forecast import (
    run_evaluate,
    run_features,
    run_forecast,
    run_train,
)
from warehouse_bigdata.jobs.optimize import run_optimization
from warehouse_bigdata.jobs.process import run_processing

DAYS = 60
SNAPSHOT = "synthetic-snapshot"


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> str:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    return str(path)


@pytest.fixture(scope="module")
def pipeline(spark, tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    root = str(tmp_path_factory.mktemp("warehouse"))
    sales_rows, _ = fx.wide_sales_rows(DAYS)
    sales_path = _write_csv(
        raw / "sales.csv",
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        + [f"d_{d}" for d in range(1, DAYS + 1)],
        sales_rows,
    )
    calendar_path = _write_csv(
        raw / "calendar.csv",
        [
            "d",
            "date",
            "wm_yr_wk",
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
        ],
        fx.calendar_rows(DAYS + 14),
    )
    prices_path = _write_csv(
        raw / "prices.csv", ["store_id", "item_id", "wm_yr_wk", "sell_price"], fx.price_rows()
    )

    checks = run_processing(
        spark,
        sales_path=sales_path,
        calendar_path=calendar_path,
        prices_path=prices_path,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        first_day=1,
        last_day=DAYS,
        days_per_chunk=25,
    )
    feature_rows = run_features(
        spark, output_root=root, source_snapshot_id=SNAPSHOT, protocol=SMALL_PROTOCOL
    )
    trained = run_train(
        spark,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        target_rows=1000,
        params=ForestParams(num_trees=5, max_depth=4),
        protocol=SMALL_PROTOCOL,
    )
    return {
        "root": root,
        "calendar": calendar_path,
        "checks": checks,
        "feature_rows": feature_rows,
        "trained": trained,
    }


def test_processing_gates_pass_and_features_exclude_lookback(pipeline):
    assert all(check.passed for check in pipeline["checks"])
    # 3 series x (60 days - 28 lookback days).
    assert pipeline["feature_rows"] == 3 * (DAYS - 28)
    trained = pipeline["trained"]
    assert trained["feature_eligible_rows"] == pipeline["feature_rows"]
    # Training rows come only from the train role (days 29..45) and never exceed it.
    assert 0 < trained["training_rows"] <= 3 * (45 - 28)


def test_evaluate_validation_and_forecast_and_optimize(spark, pipeline):
    root = pipeline["root"]
    model_run_id = pipeline["trained"]["model_run_id"]

    evaluation_run_id = run_evaluate(
        spark,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        model_run_id=model_run_id,
        calendar_path=pipeline["calendar"],
        run_type="validation",
        protocol=SMALL_PROTOCOL,
    )
    metrics = spark.read.parquet(f"{root}/outputs/evaluation/{evaluation_run_id}/forecast_metrics")
    pooled = metrics.filter("holdout_id='validation' AND horizon=0 AND metric_name='MAE'")
    assert {r["method_id"] for r in pooled.collect()} == {"random_forest", "seasonal_naive"}
    assert {r["observation_count"] for r in pooled.collect()} == {21}

    forecast_run_id, series = run_forecast(
        spark,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        model_run_id=model_run_id,
        calendar_path=pipeline["calendar"],
        run_type="validation",
        issue_day=45,
    )
    assert series == 3

    result = run_optimization(
        spark,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        forecast_run_id=forecast_run_id,
        preset_name="scarce_supply",
        evaluate=True,
    )
    assert result["run_type"] == "validation"
    assert result["recommended_series"] == 3
    assert result["rejected_series"] == 0
    assert result["evaluated_series"] == 3
    assert {row["policy_id"] for row in result["summary"]} == {
        "shortage_first_proportional",
        "equal_share",
    }


def test_forward_demo_is_forecast_and_decided_but_never_evaluated(spark, pipeline):
    root = pipeline["root"]
    model_run_id = pipeline["trained"]["model_run_id"]
    forecast_run_id, series = run_forecast(
        spark,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        model_run_id=model_run_id,
        calendar_path=pipeline["calendar"],
        run_type="forward_demo",
    )
    assert series == 3
    daily = spark.read.parquet(f"{root}/outputs/forecasts/{forecast_run_id}")
    assert {r["run_type"] for r in daily.collect()} == {"forward_demo"}
    assert daily.agg({"horizon": "max"}).first()[0] == 7

    result = run_optimization(
        spark,
        output_root=root,
        source_snapshot_id=SNAPSHOT,
        forecast_run_id=forecast_run_id,
        preset_name="balanced",
        evaluate=False,
    )
    assert result["recommended_series"] == 3
    assert "evaluation_run_id" not in result
    with pytest.raises(ValueError, match="forward_demo"):
        run_optimization(
            spark,
            output_root=root,
            source_snapshot_id=SNAPSHOT,
            forecast_run_id=forecast_run_id,
            preset_name="balanced",
            evaluate=True,
        )


def test_forecast_run_type_and_issue_day_rules(spark, pipeline):
    kwargs = dict(
        output_root=pipeline["root"],
        source_snapshot_id=SNAPSHOT,
        model_run_id=pipeline["trained"]["model_run_id"],
        calendar_path=pipeline["calendar"],
    )
    with pytest.raises(ValueError, match="issue_day"):
        run_forecast(spark, run_type="validation", **kwargs)
    with pytest.raises(ValueError, match="forward_demo"):
        run_forecast(spark, run_type="forward_demo", issue_day=45, **kwargs)
