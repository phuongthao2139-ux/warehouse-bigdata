"""Stage 2 job: normalize, enrich, aggregate, and quality-gate the M5 sources."""

import argparse
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from warehouse_bigdata.common.identity import stable_id
from warehouse_bigdata.processing.enrich import (
    category_monthly_sales,
    enrich_sales,
    item_store_weekly_sales,
    prepare_sell_prices,
    store_daily_sales,
)
from warehouse_bigdata.processing.normalize import (
    DEFAULT_DAYS_PER_CHUNK,
    DEFAULT_FIRST_DAY,
    DEFAULT_LAST_DAY,
    normalize_sales,
    prepare_calendar,
    validate_wide_sales,
)
from warehouse_bigdata.processing.quality import (
    QualityCheck,
    check_aggregates,
    check_enrichment,
    check_sales_daily,
    require_all_passed,
)

from .paths import curated_path, staging_path

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def run_processing(
    spark: "SparkSession",
    *,
    sales_path: str,
    calendar_path: str,
    prices_path: str,
    output_root: str,
    source_snapshot_id: str,
    first_day: int = DEFAULT_FIRST_DAY,
    last_day: int = DEFAULT_LAST_DAY,
    days_per_chunk: int = DEFAULT_DAYS_PER_CHUNK,
) -> list[QualityCheck]:
    """Run J02/J03 and write curated Parquet only after every quality gate passes."""

    read = spark.read.option("header", True)
    sales_wide = read.csv(sales_path)
    calendar = prepare_calendar(read.csv(calendar_path))
    sell_prices = prepare_sell_prices(read.csv(prices_path))

    series_count = validate_wide_sales(sales_wide, first_day, last_day)
    sales_daily = normalize_sales(
        sales_wide, calendar, first_day=first_day, last_day=last_day, days_per_chunk=days_per_chunk
    ).cache()
    sales_enriched = enrich_sales(sales_daily, calendar, sell_prices)
    store_daily = store_daily_sales(sales_daily, source_snapshot_id)
    category_monthly = category_monthly_sales(sales_daily)

    checks = [
        *check_sales_daily(
            sales_daily,
            sales_wide,
            first_day=first_day,
            last_day=last_day,
            series_count=series_count,
        ),
        *check_enrichment(sales_daily, sales_enriched),
        *check_aggregates(sales_daily, store_daily, category_monthly),
    ]

    pipeline_run_id = stable_id(
        "process", snapshot=source_snapshot_id, first_day=first_day, last_day=last_day
    )
    report = json.dumps([c.as_dict() for c in checks], indent=2, default=str)
    spark.createDataFrame([(report,)], "report string").coalesce(1).write.mode("overwrite").text(
        staging_path(output_root, pipeline_run_id) + "/quality_summary"
    )
    require_all_passed(checks)

    targets = {
        "sales_daily": sales_daily,
        "sales_enriched": sales_enriched,
        "store_daily_sales": store_daily,
        "item_store_weekly_sales": item_store_weekly_sales(sales_daily),
        "category_monthly_sales": category_monthly,
    }
    for name, frame in targets.items():
        frame.write.mode("overwrite").parquet(curated_path(output_root, name, source_snapshot_id))
    sales_daily.unpersist()
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize and enrich M5 sales with Spark.")
    parser.add_argument("--master", required=True, help="Spark master URL, e.g. local[*].")
    parser.add_argument("--raw-dir", required=True, help="Directory holding the M5 CSV files.")
    parser.add_argument("--output-root", required=True, help="Root for curated/staging output.")
    parser.add_argument("--source-snapshot-id", required=True)
    parser.add_argument("--sales-file", default="sales_train_evaluation.csv")
    parser.add_argument("--first-day", type=int, default=DEFAULT_FIRST_DAY)
    parser.add_argument("--last-day", type=int, default=DEFAULT_LAST_DAY)
    parser.add_argument("--days-per-chunk", type=int, default=DEFAULT_DAYS_PER_CHUNK)
    args = parser.parse_args(argv)

    from warehouse_bigdata.common.spark import build_spark_session

    raw = args.raw_dir.rstrip("/")
    spark = build_spark_session(args.master, "warehouse-process")
    try:
        checks = run_processing(
            spark,
            sales_path=f"{raw}/{args.sales_file}",
            calendar_path=f"{raw}/calendar.csv",
            prices_path=f"{raw}/sell_prices.csv",
            output_root=args.output_root,
            source_snapshot_id=args.source_snapshot_id,
            first_day=args.first_day,
            last_day=args.last_day,
            days_per_chunk=args.days_per_chunk,
        )
    finally:
        spark.stop()
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name} observed={check.observed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
