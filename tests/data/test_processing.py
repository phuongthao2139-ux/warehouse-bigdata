"""Processing stage behavior on tiny synthetic M5-shaped inputs."""

import pytest

from tests.data import spark_fixtures as fx
from warehouse_bigdata.contracts import SALES_DAILY, SALES_ENRICHED, STORE_DAILY_SALES
from warehouse_bigdata.processing.enrich import (
    category_monthly_sales,
    enrich_sales,
    prepare_sell_prices,
    store_daily_sales,
)
from warehouse_bigdata.processing.normalize import (
    ProcessingError,
    day_chunks,
    normalize_sales,
    prepare_calendar,
    require_valid_units,
)
from warehouse_bigdata.processing.quality import (
    check_aggregates,
    check_enrichment,
    check_sales_daily,
    require_all_passed,
)


@pytest.fixture(scope="module")
def sources(spark):
    rows, schema = fx.wide_sales_rows()
    wide = spark.createDataFrame(rows, schema)
    calendar = prepare_calendar(spark.createDataFrame(fx.calendar_rows(), fx.CALENDAR_SCHEMA))
    prices = prepare_sell_prices(spark.createDataFrame(fx.price_rows(), fx.PRICE_SCHEMA))
    return wide, calendar, prices


def test_day_chunks_are_contiguous_and_bounded():
    assert day_chunks(1, 10, 4) == [(1, 4), (5, 8), (9, 10)]
    with pytest.raises(ProcessingError):
        day_chunks(5, 1, 4)


def test_normalize_matches_contract_and_cell_values(spark, sources):
    wide, calendar, _ = sources
    daily = normalize_sales(wide, calendar, first_day=1, last_day=fx.DAYS, days_per_chunk=5)

    assert tuple(daily.columns) == SALES_DAILY.field_names
    assert daily.count() == len(fx.SERIES) * fx.DAYS

    cell = daily.filter("item_id='FOODS_1_001' AND store_id='TX_1' AND day_key='d_4'").first()
    assert cell["unit_sales"] == fx.sales_values(1)[3]
    assert str(cell["date"]) == "2020-01-04"
    assert cell["series_id"] == "FOODS_1_001_TX_1_evaluation"
    # Zero-sale rows are retained, not dropped.
    assert daily.filter("unit_sales = 0").count() > 0
    # Calendar days beyond the requested range are never turned into observed sales.
    assert daily.agg({"day_index": "max"}).first()[0] == fx.DAYS


def test_invalid_units_are_rejected_not_zeroed(spark, sources):
    _, calendar, _ = sources
    rows, schema = fx.wide_sales_rows()
    rows[0] = (*rows[0][:6], "x", *rows[0][7:])
    wide = spark.createDataFrame(rows, schema)
    daily = normalize_sales(wide, calendar, first_day=1, last_day=fx.DAYS)
    with pytest.raises(ProcessingError, match="non-integer"):
        require_valid_units(daily)


def test_duplicate_series_and_missing_days_fail(spark, sources):
    _, calendar, _ = sources
    rows, schema = fx.wide_sales_rows()
    duplicated = spark.createDataFrame([*rows, rows[0]], schema)
    with pytest.raises(ProcessingError, match="exactly once"):
        normalize_sales(duplicated, calendar, first_day=1, last_day=fx.DAYS)
    wide = spark.createDataFrame(rows, schema)
    with pytest.raises(ProcessingError, match="missing"):
        normalize_sales(wide, calendar, first_day=1, last_day=fx.DAYS + 1)


def test_duplicate_calendar_key_fails(spark):
    rows = fx.calendar_rows()
    with pytest.raises(ProcessingError, match="unique"):
        prepare_calendar(spark.createDataFrame([*rows, rows[0]], fx.CALENDAR_SCHEMA))


def test_duplicate_price_key_fails(spark):
    rows = fx.price_rows()
    with pytest.raises(ProcessingError, match="one row per"):
        prepare_sell_prices(spark.createDataFrame([*rows, rows[0]], fx.PRICE_SCHEMA))


def test_enrichment_preserves_grain_and_flags_missing_price(spark, sources):
    wide, calendar, prices = sources
    daily = normalize_sales(wide, calendar, first_day=1, last_day=fx.DAYS)
    enriched = enrich_sales(daily, calendar, prices)

    assert tuple(enriched.columns) == SALES_ENRICHED.field_names
    assert enriched.count() == daily.count()

    priced = enriched.filter("item_id='FOODS_1_001' AND store_id='CA_1' AND day_key='d_9'").first()
    assert priced["sell_price"] == pytest.approx(2.75)
    assert priced["price_missing"] is False

    unpriced = enriched.filter("item_id='HOBBIES_1_002'").first()
    assert unpriced["sell_price"] is None
    assert unpriced["price_missing"] is True

    # TX_1 has no price in week 11001, so it must stay null (not zero).
    late_tx = enriched.filter("store_id='TX_1' AND day_key='d_9'").first()
    assert late_tx["sell_price"] is None

    event_day = enriched.filter("day_key='d_4'").first()
    assert event_day["event_name_1"] == "TestEvent"
    assert enriched.filter("day_key='d_5'").first()["event_name_1"] is None

    # SNAP is chosen by the series' own state: TX_1 uses snap_TX which is always 0.
    assert enriched.filter("state_id='TX'").filter("snap_applicable").count() == 0
    assert enriched.filter("state_id='CA' AND day_key='d_1'").first()["snap_applicable"] is True


def test_aggregates_and_quality_gates_pass(spark, sources):
    wide, calendar, prices = sources
    daily = normalize_sales(wide, calendar, first_day=1, last_day=fx.DAYS)
    enriched = enrich_sales(daily, calendar, prices)
    store_daily = store_daily_sales(daily, "synthetic-snapshot")
    monthly = category_monthly_sales(daily)

    assert tuple(store_daily.columns) == STORE_DAILY_SALES.field_names
    ca_day_1 = store_daily.filter("store_id='CA_1' AND date='2020-01-01'").first()
    assert ca_day_1["total_units"] == fx.sales_values(0)[0] + fx.sales_values(2)[0]
    assert ca_day_1["observed_series_count"] == 2

    checks = [
        *check_sales_daily(daily, wide, first_day=1, last_day=fx.DAYS, series_count=len(fx.SERIES)),
        *check_enrichment(daily, enriched),
        *check_aggregates(daily, store_daily, monthly),
    ]
    require_all_passed(checks)
    total = sum(sum(fx.sales_values(i)) for i in range(len(fx.SERIES)))
    named = {c.name: c for c in checks}
    assert named["sales_daily.total_units_match_source"].observed == total


def test_quality_gate_reports_failures(spark, sources):
    wide, calendar, _ = sources
    daily = normalize_sales(wide, calendar, first_day=1, last_day=fx.DAYS)
    checks = check_sales_daily(
        daily.filter("day_index > 1"), wide, first_day=1, last_day=fx.DAYS, series_count=3
    )
    with pytest.raises(ValueError, match="row_count"):
        require_all_passed(checks)
