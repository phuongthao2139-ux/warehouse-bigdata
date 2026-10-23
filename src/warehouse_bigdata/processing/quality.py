"""Data-quality gates for the processing stage."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from pyspark.sql import functions as F

from .normalize import day_chunks, day_columns

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass(frozen=True, slots=True)
class QualityCheck:
    """One measured check. ``expected`` is ``None`` for pure invariant checks."""

    name: str
    passed: bool
    observed: Any
    expected: Any = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _equals(name: str, observed: Any, expected: Any) -> QualityCheck:
    return QualityCheck(name, observed == expected, observed, expected)


def wide_total_units(sales_wide: "DataFrame", first_day: int, last_day: int) -> int:
    """Sum the source ``d_*`` cells in bounded column groups."""

    columns = day_columns(sales_wide)
    by_index = {int(c[2:]): c for c in columns}
    total = 0
    for start, end in day_chunks(first_day, last_day, 64):
        cells = [F.col(by_index[d]).cast("long") for d in range(start, end + 1)]
        row = sales_wide.agg(*[F.sum(c).alias(f"s{i}") for i, c in enumerate(cells)]).first()
        total += sum(int(v or 0) for v in row)
    return total


def check_sales_daily(
    sales_daily: "DataFrame",
    sales_wide: "DataFrame",
    *,
    first_day: int,
    last_day: int,
    series_count: int,
) -> list[QualityCheck]:
    """Reconcile ``sales_daily`` against the source matrix."""

    stats = sales_daily.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("item_id", "store_id", "date").alias("keys"),
        F.sum("unit_sales").alias("units"),
        F.min("day_index").alias("min_day"),
        F.max("day_index").alias("max_day"),
        F.sum(
            F.when(F.col("unit_sales").isNull() | (F.col("unit_sales") < 0), 1).otherwise(0)
        ).alias("bad_units"),
        F.sum(F.when(F.col("date").isNull(), 1).otherwise(0)).alias("null_dates"),
        F.sum(F.when(F.col("unit_sales") == 0, 1).otherwise(0)).alias("zero_rows"),
    ).first()
    days = last_day - first_day + 1
    return [
        _equals("sales_daily.row_count", stats["rows"], series_count * days),
        _equals("sales_daily.unique_key_count", stats["keys"], stats["rows"]),
        _equals(
            "sales_daily.day_range", (stats["min_day"], stats["max_day"]), (first_day, last_day)
        ),
        _equals("sales_daily.invalid_unit_sales", stats["bad_units"], 0),
        _equals("sales_daily.null_dates", stats["null_dates"], 0),
        _equals(
            "sales_daily.total_units_match_source",
            int(stats["units"] or 0),
            wide_total_units(sales_wide, first_day, last_day),
        ),
        QualityCheck("sales_daily.zero_sale_rows_retained", True, int(stats["zero_rows"] or 0)),
    ]


def check_enrichment(sales_daily: "DataFrame", sales_enriched: "DataFrame") -> list[QualityCheck]:
    """Check the price join preserved the fact cardinality and total units."""

    before = sales_daily.agg(F.count(F.lit(1)).alias("n"), F.sum("unit_sales").alias("u")).first()
    after = sales_enriched.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum("unit_sales").alias("u"),
        F.sum(F.when(F.col("price_missing"), 1).otherwise(0)).alias("missing"),
        F.sum(F.when(F.col("sell_price") == 0, 1).otherwise(0)).alias("zero_price"),
    ).first()
    return [
        _equals("sales_enriched.row_count_preserved", after["n"], before["n"]),
        _equals("sales_enriched.total_units_preserved", after["u"], before["u"]),
        _equals("sales_enriched.zero_prices", after["zero_price"], 0),
        QualityCheck("sales_enriched.price_missing_rows", True, int(after["missing"] or 0)),
    ]


def check_aggregates(
    sales_daily: "DataFrame",
    store_daily: "DataFrame",
    category_monthly: "DataFrame",
) -> list[QualityCheck]:
    """Check aggregate totals equal the fact total over the same scope."""

    fact = sales_daily.agg(F.sum("unit_sales")).first()[0] or 0
    store_total = store_daily.agg(F.sum("total_units")).first()[0] or 0
    category_total = category_monthly.agg(F.sum("total_units")).first()[0] or 0
    return [
        _equals("store_daily_sales.total_units_reconciled", int(store_total), int(fact)),
        _equals("category_monthly_sales.total_units_reconciled", int(category_total), int(fact)),
    ]


def require_all_passed(checks: list[QualityCheck]) -> None:
    """Raise when any check failed, listing every failure."""

    failed = [c for c in checks if not c.passed]
    if failed:
        details = "; ".join(
            f"{c.name}: observed={c.observed!r} expected={c.expected!r}" for c in failed
        )
        raise ValueError(f"Data-quality gate failed: {details}")
