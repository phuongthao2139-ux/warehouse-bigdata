"""Bounded as-of history state for recursive scoring and baselines."""

from datetime import date, timedelta
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from .features import FeatureError
from .protocol import LOOKBACK_DAYS

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def build_history_state(
    sales_enriched: "DataFrame", issue_date: date, *, lookback: int = LOOKBACK_DAYS
) -> "DataFrame":
    """Return one row per eligible series with its last ``lookback`` actual days.

    Columns: ``item_id, store_id, dept_id, hist, last_known_price``. ``hist`` is ordered
    oldest to newest and ends on ``issue_date``. Only information dated on or before the
    issue date is read. A series is eligible only with a full, gap-free window; others
    are omitted (they are reported through coverage counts, never zero-filled).
    ``last_known_price`` is the latest non-null price on or before the issue date and is
    held constant through the horizon; it is null when no price was ever known.
    """

    window_start = issue_date - timedelta(days=lookback - 1)
    known = sales_enriched.filter(F.col("date") <= F.lit(issue_date))

    window = (
        known.filter(F.col("date") >= F.lit(window_start))
        .groupBy("item_id", "store_id", "dept_id")
        .agg(F.sort_array(F.collect_list(F.struct("date", "unit_sales"))).alias("pairs"))
        .filter(F.size("pairs") == lookback)
        .withColumn("hist", F.transform("pairs", lambda p: p["unit_sales"].cast("double")))
        .drop("pairs")
    )
    prices = (
        known.filter(F.col("sell_price").isNotNull())
        .groupBy("item_id", "store_id")
        .agg(F.max_by("sell_price", "date").cast("double").alias("last_known_price"))
    )
    return window.join(prices, ["item_id", "store_id"], "left")


def count_series(sales_enriched: "DataFrame") -> int:
    """Return the number of distinct item-store series in the history."""

    return sales_enriched.select("item_id", "store_id").distinct().count()


def require_eligible_series(state: "DataFrame") -> int:
    """Return the eligible series count, failing when none can be scored."""

    count = state.count()
    if count == 0:
        raise FeatureError("No series has a full gap-free history window at the issue date.")
    return count
