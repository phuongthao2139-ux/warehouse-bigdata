"""J02: wide-to-long normalization of the M5 sales matrix into ``sales_daily``."""

import re
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

ID_FIELDS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
DEFAULT_FIRST_DAY = 1
DEFAULT_LAST_DAY = 1941
DEFAULT_DAYS_PER_CHUNK = 128

CALENDAR_COLUMNS = (
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
)

SALES_DAILY_COLUMNS = (
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "series_id",
    "day_key",
    "day_index",
    "date",
    "wm_yr_wk",
    "unit_sales",
)

_DAY_KEY = re.compile(r"^d_(\d+)$")


class ProcessingError(ValueError):
    """Raised when source data violates a documented processing precondition."""


def day_key(day_index: int) -> str:
    """Return the M5 day key (``d_<n>``) for a day index."""

    return f"d_{day_index}"


def day_chunks(first_day: int, last_day: int, days_per_chunk: int) -> list[tuple[int, int]]:
    """Split ``first_day..last_day`` (inclusive) into bounded, contiguous chunks."""

    if first_day < 1 or last_day < first_day:
        raise ProcessingError(f"Invalid day range {first_day}..{last_day}.")
    if days_per_chunk < 1:
        raise ProcessingError("days_per_chunk must be at least 1.")
    return [
        (start, min(start + days_per_chunk - 1, last_day))
        for start in range(first_day, last_day + 1, days_per_chunk)
    ]


def day_columns(sales_wide: "DataFrame") -> list[str]:
    """Return the ``d_<n>`` columns of a wide sales frame ordered by day index."""

    found = [(int(m.group(1)), c) for c in sales_wide.columns if (m := _DAY_KEY.match(c))]
    return [column for _, column in sorted(found)]


def prepare_calendar(calendar: "DataFrame") -> "DataFrame":
    """Return a typed calendar with one validated row per day key.

    Raises ``ProcessingError`` when required columns are missing, ``d`` is null or
    duplicated, or a date does not parse.
    """

    missing = [c for c in CALENDAR_COLUMNS if c not in calendar.columns]
    if missing:
        raise ProcessingError(f"calendar is missing columns: {missing}")

    typed = calendar.select(
        F.col("d"),
        F.to_date(F.col("date"), "yyyy-MM-dd").alias("date"),
        F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),
        F.col("wday").cast("int").alias("wday"),
        F.col("month").cast("int").alias("month"),
        F.col("year").cast("int").alias("year"),
        *[
            F.when(F.trim(F.col(c)) == "", None).otherwise(F.col(c)).alias(c)
            for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2")
        ],
        *[F.col(c).cast("int").alias(c) for c in ("snap_CA", "snap_TX", "snap_WI")],
    )
    stats = typed.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("d").alias("distinct_d"),
        F.sum(F.when(F.col("d").isNull() | F.col("date").isNull(), 1).otherwise(0)).alias(
            "invalid"
        ),
    ).first()
    if stats["invalid"]:
        raise ProcessingError("calendar contains a null day key or an unparsable date.")
    if stats["rows"] != stats["distinct_d"]:
        raise ProcessingError("calendar.d must be unique before joining.")
    return typed


def validate_wide_sales(sales_wide: "DataFrame", first_day: int, last_day: int) -> int:
    """Validate the wide matrix and return the number of unique item-store series."""

    missing_ids = [c for c in ID_FIELDS if c not in sales_wide.columns]
    if missing_ids:
        raise ProcessingError(f"sales matrix is missing id columns: {missing_ids}")
    present = set(day_columns(sales_wide))
    missing_days = [day_key(d) for d in range(first_day, last_day + 1) if day_key(d) not in present]
    if missing_days:
        raise ProcessingError(
            f"sales matrix is missing {len(missing_days)} day columns, first: {missing_days[0]}"
        )
    stats = sales_wide.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("item_id", "store_id").alias("series"),
    ).first()
    if stats["rows"] != stats["series"]:
        raise ProcessingError("Each (item_id, store_id) series must appear exactly once.")
    return int(stats["series"])


def normalize_chunk(
    sales_wide: "DataFrame",
    calendar: "DataFrame",
    start_day: int,
    end_day: int,
) -> "DataFrame":
    """Unpivot one bounded day range into ``sales_daily`` rows.

    ``unit_sales`` is parsed strictly: a value that is not a nonnegative integer
    becomes null (never zero) so ``require_valid_units`` can reject the chunk. The
    result keeps every zero-sale observation.
    """

    keys = [day_key(d) for d in range(start_day, end_day + 1)]
    long = sales_wide.select(*ID_FIELDS, *[F.col(k).cast("string").alias(k) for k in keys]).unpivot(
        list(ID_FIELDS), keys, "day_key", "raw_unit_sales"
    )

    parsed = long.withColumn(
        "unit_sales",
        F.when(
            F.trim(F.col("raw_unit_sales")).rlike(r"^\d+$"),
            F.trim(F.col("raw_unit_sales")).cast("int"),
        ),
    )
    joined = parsed.join(
        F.broadcast(calendar.select("d", "date", "wm_yr_wk")),
        F.col("day_key") == F.col("d"),
        "left",
    )
    return joined.select(
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        F.col("id").alias("series_id"),
        "day_key",
        F.regexp_extract("day_key", r"^d_(\d+)$", 1).cast("int").alias("day_index"),
        "date",
        "wm_yr_wk",
        "unit_sales",
    )


def require_valid_units(chunk: "DataFrame") -> None:
    """Fail when a normalized frame has unparsable units or an unmapped calendar day."""

    stats = chunk.agg(
        F.sum(F.when(F.col("unit_sales").isNull(), 1).otherwise(0)).alias("bad_units"),
        F.sum(F.when(F.col("date").isNull(), 1).otherwise(0)).alias("unmapped_days"),
    ).first()
    if stats["bad_units"]:
        raise ProcessingError(f"{stats['bad_units']} rows have non-integer or negative unit_sales.")
    if stats["unmapped_days"]:
        raise ProcessingError(f"{stats['unmapped_days']} rows have no calendar mapping.")


def normalize_sales(
    sales_wide: "DataFrame",
    calendar: "DataFrame",
    *,
    first_day: int = DEFAULT_FIRST_DAY,
    last_day: int = DEFAULT_LAST_DAY,
    days_per_chunk: int = DEFAULT_DAYS_PER_CHUNK,
) -> "DataFrame":
    """Return ``sales_daily`` for ``first_day..last_day`` as a union of bounded chunks.

    The calendar must come from ``prepare_calendar``. Days outside the requested range
    (for example ``d_1942``-``d_1969``) are never treated as observed sales.
    """

    validate_wide_sales(sales_wide, first_day, last_day)
    result: DataFrame | None = None
    for start, end in day_chunks(first_day, last_day, days_per_chunk):
        chunk = normalize_chunk(sales_wide, calendar, start, end)
        result = chunk if result is None else result.unionByName(chunk)
    assert result is not None  # day_chunks never returns an empty list
    return result.select(*SALES_DAILY_COLUMNS)
