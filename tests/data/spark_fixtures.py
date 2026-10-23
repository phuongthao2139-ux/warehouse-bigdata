"""Tiny deterministic M5-shaped inputs for Spark tests (synthetic, not real M5 data)."""

from datetime import date, timedelta

DAYS = 12
FIRST_DATE = date(2020, 1, 1)  # a Wednesday


def calendar_rows(days: int = DAYS + 7) -> list[tuple]:
    rows = []
    for i in range(days):
        current = FIRST_DATE + timedelta(days=i)
        event = "TestEvent" if i == 3 else None
        rows.append(
            (
                f"d_{i + 1}",
                current.isoformat(),
                str(11000 + i // 7),
                str(current.isoweekday() % 7 + 1),
                str(current.month),
                str(current.year),
                event,
                "Cultural" if event else None,
                None,
                None,
                "1" if i % 2 == 0 else "0",
                "0",
                "1",
            )
        )
    return rows


CALENDAR_SCHEMA = (
    "d string, date string, wm_yr_wk string, wday string, month string, year string, "
    "event_name_1 string, event_type_1 string, event_name_2 string, event_type_2 string, "
    "snap_CA string, snap_TX string, snap_WI string"
)

SERIES = [
    ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
    ("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
    ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
]


def sales_values(series_index: int, days: int = DAYS) -> list[int]:
    return [(series_index * 3 + d) % 5 for d in range(days)]


def wide_sales_rows(days: int = DAYS) -> tuple[list[tuple], str]:
    rows = []
    for index, (item, dept, cat, store, state) in enumerate(SERIES):
        series_id = f"{item}_{store}_evaluation"
        rows.append(
            (series_id, item, dept, cat, store, state, *map(str, sales_values(index, days)))
        )
    day_cols = ", ".join(f"d_{d + 1} string" for d in range(days))
    schema = (
        "id string, item_id string, dept_id string, cat_id string, store_id string, "
        f"state_id string, {day_cols}"
    )
    return rows, schema


def price_rows() -> list[tuple]:
    # FOODS_1_001/CA_1 priced in both weeks; FOODS_1_001/TX_1 only week 11000.
    return [
        ("CA_1", "FOODS_1_001", "11000", "2.50"),
        ("CA_1", "FOODS_1_001", "11001", "2.75"),
        ("TX_1", "FOODS_1_001", "11000", "3.00"),
    ]


PRICE_SCHEMA = "store_id string, item_id string, wm_yr_wk string, sell_price string"
