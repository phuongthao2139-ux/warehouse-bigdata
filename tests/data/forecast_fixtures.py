"""Synthetic 60-day, 3-series history for forecasting tests (not real M5 data)."""

from datetime import date, timedelta

from tests.data import spark_fixtures as fx
from warehouse_bigdata.forecasting.protocol import Protocol

DAYS = 60
SMALL_PROTOCOL = Protocol(
    train_last_day=45,
    validation_last_day=52,
    test_last_day=59,
    validation_origins=(45,),
    test_origins=(52,),
)


def day_date(day_index: int) -> date:
    return fx.FIRST_DATE + timedelta(days=day_index - 1)


def units(series_index: int, day_index: int) -> int:
    """Weekly pattern per series plus a mild trend, nonnegative integers."""

    base = (3, 6, 1)[series_index]
    weekday_bump = (day_index % 7) * (series_index + 1)
    return base + weekday_bump + day_index // 20


def enriched_rows(days: int = DAYS) -> list[tuple]:
    rows = []
    for index, (item, dept, cat, store, state) in enumerate(fx.SERIES):
        for day in range(1, days + 1):
            price = None if (index == 2 and day < 20) else 2.0 + index
            rows.append(
                (
                    item,
                    dept,
                    cat,
                    store,
                    state,
                    f"{item}_{store}_evaluation",
                    f"d_{day}",
                    day,
                    day_date(day),
                    11000 + day // 7,
                    units(index, day),
                    price,
                    price is None,
                    day_date(day).isoweekday() % 7 + 1,
                    day_date(day).month,
                    day_date(day).year,
                    False,
                    units(index, day) == 0,
                    "TestEvent" if day % 15 == 0 else None,
                    None,
                    None,
                    None,
                )
            )
    return rows


ENRICHED_SCHEMA = (
    "item_id string, dept_id string, cat_id string, store_id string, state_id string, "
    "series_id string, day_key string, day_index int, date date, wm_yr_wk int, unit_sales int, "
    "sell_price double, price_missing boolean, wday int, month int, year int, "
    "snap_applicable boolean, is_zero_sale boolean, event_name_1 string, event_type_1 string, "
    "event_name_2 string, event_type_2 string"
)

CALENDAR_SCHEMA = (
    "d string, date date, wm_yr_wk int, wday int, month int, year int, "
    "event_name_1 string, event_type_1 string, event_name_2 string, event_type_2 string, "
    "snap_CA int, snap_TX int, snap_WI int"
)


def calendar_rows(days: int = DAYS + 14) -> list[tuple]:
    return [
        (
            f"d_{d}",
            day_date(d),
            11000 + d // 7,
            day_date(d).isoweekday() % 7 + 1,
            day_date(d).month,
            day_date(d).year,
            "TestEvent" if d % 15 == 0 else None,
            None,
            None,
            None,
            0,
            0,
            0,
        )
        for d in range(1, days + 1)
    ]
