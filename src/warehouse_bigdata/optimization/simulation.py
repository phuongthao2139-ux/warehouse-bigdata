"""Seven-day inventory simulation against observed sales as a demand proxy (docs/08 s11.3)."""

from dataclasses import dataclass

from .allocation import AllocationError, _validate_units


@dataclass(frozen=True, slots=True)
class WeekOutcome:
    fulfilled_units: int
    unmet_units: int
    ending_stock_units: int
    stockout_days: int


def simulate_week(opening_stock: int, allocated: int, daily_sales: list[int]) -> WeekOutcome:
    """Simulate seven days from ``stock = opening + allocated``.

    Unmet units are lost (no backorder) and nothing arrives mid-horizon.
    """

    if len(daily_sales) != 7:
        raise AllocationError("Exactly seven observed days are required.")
    _validate_units("opening_stock", opening_stock)
    _validate_units("allocated", allocated)
    stock = opening_stock + allocated
    fulfilled = unmet = stockout_days = 0
    for sold in daily_sales:
        _validate_units("daily_sales", sold)
        served = min(stock, sold)
        fulfilled += served
        unmet += sold - served
        stockout_days += 1 if sold > stock else 0
        stock -= served
    return WeekOutcome(fulfilled, unmet, stock, stockout_days)
