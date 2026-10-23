"""Store replenishment requests (docs/08 section 7)."""

import math
from dataclasses import dataclass

from .allocation import AllocationError, _validate_units


@dataclass(frozen=True, slots=True)
class Request:
    """One store's request split into critical and safety parts."""

    forecast_7d_units: float
    planning_demand_units: int
    opening_stock_units: int
    safety_stock_units: int
    target_stock_units: int
    critical_need_units: int
    safety_need_units: int
    request_units: int
    forecast_shortage_flag: bool


def compute_request(
    daily_predictions: list[float], opening_stock: int, safety_stock: int
) -> Request:
    """Aggregate seven unrounded predictions, take one ceiling, and derive the request.

    ``D = ceil(sum f)``, ``R = max(0, D + S - I)``, ``critical = max(0, D - I)`` and
    ``safety = R - critical``.
    """

    if len(daily_predictions) != 7:
        raise AllocationError(
            f"Exactly seven predictions are required, got {len(daily_predictions)}."
        )
    if any(not math.isfinite(p) or p < 0 for p in daily_predictions):
        raise AllocationError("Predictions must be finite and nonnegative.")
    _validate_units("opening_stock", opening_stock)
    _validate_units("safety_stock", safety_stock)

    forecast = math.fsum(daily_predictions)
    demand = math.ceil(forecast)
    request = max(0, demand + safety_stock - opening_stock)
    critical = max(0, demand - opening_stock)
    return Request(
        forecast_7d_units=forecast,
        planning_demand_units=demand,
        opening_stock_units=opening_stock,
        safety_stock_units=safety_stock,
        target_stock_units=demand + safety_stock,
        critical_need_units=critical,
        safety_need_units=request - critical,
        request_units=request,
        forecast_shortage_flag=opening_stock < demand,
    )
