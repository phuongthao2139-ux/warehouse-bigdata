"""Deterministic integer allocation of one SKU's central supply (docs/08 sections 8-10).

All functions are pure and validate their inputs explicitly (never with ``assert``).
Ties are always broken by ascending ``store_id`` so results do not depend on input order.
"""

from collections.abc import Mapping

POLICY_SHORTAGE_FIRST = "shortage_first_proportional"
POLICY_EQUAL_SHARE = "equal_share"
POLICY_VERSION = "v1"

MAX_UNITS = 2**31 - 1


class AllocationError(ValueError):
    """Raised for invalid allocation inputs or a violated allocation invariant."""


def _validate_units(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AllocationError(f"{name} must be an integer, got {value!r}.")
    if value < 0 or value > MAX_UNITS:
        raise AllocationError(f"{name} must be within 0..{MAX_UNITS}, got {value}.")


def _validate_needs(needs: Mapping[str, int], supply: int) -> None:
    _validate_units("supply", supply)
    for store, units in needs.items():
        _validate_units(f"need[{store}]", units)


def apportion_proportionally(needs: Mapping[str, int], supply: int) -> dict[str, int]:
    """Split ``min(supply, sum(needs))`` proportionally to ``needs`` (largest remainder).

    Quotas and remainders use integer arithmetic only. The result sums exactly to the
    capacity and never exceeds any store's need.
    """

    _validate_needs(needs, supply)
    stores = sorted(needs)
    total_need = sum(needs.values())
    capacity = min(supply, total_need)
    allocated = {store: 0 for store in stores}
    if capacity == 0:
        return allocated

    for store in stores:
        allocated[store] = (capacity * needs[store]) // total_need
    remaining = capacity - sum(allocated.values())
    ranked = sorted(stores, key=lambda s: (-((capacity * needs[s]) % total_need), s))
    for store in ranked[:remaining]:
        allocated[store] += 1

    if sum(allocated.values()) != capacity or any(allocated[s] > needs[s] for s in stores):
        raise AllocationError("Proportional apportionment violated its invariants.")
    return allocated


def allocate_shortage_first(
    critical: Mapping[str, int], safety: Mapping[str, int], central_available: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Policy ``shortage_first_proportional_v1``.

    Stage 1 covers critical needs proportionally; stage 2 spends what remains on safety
    needs proportionally. Returns ``(critical_allocated, safety_allocated)`` per store.
    """

    if set(critical) != set(safety):
        raise AllocationError("critical and safety needs must cover the same stores.")
    _validate_units("central_available", central_available)
    first = apportion_proportionally(critical, central_available)
    second = apportion_proportionally(safety, central_available - sum(first.values()))
    if sum(first.values()) + sum(second.values()) > central_available:
        raise AllocationError("Allocated more than the central supply.")
    return first, second


def allocate_equal_share(requests: Mapping[str, int], central_available: int) -> dict[str, int]:
    """Baseline ``equal_share_v1``: even shares capped by each request, leftovers recycled."""

    _validate_needs(requests, central_available)
    stores = sorted(requests)
    result = {store: 0 for store in stores}
    remaining = min(central_available, sum(requests.values()))

    while remaining > 0:
        active = [s for s in stores if result[s] < requests[s]]
        if not active:
            raise AllocationError("Equal share ran out of active stores with supply remaining.")
        bulk = remaining // len(active)
        if bulk > 0:
            for store in active:
                add = min(bulk, requests[store] - result[store])
                result[store] += add
                remaining -= add
        else:
            for store in active[:remaining]:
                result[store] += 1
            remaining = 0

    if sum(result.values()) > central_available or any(result[s] > requests[s] for s in stores):
        raise AllocationError("Equal share violated its invariants.")
    return result
