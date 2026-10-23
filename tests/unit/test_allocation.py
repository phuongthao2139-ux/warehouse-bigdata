"""Pure allocation, replenishment, and simulation rules (docs/08 sections 7-13)."""

import itertools
import random

import pytest

from warehouse_bigdata.optimization.allocation import (
    AllocationError,
    allocate_equal_share,
    allocate_shortage_first,
    apportion_proportionally,
)
from warehouse_bigdata.optimization.replenishment import compute_request
from warehouse_bigdata.optimization.simulation import simulate_week


def test_request_example_a_from_the_design():
    request = compute_request([80.0 / 7] * 7, opening_stock=50, safety_stock=10)
    assert request.planning_demand_units == 80
    assert request.target_stock_units == 90
    assert request.request_units == 40
    assert request.critical_need_units == 30
    assert request.safety_need_units == 10
    assert request.forecast_shortage_flag is True


def test_request_is_zero_when_stock_covers_target():
    request = compute_request([80.0 / 7] * 7, opening_stock=100, safety_stock=10)
    assert request.request_units == 0
    assert request.critical_need_units == 0
    assert request.safety_need_units == 0
    assert request.forecast_shortage_flag is False


def test_ceiling_is_applied_once_to_the_unrounded_sum():
    # Each day 0.3 -> sum 2.1 -> D=3, not 7 * ceil(0.3) = 7.
    assert compute_request([0.3] * 7, 0, 0).planning_demand_units == 3
    assert compute_request([0.0] * 7, 5, 0).request_units == 0


def test_request_rejects_bad_inputs():
    with pytest.raises(AllocationError):
        compute_request([1.0] * 6, 0, 0)
    with pytest.raises(AllocationError):
        compute_request([1.0] * 6 + [float("nan")], 0, 0)
    with pytest.raises(AllocationError):
        compute_request([1.0] * 6 + [-0.1], 0, 0)
    with pytest.raises(AllocationError):
        compute_request([1.0] * 7, -1, 0)
    with pytest.raises(AllocationError):
        compute_request([1.0] * 7, 1.5, 0)  # type: ignore[arg-type]


def test_proportional_apportionment_example():
    result = apportion_proportionally({"A": 10, "B": 20, "C": 30}, 30)
    assert result == {"A": 5, "B": 10, "C": 15}
    assert sum(result.values()) == 30


def test_largest_remainder_and_store_id_tie_break():
    # Equal needs, supply 1: remainders tie, ascending store_id wins.
    assert apportion_proportionally({"B": 5, "A": 5}, 1) == {"A": 1, "B": 0}
    result = apportion_proportionally({"A": 1, "B": 1, "C": 1}, 2)
    assert result == {"A": 1, "B": 1, "C": 0}


def test_apportion_edge_cases():
    assert apportion_proportionally({"A": 4, "B": 6}, 0) == {"A": 0, "B": 0}
    assert apportion_proportionally({"A": 4, "B": 6}, 100) == {"A": 4, "B": 6}
    assert apportion_proportionally({"A": 0, "B": 0}, 5) == {"A": 0, "B": 0}
    assert apportion_proportionally({}, 5) == {}


def test_apportion_is_independent_of_input_order():
    needs = {"S1": 7, "S2": 3, "S3": 11, "S4": 5}
    base = apportion_proportionally(needs, 13)
    for order in itertools.permutations(needs):
        assert apportion_proportionally({k: needs[k] for k in order}, 13) == base


def test_apportion_invariants_hold_randomly():
    rng = random.Random(0)
    for _ in range(300):
        needs = {f"S{i}": rng.randint(0, 20) for i in range(rng.randint(1, 8))}
        supply = rng.randint(0, 120)
        result = apportion_proportionally(needs, supply)
        assert sum(result.values()) == min(supply, sum(needs.values()))
        assert all(0 <= result[s] <= needs[s] for s in needs)


def test_shortage_first_covers_critical_before_safety():
    critical = {"A": 10, "B": 30}
    safety = {"A": 20, "B": 0}
    first, second = allocate_shortage_first(critical, safety, 20)
    assert first == {"A": 5, "B": 15}
    assert second == {"A": 0, "B": 0}

    first, second = allocate_shortage_first(critical, safety, 50)
    assert first == {"A": 10, "B": 30}
    assert second == {"A": 10, "B": 0}


def test_shortage_first_ample_and_empty_supply():
    first, second = allocate_shortage_first({"A": 3, "B": 2}, {"A": 1, "B": 1}, 100)
    assert (first, second) == ({"A": 3, "B": 2}, {"A": 1, "B": 1})
    first, second = allocate_shortage_first({"A": 3}, {"A": 1}, 0)
    assert (first, second) == ({"A": 0}, {"A": 0})


def test_safety_only_needs_are_allocated_directly():
    first, second = allocate_shortage_first({"A": 0, "B": 0}, {"A": 4, "B": 4}, 6)
    assert first == {"A": 0, "B": 0}
    assert second == {"A": 3, "B": 3}


def test_equal_share_caps_and_redistributes():
    assert allocate_equal_share({"A": 2, "B": 10, "C": 10}, 12) == {"A": 2, "B": 5, "C": 5}
    assert allocate_equal_share({"A": 1, "B": 1, "C": 1}, 2) == {"A": 1, "B": 1, "C": 0}
    assert allocate_equal_share({"A": 3, "B": 3}, 100) == {"A": 3, "B": 3}
    assert allocate_equal_share({"A": 0, "B": 4}, 3) == {"A": 0, "B": 3}


def test_equal_share_invariants_hold_randomly():
    rng = random.Random(1)
    for _ in range(300):
        requests = {f"S{i}": rng.randint(0, 20) for i in range(rng.randint(1, 8))}
        supply = rng.randint(0, 120)
        result = allocate_equal_share(requests, supply)
        assert sum(result.values()) == min(supply, sum(requests.values()))
        assert all(0 <= result[s] <= requests[s] for s in requests)


@pytest.mark.parametrize("bad", [-1, 1.5, True, 2**31])
def test_invalid_supply_is_rejected(bad):
    with pytest.raises(AllocationError):
        apportion_proportionally({"A": 1}, bad)  # type: ignore[arg-type]
    with pytest.raises(AllocationError):
        allocate_equal_share({"A": 1}, bad)  # type: ignore[arg-type]


def test_invalid_need_and_mismatched_stores_are_rejected():
    with pytest.raises(AllocationError):
        apportion_proportionally({"A": -1}, 3)
    with pytest.raises(AllocationError):
        allocate_shortage_first({"A": 1}, {"B": 1}, 3)


def test_simulation_matches_the_design_recurrence():
    outcome = simulate_week(opening_stock=5, allocated=3, daily_sales=[2, 3, 4, 0, 1, 0, 2])
    # stock 8 -> sells 2 (6) -> 3 (3) -> 3 of 4 (0, unmet 1) -> 0 -> 0 of 1 (unmet 1) -> 0 -> 0 of 2 (unmet 2)
    assert outcome.fulfilled_units == 8
    assert outcome.unmet_units == 4
    assert outcome.ending_stock_units == 0
    assert outcome.stockout_days == 3


def test_simulation_with_surplus_keeps_ending_stock():
    outcome = simulate_week(10, 5, [1] * 7)
    assert (outcome.fulfilled_units, outcome.unmet_units) == (7, 0)
    assert outcome.ending_stock_units == 8
    assert outcome.stockout_days == 0
    with pytest.raises(AllocationError):
        simulate_week(1, 0, [1] * 6)
