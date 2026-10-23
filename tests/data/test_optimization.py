"""Spark decision pipeline: scenario, replenishment, allocation, and policy evaluation."""

from datetime import date
from decimal import Decimal

import pytest

from tests.data import forecast_fixtures as ff
from warehouse_bigdata.contracts import (
    ALLOCATION_RECOMMENDATIONS,
    CENTRAL_SUPPLY_SCENARIO,
    INVENTORY_POLICY_EVALUATION,
    INVENTORY_SCENARIO,
    REPLENISHMENT_RECOMMENDATIONS,
)
from warehouse_bigdata.optimization.allocation import (
    POLICY_EQUAL_SHARE,
    POLICY_SHORTAGE_FIRST,
    allocate_equal_share,
    allocate_shortage_first,
)
from warehouse_bigdata.optimization.decisions import (
    DecisionError,
    allocate,
    build_replenishment,
    decision_run_id_for,
)
from warehouse_bigdata.optimization.evaluation import (
    evaluate_policies,
    summarize_policies,
)
from warehouse_bigdata.optimization.replenishment import compute_request
from warehouse_bigdata.optimization.scenario import (
    PRESETS,
    ScenarioPreset,
    build_scenario,
    scenario_id_for,
)
from warehouse_bigdata.optimization.simulation import simulate_week

ISSUE = ff.day_date(45)
SNAPSHOT = "synthetic-snapshot"


@pytest.fixture(scope="module")
def sales(spark):
    return spark.createDataFrame(ff.enriched_rows(), ff.ENRICHED_SCHEMA).cache()


def _forecast_rows(per_series: dict[tuple[str, str], list[float]], run="synthetic-fr"):
    rows = []
    for (item, store), values in per_series.items():
        for h, value in enumerate(values, start=1):
            rows.append(
                (
                    run,
                    "synthetic-model",
                    SNAPSHOT,
                    "rf_lag_v1",
                    ISSUE,
                    ff.day_date(45 + h),
                    h,
                    item,
                    store,
                    value,
                    value,
                    "validation",
                )
            )
    return rows


FORECAST_SCHEMA = (
    "forecast_run_id string, model_run_id string, source_snapshot_id string, "
    "feature_version string, issue_date date, forecast_date date, horizon int, item_id string, "
    "store_id string, raw_prediction double, predicted_units double, run_type string"
)

SERIES_KEYS = [("FOODS_1_001", "CA_1"), ("FOODS_1_001", "TX_1"), ("HOBBIES_1_002", "CA_1")]


@pytest.fixture(scope="module")
def forecast(spark):
    per_series = {
        SERIES_KEYS[0]: [4.0] * 7,
        SERIES_KEYS[1]: [1.0] * 7,
        SERIES_KEYS[2]: [0.3] * 7,
    }
    return spark.createDataFrame(_forecast_rows(per_series), FORECAST_SCHEMA)


@pytest.fixture(scope="module")
def scenario(spark, sales, forecast):
    eligible = forecast.select("item_id", "store_id").distinct()
    return build_scenario(sales, eligible, ISSUE, PRESETS["balanced"], SNAPSHOT)


def test_scenario_matches_contracts_and_exact_arithmetic(scenario, sales):
    stores, central = scenario
    assert tuple(stores.columns) == INVENTORY_SCENARIO.field_names
    assert tuple(central.columns) == CENTRAL_SUPPLY_SCENARIO.field_names

    h7 = {
        (item, store): sum(ff.units(i, d) for d in range(39, 46))
        for i, (item, store) in enumerate(SERIES_KEYS)
    }
    row = stores.filter("item_id='FOODS_1_001' AND store_id='CA_1'").first()
    assert row["opening_stock_units"] == int(Decimal(h7[SERIES_KEYS[0]]) * Decimal("0.75"))
    expected_safety = -(-int(Decimal(h7[SERIES_KEYS[0]]) * 100 * 20) // 10000)
    assert row["safety_stock_units"] == expected_safety
    assert row["origin_label"] == "SIMULATED_FROM_M5_HISTORY"
    assert row["scenario_seed"] is None

    food_total = h7[SERIES_KEYS[0]] + h7[SERIES_KEYS[1]]
    supply = central.filter("item_id='FOODS_1_001'").first()
    assert supply["central_available_units"] == int(Decimal(food_total) * Decimal("0.60"))
    # One central row per item, not per store.
    assert central.count() == 2


def test_scenario_ignores_the_future_and_is_reproducible(spark, sales, forecast):
    eligible = forecast.select("item_id", "store_id").distinct()
    altered_rows = [(*r[:10], 500, *r[11:]) if r[7] > 45 else r for r in ff.enriched_rows()]
    altered = spark.createDataFrame(altered_rows, ff.ENRICHED_SCHEMA)
    a = build_scenario(sales, eligible, ISSUE, PRESETS["balanced"], SNAPSHOT)[0]
    b = build_scenario(altered, eligible, ISSUE, PRESETS["balanced"], SNAPSHOT)[0]
    key = lambda df: sorted(map(tuple, df.drop("scenario_id").collect()))  # noqa: E731
    assert key(a) == key(b)
    assert scenario_id_for(PRESETS["balanced"], ISSUE, SNAPSHOT) == scenario_id_for(
        PRESETS["balanced"], ISSUE, SNAPSHOT
    )
    assert scenario_id_for(PRESETS["balanced"], ISSUE, SNAPSHOT) != scenario_id_for(
        PRESETS["scarce_supply"], ISSUE, SNAPSHOT
    )
    custom = ScenarioPreset("balanced", Decimal("0.76"), Decimal("0.20"), Decimal("0.60"))
    assert scenario_id_for(custom, ISSUE, SNAPSHOT) != scenario_id_for(
        PRESETS["balanced"], ISSUE, SNAPSHOT
    )


def test_series_without_full_history_are_left_out_of_the_scenario(spark, sales, forecast):
    early = date(2020, 1, 4)  # only three days of history exist
    eligible = forecast.select("item_id", "store_id").distinct()
    stores, central = build_scenario(sales, eligible, early, PRESETS["balanced"], SNAPSHOT)
    assert stores.count() == 0 and central.count() == 0


def test_replenishment_matches_pure_rules(forecast, scenario):
    stores, _ = scenario
    run_id = decision_run_id_for("synthetic-fr", "scn", ISSUE, [POLICY_SHORTAGE_FIRST])
    replenishment, rejected = build_replenishment(forecast, stores, decision_run_id=run_id)
    assert tuple(replenishment.columns) == REPLENISHMENT_RECOMMENDATIONS.field_names
    assert rejected.count() == 0

    rows = {(r["item_id"], r["store_id"]): r for r in replenishment.collect()}
    scenario_rows = {(r["item_id"], r["store_id"]): r for r in stores.collect()}
    predictions = {SERIES_KEYS[0]: [4.0] * 7, SERIES_KEYS[1]: [1.0] * 7, SERIES_KEYS[2]: [0.3] * 7}
    for key, row in rows.items():
        expected = compute_request(
            predictions[key],
            scenario_rows[key]["opening_stock_units"],
            scenario_rows[key]["safety_stock_units"],
        )
        assert row["planning_demand_units"] == expected.planning_demand_units
        assert row["request_units"] == expected.request_units
        assert row["critical_need_units"] == expected.critical_need_units
        assert row["safety_need_units"] == expected.safety_need_units
        assert row["request_units"] == row["critical_need_units"] + row["safety_need_units"]
        assert row["forecast_7d_units"] == pytest.approx(expected.forecast_7d_units)
    assert rows[SERIES_KEYS[2]]["planning_demand_units"] == 3  # ceil(2.1)


def test_incomplete_or_unscenarioed_series_are_rejected_not_zeroed(spark, scenario):
    stores, _ = scenario
    per_series = {
        SERIES_KEYS[0]: [4.0] * 6,  # only six horizons
        SERIES_KEYS[1]: [1.0] * 7,
        ("MISSING_ITEM", "CA_1"): [1.0] * 7,  # no scenario row
    }
    forecast = spark.createDataFrame(_forecast_rows(per_series), FORECAST_SCHEMA)
    replenishment, rejected = build_replenishment(forecast, stores, decision_run_id="d")
    reasons = {(r["item_id"], r["store_id"]): r["reason_code"] for r in rejected.collect()}
    assert reasons == {
        SERIES_KEYS[0]: "INVALID_FORECAST",
        ("MISSING_ITEM", "CA_1"): "MISSING_SCENARIO",
    }
    assert [(r["item_id"], r["store_id"]) for r in replenishment.collect()] == [SERIES_KEYS[1]]


def test_nan_prediction_is_invalid(spark, scenario):
    stores, _ = scenario
    values = [1.0] * 6 + [float("nan")]
    forecast = spark.createDataFrame(_forecast_rows({SERIES_KEYS[0]: values}), FORECAST_SCHEMA)
    replenishment, rejected = build_replenishment(forecast, stores, decision_run_id="d")
    assert replenishment.count() == 0
    assert rejected.first()["reason_code"] == "INVALID_FORECAST"


def test_mixed_forecast_runs_are_rejected(spark, scenario):
    stores, _ = scenario
    rows = _forecast_rows({SERIES_KEYS[0]: [1.0] * 7}, "run-a") + _forecast_rows(
        {SERIES_KEYS[1]: [1.0] * 7}, "run-b"
    )
    forecast = spark.createDataFrame(rows, FORECAST_SCHEMA)
    with pytest.raises(DecisionError, match="forecast_run_id"):
        build_replenishment(forecast, stores, decision_run_id="d")


def test_scenario_issue_date_must_match(spark, forecast, scenario):
    stores, _ = scenario
    from pyspark.sql import functions as F

    wrong = stores.withColumn("issue_date", F.lit(date(2020, 1, 1)))
    with pytest.raises(DecisionError, match="issue_date"):
        build_replenishment(forecast, wrong, decision_run_id="d")


@pytest.fixture(scope="module")
def decisions(spark, forecast, scenario):
    stores, central = scenario
    replenishment, _ = build_replenishment(forecast, stores, decision_run_id="synthetic-dec")
    replenishment = replenishment.cache()
    proposed = allocate(replenishment, central, policy_id=POLICY_SHORTAGE_FIRST).cache()
    baseline = allocate(replenishment, central, policy_id=POLICY_EQUAL_SHARE).cache()
    return replenishment, central, proposed, baseline


def test_allocation_matches_contract_and_pure_functions(decisions):
    replenishment, central, proposed, baseline = decisions
    assert tuple(proposed.columns) == ALLOCATION_RECOMMENDATIONS.field_names
    assert proposed.count() == replenishment.count() == baseline.count()

    supply = {r["item_id"]: r["central_available_units"] for r in central.collect()}
    for item in supply:
        reps = [r for r in replenishment.collect() if r["item_id"] == item]
        critical = {r["store_id"]: r["critical_need_units"] for r in reps}
        safety = {r["store_id"]: r["safety_need_units"] for r in reps}
        first, second = allocate_shortage_first(critical, safety, supply[item])
        shares = allocate_equal_share({s: critical[s] + safety[s] for s in critical}, supply[item])
        for r in proposed.filter(f"item_id='{item}'").collect():
            assert r["critical_allocated_units"] == first[r["store_id"]]
            assert r["safety_allocated_units"] == second[r["store_id"]]
            assert r["allocated_units"] == first[r["store_id"]] + second[r["store_id"]]
        for r in baseline.filter(f"item_id='{item}'").collect():
            assert r["allocated_units"] == shares[r["store_id"]]
            assert r["critical_allocated_units"] is None


def test_allocation_invariants(decisions):
    _, central, proposed, baseline = decisions
    supply = {r["item_id"]: r["central_available_units"] for r in central.collect()}
    for allocation in (proposed, baseline):
        rows = allocation.collect()
        assert all(0 <= r["allocated_units"] <= r["requested_units"] for r in rows)
        assert all(
            r["unmet_request_units"] == r["requested_units"] - r["allocated_units"] for r in rows
        )
        for item, units in supply.items():
            assert sum(r["allocated_units"] for r in rows if r["item_id"] == item) <= units


def test_policy_evaluation_matches_the_pure_simulation(decisions, sales):
    replenishment, _, proposed, baseline = decisions
    both = proposed.unionByName(baseline)
    evaluation = evaluate_policies(
        both, replenishment, sales, evaluation_run_id="synthetic-ev", run_type="validation"
    ).cache()
    assert tuple(evaluation.columns) == INVENTORY_POLICY_EVALUATION.field_names
    assert evaluation.count() == 6

    opening = {
        (r["item_id"], r["store_id"]): r["opening_stock_units"] for r in replenishment.collect()
    }
    index = {key: i for i, key in enumerate(SERIES_KEYS)}
    for r in evaluation.collect():
        key = (r["item_id"], r["store_id"])
        daily = [ff.units(index[key], d) for d in range(46, 53)]
        allocated = next(
            a["allocated_units"]
            for a in both.collect()
            if (a["item_id"], a["store_id"], a["policy_id"]) == (*key, r["policy_id"])
        )
        expected = simulate_week(opening[key], allocated, daily)
        assert r["actual_sales_proxy_7d_units"] == sum(daily)
        assert r["simulated_fulfilled_units"] == expected.fulfilled_units
        assert r["simulated_unmet_units"] == expected.unmet_units
        assert r["ending_stock_units"] == expected.ending_stock_units
        assert r["simulated_stockout_days"] == expected.stockout_days
        assert r["observed_label_window_start"] == ff.day_date(46)
        assert r["observed_label_window_end"] == ff.day_date(52)


def test_forward_demo_cannot_be_scored(decisions, sales):
    replenishment, _, proposed, _ = decisions
    with pytest.raises(ValueError, match="run_type"):
        evaluate_policies(
            proposed, replenishment, sales, evaluation_run_id="e", run_type="forward_demo"
        )


def test_series_without_all_seven_outcomes_are_excluded(spark, decisions):
    replenishment, _, proposed, _ = decisions
    truncated = spark.createDataFrame([r for r in ff.enriched_rows(days=50)], ff.ENRICHED_SCHEMA)
    evaluation = evaluate_policies(
        proposed, replenishment, truncated, evaluation_run_id="e", run_type="validation"
    )
    assert evaluation.count() == 0


def test_summary_counts_central_supply_once_per_item(decisions, sales):
    replenishment, central, proposed, baseline = decisions
    evaluation = evaluate_policies(
        proposed.unionByName(baseline),
        replenishment,
        sales,
        evaluation_run_id="e",
        run_type="validation",
    )
    summary = summarize_policies(evaluation, proposed.unionByName(baseline))
    rows = {r["policy_id"]: r for r in summary.collect()}
    total_supply = sum(r["central_available_units"] for r in central.collect())
    for row in rows.values():
        assert row["central_available_units"] == total_supply
        assert row["actual_units"] > 0
        assert row["demand_fulfillment_rate"] == pytest.approx(
            row["simulated_fulfilled_units"] / row["actual_units"]
        )
        assert (
            row["simulated_fulfilled_units"] + row["simulated_unmet_units"] == row["actual_units"]
        )
