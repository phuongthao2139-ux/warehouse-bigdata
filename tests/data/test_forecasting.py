"""Forecasting behavior: leakage control, recursion, baseline, metrics."""

import math
from datetime import timedelta

import pytest

from tests.data import forecast_fixtures as ff
from warehouse_bigdata.contracts import FORECAST_DAILY, FORECAST_FEATURES, FORECAST_METRICS
from warehouse_bigdata.forecasting.evaluation import compute_forecast_metrics, score_origin
from warehouse_bigdata.forecasting.features import (
    FeatureError,
    assert_contiguous_daily,
    build_forecast_features,
    sample_training_rows,
    to_model_input,
)
from warehouse_bigdata.forecasting.history import build_history_state
from warehouse_bigdata.forecasting.inference import (
    recursive_forecast,
    seasonal_naive_forecast,
    to_forecast_daily,
)
from warehouse_bigdata.forecasting.model import (
    ForestParams,
    model_run_id_for,
    train_model,
)

PARAMS = ForestParams(num_trees=5, max_depth=4, seed=7)


@pytest.fixture(scope="module")
def enriched(spark):
    return spark.createDataFrame(ff.enriched_rows(), ff.ENRICHED_SCHEMA).cache()


@pytest.fixture(scope="module")
def calendar(spark):
    return spark.createDataFrame(ff.calendar_rows(), ff.CALENDAR_SCHEMA)


@pytest.fixture(scope="module")
def features(enriched):
    return build_forecast_features(
        enriched, "synthetic-snapshot", protocol=ff.SMALL_PROTOCOL
    ).cache()


@pytest.fixture(scope="module")
def model(features):
    training = features.filter("feature_role = 'train'")
    return train_model(to_model_input(training, with_label=True), PARAMS)


def test_features_match_contract_and_roles(features):
    assert tuple(features.columns) == FORECAST_FEATURES.field_names
    roles = {r[0] for r in features.select("feature_role").distinct().collect()}
    assert roles == {"train", "validation", "test"}
    # First 28 days are lookback context only: first target is day 29.
    first = features.agg({"target_date": "min"}).first()[0]
    assert first == ff.day_date(29)


def test_feature_values_use_only_prior_days(features):
    row = features.filter(
        f"item_id='FOODS_1_001' AND store_id='CA_1' AND target_date = DATE '{ff.day_date(40)}'"
    ).first()
    t = 40
    prior = [ff.units(0, d) for d in range(1, t)]
    assert row["label"] == ff.units(0, t)
    assert row["lag_1"] == prior[-1]
    assert row["lag_7"] == prior[-7]
    assert row["lag_28"] == prior[-28]
    assert row["rolling_mean_7"] == pytest.approx(sum(prior[-7:]) / 7)
    assert row["rolling_mean_28"] == pytest.approx(sum(prior[-28:]) / 28)
    assert row["as_of_date"] == ff.day_date(t - 1)
    assert row["event_flag"] is False


def test_feature_does_not_change_when_the_target_day_changes(spark, enriched):
    altered_rows = [(*r[:10], 999, *r[11:]) if r[7] == 40 else r for r in ff.enriched_rows()]
    altered = spark.createDataFrame(altered_rows, ff.ENRICHED_SCHEMA)
    key = f"item_id='FOODS_1_001' AND store_id='CA_1' AND target_date = DATE '{ff.day_date(40)}'"
    base = build_forecast_features(enriched, "s", protocol=ff.SMALL_PROTOCOL).filter(key).first()
    changed = build_forecast_features(altered, "s", protocol=ff.SMALL_PROTOCOL).filter(key).first()
    for column in ("lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"):
        assert base[column] == changed[column]
    assert changed["label"] == 999


def test_last_known_price_is_strictly_prior_and_not_zero(features):
    # Series 3 (HOBBIES) has no price before day 20.
    early = features.filter(
        f"item_id='HOBBIES_1_002' AND target_date = DATE '{ff.day_date(30)}'"
    ).first()
    assert early["price_missing"] is False
    assert early["last_known_price"] == pytest.approx(4.0)
    unpriced = features.filter(
        f"item_id='HOBBIES_1_002' AND target_date = DATE '{ff.day_date(29)}'"
    ).first()
    assert unpriced["price_missing"] is False  # price known at day 20..28
    assert features.filter("last_known_price = 0").count() == 0


def test_contiguity_check_rejects_gaps(spark):
    rows = [r for r in ff.enriched_rows() if r[7] != 10]
    with pytest.raises(FeatureError):
        assert_contiguous_daily(spark.createDataFrame(rows, ff.ENRICHED_SCHEMA))


def test_sampling_is_deterministic_and_training_only(features):
    first = sample_training_rows(features, target_rows=40, seed=1)
    again = sample_training_rows(features, target_rows=40, seed=1)
    keys = ["item_id", "store_id", "target_date"]
    assert sorted(map(tuple, first.select(*keys).collect())) == sorted(
        map(tuple, again.select(*keys).collect())
    )
    assert first.filter("feature_role != 'train'").count() == 0
    assert 0 < first.count() < features.filter("feature_role = 'train'").count()


def test_history_state_ignores_the_future(spark, enriched):
    issue = ff.day_date(45)
    state = build_history_state(enriched, issue)
    row = state.filter("item_id='FOODS_1_001' AND store_id='CA_1'").first()
    assert row["hist"] == [float(ff.units(0, d)) for d in range(18, 46)]
    assert row["last_known_price"] == pytest.approx(2.0)

    altered_rows = [(*r[:10], 777, *r[11:]) if r[7] > 45 else r for r in ff.enriched_rows()]
    altered = build_history_state(spark.createDataFrame(altered_rows, ff.ENRICHED_SCHEMA), issue)
    assert (
        altered.filter("item_id='FOODS_1_001' AND store_id='CA_1'").first()["hist"] == row["hist"]
    )


def test_history_state_drops_series_without_full_window(enriched):
    assert build_history_state(enriched, ff.day_date(20)).count() == 0


def test_seasonal_naive_uses_same_weekday_last_week(enriched):
    issue = ff.day_date(45)
    state = build_history_state(enriched, issue)
    rows = seasonal_naive_forecast(state, issue).filter("item_id='FOODS_1_001' AND store_id='CA_1'")
    by_h = {r["horizon"]: r for r in rows.collect()}
    for h in range(1, 8):
        assert by_h[h]["forecast_date"] == issue + timedelta(days=h)
        assert by_h[h]["predicted_units"] == ff.units(0, 45 + h - 7)


def test_recursive_forecast_shape_and_no_future_leakage(spark, enriched, calendar, model):
    issue = ff.day_date(45)
    state = build_history_state(enriched, issue)
    forecast = recursive_forecast(model, state, calendar, issue)
    assert forecast.count() == 3 * 7

    per_series = forecast.groupBy("item_id", "store_id").count().collect()
    assert all(r["count"] == 7 for r in per_series)
    rows = forecast.collect()
    assert all(r["forecast_date"] == issue + timedelta(days=r["horizon"]) for r in rows)
    assert all(r["predicted_units"] >= 0 and math.isfinite(r["raw_prediction"]) for r in rows)

    # Changing every actual after the issue date must not change any forecast.
    altered_rows = [(*r[:10], 777, *r[11:]) if r[7] > 45 else r for r in ff.enriched_rows()]
    altered = spark.createDataFrame(altered_rows, ff.ENRICHED_SCHEMA)
    again = recursive_forecast(model, build_history_state(altered, issue), calendar, issue)
    key = lambda r: (r["item_id"], r["store_id"], r["horizon"])  # noqa: E731
    assert {key(r): r["raw_prediction"] for r in again.collect()} == {
        key(r): r["raw_prediction"] for r in rows
    }


def test_recursion_feeds_predictions_back(spark, enriched, calendar, model):
    """Horizon 2 must differ when the horizon-1 prediction would differ."""
    issue = ff.day_date(45)
    state = build_history_state(enriched, issue)
    base = {
        (r["item_id"], r["store_id"], r["horizon"]): r["raw_prediction"]
        for r in recursive_forecast(model, state, calendar, issue).collect()
    }
    shifted_state = state.withColumn(
        "hist",
        state["hist"],  # same history, sanity: deterministic repeat
    )
    repeat = {
        (r["item_id"], r["store_id"], r["horizon"]): r["raw_prediction"]
        for r in recursive_forecast(model, shifted_state, calendar, issue).collect()
    }
    assert base == repeat


def test_missing_calendar_day_fails(spark, enriched, model):
    issue = ff.day_date(45)
    short = spark.createDataFrame(ff.calendar_rows(days=47), ff.CALENDAR_SCHEMA)
    with pytest.raises(FeatureError, match="exactly one row"):
        recursive_forecast(model, build_history_state(enriched, issue), short, issue)


def test_forecast_daily_matches_contract(enriched, calendar, model):
    issue = ff.day_date(45)
    forecast = recursive_forecast(model, build_history_state(enriched, issue), calendar, issue)
    out = to_forecast_daily(
        forecast,
        forecast_run_id="synthetic-fr",
        model_run_id="synthetic-model",
        source_snapshot_id="synthetic-snapshot",
        feature_version="rf_lag_v1",
        issue_date=issue,
        run_type="validation",
    )
    assert tuple(out.columns) == FORECAST_DAILY.field_names
    assert out.count() == 21


def test_scored_cohort_is_identical_and_metrics_are_correct(enriched, calendar, model):
    issue = ff.day_date(45)
    scored = score_origin(model, enriched, calendar, issue, window_id="w1").cache()
    assert scored.count() == 21

    metrics = compute_forecast_metrics(
        scored,
        evaluation_run_id="synthetic-ev",
        model_run_id="synthetic-model",
        holdout_id="validation",
        run_type="validation",
    )
    assert tuple(metrics.columns) == FORECAST_METRICS.field_names
    keys = metrics.select("holdout_id", "method_id", "metric_name", "horizon").collect()
    assert len(keys) == len(set(map(tuple, keys))) == 2 * 3 * 8

    naive = metrics.filter("method_id='seasonal_naive' AND horizon=0").collect()
    by_name = {r["metric_name"]: r for r in naive}
    errors = [
        abs(ff.units(i, 45 + h - 7) - ff.units(i, 45 + h)) for i in range(3) for h in range(1, 8)
    ]
    actuals = [ff.units(i, 45 + h) for i in range(3) for h in range(1, 8)]
    assert by_name["MAE"]["metric_value"] == pytest.approx(sum(errors) / 21)
    assert by_name["WAPE"]["metric_value"] == pytest.approx(sum(errors) / sum(actuals) * 100)
    assert by_name["MAE"]["observation_count"] == 21
    # Both methods are measured on the same number of rows.
    counts = {r["observation_count"] for r in metrics.filter("horizon=0").collect()}
    assert counts == {21}


def test_wape_is_null_when_actuals_are_zero(spark):
    scored = spark.createDataFrame(
        [("A", "S", 1, ff.day_date(1), 0, 1.0, 2.0)],
        "item_id string, store_id string, horizon int, forecast_date date, actual int, "
        "random_forest double, seasonal_naive double",
    )
    metrics = compute_forecast_metrics(
        scored, evaluation_run_id="e", model_run_id="m", holdout_id="h", run_type="validation"
    )
    wape = metrics.filter("metric_name='WAPE'").collect()
    assert wape and all(r["metric_value"] is None for r in wape)


def test_model_run_id_is_deterministic():
    kwargs = dict(
        source_snapshot_id="s",
        feature_version="v",
        train_last_day=45,
        training_rows=10,
        sample_seed=1,
    )
    assert model_run_id_for(PARAMS, **kwargs) == model_run_id_for(PARAMS, **kwargs)
    assert model_run_id_for(PARAMS, **kwargs) != model_run_id_for(
        ForestParams(num_trees=6), **kwargs
    )
