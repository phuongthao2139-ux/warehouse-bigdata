"""Deterministic simulated inventory scenarios (docs/08 section 6)."""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from warehouse_bigdata.common.identity import stable_digest, stable_id

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

SCENARIO_GENERATOR_VERSION = "historical_scaled_v1"
SCENARIO_TYPE = "historical_scaled"
ORIGIN_LABEL = "SIMULATED_FROM_M5_HISTORY"
CENTRAL_WAREHOUSE_ID = "CENTRAL_01"
LOOKBACK_DAYS = 7


@dataclass(frozen=True, slots=True)
class ScenarioPreset:
    """Scenario factors. Illustrative test fixtures, not calibrated policies."""

    name: str
    stock_cover_factor: Decimal
    safety_cover_factor: Decimal
    central_cover_factor: Decimal


PRESETS: dict[str, ScenarioPreset] = {
    p.name: p
    for p in (
        ScenarioPreset("balanced", Decimal("0.75"), Decimal("0.20"), Decimal("0.60")),
        ScenarioPreset("scarce_supply", Decimal("0.35"), Decimal("0.20"), Decimal("0.20")),
        ScenarioPreset("ample_supply", Decimal("1.00"), Decimal("0.20"), Decimal("1.50")),
    )
}


def scenario_id_for(preset: ScenarioPreset, issue_date: date, source_snapshot_id: str) -> str:
    """Return the immutable identity of the inputs that determine a scenario."""

    return stable_id(
        "scenario",
        generator=SCENARIO_GENERATOR_VERSION,
        preset=preset.name,
        factors=[
            str(preset.stock_cover_factor),
            str(preset.safety_cover_factor),
            str(preset.central_cover_factor),
        ],
        issue_date=issue_date.isoformat(),
        source_snapshot_id=source_snapshot_id,
    )


def build_scenario(
    sales: "DataFrame",
    eligible: "DataFrame",
    issue_date: date,
    preset: ScenarioPreset,
    source_snapshot_id: str,
) -> tuple["DataFrame", "DataFrame"]:
    """Return ``(inventory_scenario, central_supply_scenario)`` for ``issue_date``.

    Only sales on ``issue_date - 6 .. issue_date`` are read; nothing after the issue date
    can influence stock or supply. ``eligible`` holds the ``(item_id, store_id)`` series
    that define each product's store population. A series needs all seven history days,
    otherwise it is left out. Arithmetic is exact (decimal) so floors and ceilings do not
    depend on floating-point rounding.
    """

    start = issue_date - timedelta(days=LOOKBACK_DAYS - 1)
    history = (
        sales.filter((F.col("date") >= F.lit(start)) & (F.col("date") <= F.lit(issue_date)))
        .groupBy("item_id", "store_id")
        .agg(
            F.count(F.lit(1)).alias("days"),
            F.sum(F.col("unit_sales").cast("decimal(20,6)")).alias("h7"),
        )
        .filter(F.col("days") == LOOKBACK_DAYS)
        .join(eligible.select("item_id", "store_id").distinct(), ["item_id", "store_id"])
    )

    scenario_id = scenario_id_for(preset, issue_date, source_snapshot_id)
    digest = stable_digest(
        generator=SCENARIO_GENERATOR_VERSION,
        factors=[
            str(preset.stock_cover_factor),
            str(preset.safety_cover_factor),
            str(preset.central_cover_factor),
        ],
    )

    def factor(value: Decimal):
        return F.lit(value).cast("decimal(10,4)")

    store_rows = history.select(
        F.lit(scenario_id).alias("scenario_id"),
        F.lit(SCENARIO_GENERATOR_VERSION).alias("scenario_version"),
        F.lit(SCENARIO_TYPE).alias("scenario_type"),
        F.lit(source_snapshot_id).alias("source_snapshot_id"),
        F.lit(issue_date).alias("issue_date"),
        "item_id",
        "store_id",
        F.floor(F.col("h7") * factor(preset.stock_cover_factor))
        .cast("int")
        .alias("opening_stock_units"),
        F.ceil(F.col("h7") * factor(preset.safety_cover_factor))
        .cast("int")
        .alias("safety_stock_units"),
        F.lit(digest).alias("parameter_digest"),
        F.lit(ORIGIN_LABEL).alias("origin_label"),
        F.lit(None).cast("int").alias("scenario_seed"),
        "h7",
    )
    central = (
        store_rows.groupBy("item_id")
        .agg(F.sum("h7").alias("item_h7"))
        .select(
            F.lit(scenario_id).alias("scenario_id"),
            F.lit(issue_date).alias("issue_date"),
            "item_id",
            F.lit(CENTRAL_WAREHOUSE_ID).alias("source_warehouse_id"),
            F.floor(F.col("item_h7") * factor(preset.central_cover_factor))
            .cast("int")
            .alias("central_available_units"),
            F.lit(ORIGIN_LABEL).alias("origin_label"),
        )
    )
    return store_rows.drop("h7"), central
