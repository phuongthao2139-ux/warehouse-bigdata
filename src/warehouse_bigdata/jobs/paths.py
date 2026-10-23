"""Artifact locations following the HDFS conventions in ``docs/CONTRACTS.md``."""


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *parts])


def curated_path(root: str, dataset: str, source_snapshot_id: str) -> str:
    return _join(root, "curated", dataset, f"source_snapshot_id={source_snapshot_id}")


def features_path(root: str, source_snapshot_id: str, feature_version: str) -> str:
    return _join(
        root,
        "features",
        "demand_forecasting",
        f"source_snapshot_id={source_snapshot_id}",
        f"feature_version={feature_version}",
    )


def model_path(root: str, model_run_id: str) -> str:
    return _join(root, "models", model_run_id)


def forecast_path(root: str, forecast_run_id: str) -> str:
    return _join(root, "outputs", "forecasts", forecast_run_id)


def decision_path(root: str, decision_run_id: str) -> str:
    return _join(root, "outputs", "decisions", decision_run_id)


def evaluation_path(root: str, evaluation_run_id: str) -> str:
    return _join(root, "outputs", "evaluation", evaluation_run_id)


def scenario_path(root: str, scenario_id: str) -> str:
    return _join(root, "outputs", "scenarios", scenario_id)


def staging_path(root: str, pipeline_run_id: str) -> str:
    return _join(root, "staging", f"pipeline_run_id={pipeline_run_id}")
