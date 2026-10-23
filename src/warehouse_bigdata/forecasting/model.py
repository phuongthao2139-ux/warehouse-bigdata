"""Spark MLlib Random Forest pipeline (docs/07 section 12)."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

from warehouse_bigdata.common.identity import stable_id

from .features import MODEL_NUMERIC_FEATURES

if TYPE_CHECKING:
    from pyspark.ml import PipelineModel
    from pyspark.sql import DataFrame


@dataclass(frozen=True, slots=True)
class ForestParams:
    """Pilot hyperparameters (docs/07 section 12.2); not a tuned configuration."""

    num_trees: int = 20
    max_depth: int = 6
    max_bins: int = 32
    min_instances_per_node: int = 5
    subsampling_rate: float = 0.8
    max_memory_mb: int = 128
    seed: int = 42


def build_training_pipeline(params: ForestParams | None = None) -> Pipeline:
    """Return the unfitted pipeline. Fit it on approved training rows only."""

    params = params or ForestParams()

    store_indexer = StringIndexer(
        inputCol="store_id", outputCol="store_index", handleInvalid="keep"
    )
    dept_indexer = StringIndexer(inputCol="dept_id", outputCol="dept_index", handleInvalid="keep")
    encoder = OneHotEncoder(
        inputCols=["store_index", "dept_index"],
        outputCols=["store_vector", "dept_vector"],
        handleInvalid="keep",
    )
    price_imputer = Imputer(
        inputCols=["last_known_price"], outputCols=["price_imputed"], strategy="median"
    )
    assembler = VectorAssembler(
        inputCols=[*MODEL_NUMERIC_FEATURES, "price_imputed", "store_vector", "dept_vector"],
        outputCol="features",
        handleInvalid="error",
    )
    forest = RandomForestRegressor(
        labelCol="label",
        featuresCol="features",
        predictionCol="raw_prediction",
        numTrees=params.num_trees,
        maxDepth=params.max_depth,
        maxBins=params.max_bins,
        minInstancesPerNode=params.min_instances_per_node,
        subsamplingRate=params.subsampling_rate,
        maxMemoryInMB=params.max_memory_mb,
        seed=params.seed,
    )
    return Pipeline(stages=[store_indexer, dept_indexer, encoder, price_imputer, assembler, forest])


def model_run_id_for(
    params: ForestParams,
    *,
    source_snapshot_id: str,
    feature_version: str,
    train_last_day: int,
    training_rows: int,
    sample_seed: int,
) -> str:
    """Return the identity of one trained artifact from its defining inputs."""

    return stable_id(
        "model",
        params=asdict(params),
        source_snapshot_id=source_snapshot_id,
        feature_version=feature_version,
        train_last_day=train_last_day,
        training_rows=training_rows,
        sample_seed=sample_seed,
    )


def train_model(training: "DataFrame", params: ForestParams | None = None) -> "PipelineModel":
    """Fit the pipeline on ``training`` (model-input columns plus ``label``)."""

    return build_training_pipeline(params or ForestParams()).fit(training)
