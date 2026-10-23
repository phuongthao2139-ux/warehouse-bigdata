"""Spark session construction shared by batch jobs."""

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def build_spark_session(
    master_url: str,
    app_name: str,
    *,
    shuffle_partitions: int | None = None,
) -> "SparkSession":
    """Create or reuse a UTC-pinned Spark session for ``master_url``.

    The session is configured for batch DataFrame work only. For a local master the
    driver's Python interpreter is also used by Spark workers so a project virtual
    environment is respected.
    """

    from pyspark.sql import SparkSession

    if master_url.startswith("local"):
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = (
        SparkSession.builder.master(master_url)
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    return builder.getOrCreate()
