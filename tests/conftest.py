"""Shared pytest fixtures. Spark is started once per session and only on first use."""

import pytest


@pytest.fixture(scope="session")
def spark():
    from warehouse_bigdata.common.spark import build_spark_session

    session = build_spark_session("local[2]", "warehouse-tests", shuffle_partitions=2)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
