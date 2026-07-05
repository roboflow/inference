from __future__ import annotations

from typing import Any, Mapping

from development.profiling.data.base import DataSource
from development.profiling.data.dummy import DummyDataSource
from development.profiling.data.images import ImageDirectoryDataSource


DATA_SOURCES = {
    "dummy": DummyDataSource,
    "images": ImageDirectoryDataSource,
}


def build_data_source(name: str, config: Mapping[str, Any] | None = None) -> DataSource:
    """Build a profiling data source by name.

    Args:
        name (str): Registered data source name.
        config (Mapping[str, Any] | None): Optional source-specific config.

    Returns:
        Configured data source.

    Raises:
        ValueError: If the data source name is unknown.
    """
    try:
        data_source_cls = DATA_SOURCES[name]
    except KeyError as error:
        available = ", ".join(sorted(DATA_SOURCES))
        raise ValueError(
            f"Unknown data source '{name}'. Available data sources: {available}"
        ) from error

    data_source = data_source_cls.from_config(config)

    return data_source
