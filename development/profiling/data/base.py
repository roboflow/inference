from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class DataRecord:
    """Reusable input record emitted by profiling data sources."""

    id: str
    image: Any | None = None
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    """Interface implemented by profiling data sources."""

    def iter_records(self) -> Iterable[DataRecord]:
        """Iterate over deterministic profiling records.

        Returns:
            Iterable of selected profiling records.
        """
        ...

    def describe(self) -> Mapping[str, Any]:
        """Describe the data source for manifest output.

        Returns:
            Data-source-specific metadata.
        """
        ...
