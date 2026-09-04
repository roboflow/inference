"""Compact runtime-selection metadata shared by optimized model paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol

OPTIMIZATION_RUNTIME_METADATA_SCHEMA_VERSION = "1.0"


class SupportsOptimizationRuntimeMetadata(Protocol):
    """Model exposing its resolved optimization path for diagnostics."""

    @property
    def optimization_runtime_metadata(self) -> Mapping[str, Any]:
        """Return versioned requested, selected, and last-executed stage data.

        Returns:
            Bounded runtime selection metadata for diagnostics and profiling proof.
        """


@dataclass(frozen=True)
class SelectionSnapshot(Mapping[str, Any]):
    """Immutable requested/effective selection retained on the hot path."""

    requested_id: str
    effective_id: str
    fallback_reason: Optional[str] = None

    @property
    def fallback_occurred(self) -> bool:
        """Report whether this resolution selected a fallback.

        Returns:
            True when the selection includes a fallback reason.
        """
        return self.fallback_reason is not None

    def __getitem__(self, key: str) -> Any:
        """Expose the immutable snapshot through the legacy mapping interface.

        Args:
            key: Selection field to read.

        Returns:
            The requested selection value.

        Raises:
            KeyError: If the field is not present in this snapshot.
        """
        if key == "requested_id":
            value: Any = self.requested_id
        elif key == "effective_id":
            value = self.effective_id
        elif key == "fallback_occurred":
            value = self.fallback_occurred
        elif key == "fallback_reason":
            value = self.fallback_reason
        else:
            raise KeyError(key)

        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate over fields available in this snapshot.

        Returns:
            Iterator over the mapping keys.
        """
        keys = [
            "requested_id",
            "effective_id",
            "fallback_occurred",
            "fallback_reason",
        ]
        iterator = iter(keys)

        return iterator

    def __len__(self) -> int:
        """Return the number of fields available in this snapshot.

        Returns:
            Number of mapping keys.
        """
        value = 4
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize only when metadata is explicitly requested.

        Returns:
            Requested/effective IDs and bounded fallback information.
        """
        value: Dict[str, Any] = {
            "requested_id": self.requested_id,
            "effective_id": self.effective_id,
            "fallback_occurred": self.fallback_occurred,
        }
        if self.fallback_reason is not None:
            value["fallback_reason"] = self.fallback_reason

        return value
