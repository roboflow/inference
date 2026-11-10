"""Per-row metadata helpers shared by the geometry blocks."""

from typing import Any, Dict, Optional

# Host copies of tensor fields that some producers keep per row. They go stale
# when a block changes boxes, classes or confidences. data.coordinates and
# blocks.query keep their own copies of the same keys.
HOST_MIRROR_KEYS = ("_host_xyxy", "_host_class_id", "_host_confidence")


def without_host_mirrors(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of one row's metadata without stale host mirrors.

    Args:
        row: Row metadata, or ``None``.

    Returns:
        A new dict; empty for ``None``.
    """
    cleaned = {
        key: value for key, value in (row or {}).items() if key not in HOST_MIRROR_KEYS
    }

    return cleaned
