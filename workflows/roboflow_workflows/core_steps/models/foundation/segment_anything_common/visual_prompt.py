"""Shared point-prompt parsing and synthetic class metadata."""

from typing import Any, List, Optional, Tuple

SYNTHETIC_POINT_PROMPT_CLASS_ID = -1
SYNTHETIC_POINT_PROMPT_CLASS_NAME = "foreground"


def normalise_labeled_points(
    points: Optional[List[Any]],
) -> List[Tuple[float, float, bool]]:
    """Convert workflow point values to ``(x, y, positive)`` tuples."""
    if not points:
        return []
    result: List[Tuple[float, float, bool]] = []
    for raw_point in points:
        if isinstance(raw_point, dict):
            if "x" not in raw_point or "y" not in raw_point:
                raise ValueError(
                    "Each point prompt must define `x` and `y` coordinates. "
                    f"Got: {raw_point!r}."
                )
            x, y = raw_point["x"], raw_point["y"]
            positive = raw_point.get("positive", True)
        elif isinstance(raw_point, (list, tuple)) and len(raw_point) in {2, 3}:
            x, y = raw_point[:2]
            positive = raw_point[2] if len(raw_point) == 3 else True
        else:
            raise ValueError(
                "Each point prompt must be an object or a sequence with two or "
                f"three values. Got: {raw_point!r}."
            )
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError(f"Point coordinates must be numbers. Got: {raw_point!r}.")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Point coordinates must be numbers. Got: {raw_point!r}.")
        result.append((float(x), float(y), bool(positive)))
    return result
