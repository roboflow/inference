"""Native video clips for Vision Events: shared encoder, capture and upload code.

The recorder block (`roboflow_core/video_clip_recorder@v1`) buffers the frames a
workflow processes in a rolling H.264 ring per `video_identifier`; the Vision
Events sink captures a window around an event from that ring and uploads it.

Modules:

* `grid` - pure planner: output slot grid, 90 kHz PTS, final-frame duration.
* `ring` - `Ring` protocol, `RingStats`, the refcounted packet store and
  `BaseEncodedRing`, the backend-neutral ring every encoder subclasses.
* `x264_ring` - the default libx264 backend (`X264Ring`).
* `capture` - `ClipAnchor`, `PendingCapture` and the partial reasons.
* `artifact` - `ClipArtifact`, the muxed MP4 handed to the uploader.
* `report` - `ClipReport`, the bounds and loss accounting of one clip.
* `admission` - `Budget`, the process-wide resource caps.
* `uploader` - `ClipUploader`, signed-URL PUT and the completion record.

Process settings are read once, at import, from the environment variables
below (the same convention as `inference/core/env.py`). Invalid or
non-positive values fall back to the default with a warning. `read_env` is the
package's only environment read; the recorder and NVENC backends read their own
settings through `env_positive_float` / `env_positive_int`.
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
GIB = 1024 * MIB


def read_env(name: str) -> Optional[str]:
    """Read one environment variable (the package's single environment read).

    Args:
        name: Environment variable name.

    Returns:
        The stripped value, or ``None`` when unset or empty.
    """
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return None

    return raw_value.strip()


def env_positive_float(
    name: str, *, default: float, maximum: Optional[float] = None
) -> float:
    """Read a positive number setting from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset, empty or invalid.
        maximum: Largest accepted value; larger values are clamped to it.

    Returns:
        The parsed value, or ``default``.
    """
    raw_value = read_env(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s.", name, raw_value, default)
        return default

    if not math.isfinite(value) or value <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using %s.", name, raw_value, default
        )
        return default

    if maximum is not None and value > maximum:
        logger.warning("Clamping %s=%r to %s.", name, raw_value, maximum)
        return maximum

    return value


def env_positive_int(name: str, *, default: int) -> int:
    """Read a positive integer setting from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset, empty or invalid.

    Returns:
        The parsed value (fractions are truncated), or ``default``.
    """
    value = int(env_positive_float(name, default=float(default)))
    if value <= 0:
        return default

    return value


def env_optional_path(name: str) -> Optional[Path]:
    """Read an optional directory setting from the environment.

    Args:
        name: Environment variable name.

    Returns:
        The path, or ``None`` when the variable is unset or empty.
    """
    raw_value = read_env(name)
    if raw_value is None:
        return None

    return Path(raw_value)


WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES = env_positive_int(
    "WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES", default=GIB
)
WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES = env_positive_int(
    "WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES", default=16
)
WORKFLOWS_VIDEO_CLIP_MAX_UPLOAD_CONCURRENCY = env_positive_int(
    "WORKFLOWS_VIDEO_CLIP_MAX_UPLOAD_CONCURRENCY", default=2
)
WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES = env_positive_int(
    "WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES", default=512 * MIB
)
WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES = env_positive_int(
    "WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES", default=64 * MIB
)
WORKFLOWS_VIDEO_CLIP_TEMP_DIR = env_optional_path("WORKFLOWS_VIDEO_CLIP_TEMP_DIR")
