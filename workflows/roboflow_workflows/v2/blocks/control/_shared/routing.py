"""Value routing and throttling: V2 ports of three V1 flow-control blocks.

    roboflow_core/switch_case@v1    route to the first case whose key equals str(value)
    roboflow_core/delta_filter@v1   continue only when a lane's value changed
    roboflow_core/rate_limiter@v1   continue at most once per cooldown per lane

The two stateful blocks keep one record per lane::

    lane = ("video", video_metadata.video_identifier)  image with video metadata
           ("source", SampleContext.source_id)          else, lane field's source
           ("position", index[:1])                      else (passive fallback)

* The lane field is ``image`` for delta_filter, and ``video_reference_image``
  (else ``depends_on``) for rate_limiter. Only that field's source and time
  count, not the whole call's: a context-free value next to an image with a
  source still uses the image's source.
* Crops keep their frame's video metadata, so crops of one frame share the
  frame's lane. A plain ``session.run`` without metadata gives each top-level
  batch position its own lane across runs; a one-image run behaves like V1.
* Order comes from the engine, not from the lock. One run calls a step's
  invocations one after another in index order; pipelined runs keep pulse
  order within a domain. Calls of different domains may overlap: one lock per
  block keeps the records consistent, and a lane fed by two domains sees
  arrival order.
* Capacity: ``max_sources`` bounds the records. One lane too many fails with
  ``SourceCapacityError`` and changes nothing. Records are never evicted by
  time; a processing reset replaces the block and its records.

``rate_limiter`` differs from V1 on purpose: each lane has its own cooldown
(``scope="global"`` keeps V1's single one), and without a usable video time
it measures the observation time of the lane field's sample instead of the
wall clock when the block runs.
"""

import dataclasses
import math
from fractions import Fraction
from typing import Any, Dict, Hashable, Optional, Tuple
import numpy as np
import torch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import Block


DEFAULT_MAX_SOURCES = 256

Lane = Tuple[Hashable, ...]

class SourceCapacityError(RuntimeError):
    """A new lane's record would exceed the block's ``max_sources``."""

class ClockMismatchError(ValueError):
    """A rate limiter lane received times from two different clocks."""

def _values_equal(first: Any, second: Any) -> bool:
    # Python == for scalars and plain objects, so 1 == 1.0 == True. Containers,
    # tensors, arrays and dataclasses compare element by element; a 0-d tensor
    # equals the Python scalar it holds. Raises TypeError when == is no bool.
    if first is second:
        return True

    first_tensor = isinstance(first, torch.Tensor)
    second_tensor = isinstance(second, torch.Tensor)
    if first_tensor and second_tensor:
        same = first.shape == second.shape and torch.equal(
            first, second.to(first.device)
        )
        return same
    if first_tensor or second_tensor:
        tensor, other = (first, second) if first_tensor else (second, first)
        same = tensor.dim() == 0 and _is_scalar(other) and tensor.item() == other
        return same

    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        same = (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and first.shape == second.shape
            and bool(np.array_equal(first, second))
        )
        return same

    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        same = (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_values_equal(a, b) for a, b in zip(first, second))
        )
        return same

    if isinstance(first, dict) or isinstance(second, dict):
        same = (
            isinstance(first, dict)
            and isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_values_equal(first[key], second[key]) for key in first)
        )
        return same

    if _is_dataclass_instance(first) or _is_dataclass_instance(second):
        same = type(first) is type(second) and all(
            _values_equal(getattr(first, field.name), getattr(second, field.name))
            for field in dataclasses.fields(first)
        )
        return same

    result = first == second
    if not isinstance(result, (bool, np.bool_)):
        raise TypeError(
            f"cannot decide whether {type(first).__name__} and "
            f"{type(second).__name__} values are equal: == returned "
            f"{type(result).__name__}, not a bool"
        )

    return bool(result)

def _within_cooldown(elapsed: Fraction, cooldown_seconds: float) -> bool:
    if math.isinf(cooldown_seconds):
        return True

    # str() keeps the written value: 0.1 is 1/10, not its binary float.
    within = elapsed < Fraction(str(cooldown_seconds))

    return within

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex, np.generic))

def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)

def _lane_of(image: Optional[ImageData], *, block: Block, field: str) -> Lane:
    metadata = image.video_metadata if image is not None else None
    if metadata is not None:
        return ("video", metadata.video_identifier)

    # The field's own source: the whole call's merge is None when another
    # argument carries no context, which would collapse sources to positions.
    context = block.execution_context
    index = context.indices[0]
    sample = context.sample_at(index, field=field)
    if sample is not None:
        return ("source", sample.source_id)

    lane = ("position", index[:1])

    return lane

def _video_time_of(image: Optional[ImageData]) -> Optional[Tuple[str, Fraction]]:
    metadata = image.video_metadata if image is not None else None
    if metadata is None:
        return None

    fps = metadata.fps
    if fps is None or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        return None

    # str(fps) keeps the written rate: 29.97 is 2997/100, not its binary float.
    seconds = Fraction(metadata.frame_number) / Fraction(str(fps))
    clock_id = f"video-time:{metadata.video_identifier}"

    return clock_id, seconds

def _check_clock(last_clock_id: str, *, clock_id: str, lane: Lane) -> None:
    if clock_id != last_clock_id:
        raise ClockMismatchError(
            f"roboflow_core/rate_limiter@v1: lane {lane!r} last passed on clock "
            f"{last_clock_id!r} and is now timed on {clock_id!r}; times of two "
            "clocks cannot be compared. Give every call of a lane the same "
            "video_reference_image (with a finite positive fps) or none; with "
            "scope='global', video time works only for one video."
        )

def _check_capacity(
    block: Block, *, records: Dict[Lane, Any], lane: Lane, limit: int
) -> None:
    if len(records) >= limit:
        raise SourceCapacityError(
            f"{block.type}: lane {lane!r} would be lane {len(records) + 1} of "
            f"max_sources={limit}. The block keeps one record per video "
            "stream, source or batch position for the step's lifetime; none "
            "is dropped by time. Raise max_sources, or reset processing to "
            "drop the records."
        )
