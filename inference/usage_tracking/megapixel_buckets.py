"""Megapixel bucketing for model usage telemetry.

Buckets are keyed on the size of the image the caller sent, measured before any
resize, so a bucket describes the work a request asked for rather than the
shape the model happens to run at. Model input size is deliberately not used
here. ``unknown`` covers calls whose native size is not recoverable, which
keeps the sum of bucket frames equal to the row's processed frames. Bucket maps
are merge-friendly sums; averages are derived downstream.

Preprocess records that size under one of four conventions, all read here:
``img_dims``, a per-image sequence of (height, width), from core Roboflow
models; ``image_dims``, a single (width, height) pair, from the VLM and depth
families; a per-image sequence of records carrying ``original_size``, from the
``inference_models`` detection, segmentation and keypoint adapters; and a bare
per-image sequence of (height, width), from the ``inference_models``
classification adapter. A batch is attributed to its first image, since a call
reports one bucket.

Bucket ``execution_duration`` is the model's predict phase alone, excluding
pre- and post-processing, for entrypoints that separate the phases. Everything
else falls back to the decorator's full call duration, which is also what the
row-level ``execution_duration`` always reports. Bucket durations therefore do
not reconcile against the row total. Falling back are the families that
override ``infer()`` wholesale, the SAM ``infer_from_request`` entrypoints, and
any call whose ``predict()`` defers its work. Timing of that phase lives in
:mod:`inference.usage_tracking.predict_timing`.

The measured image size is published through :class:`~contextvars.ContextVar`
rather than an attribute on the model. Model instances are shared across the
server's worker threads, so an attribute would let one request overwrite the
measurement of another request that is still running.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple, Union

from inference.usage_tracking.predict_timing import clear_measured_predict_duration

# Inclusive upper bounds in megapixels. Final bucket catches everything above.
_MEGAPIXEL_BUCKET_UPPER_BOUNDS: Tuple[Tuple[str, float], ...] = (
    ("0-0.25", 0.25),
    ("0.25-0.5", 0.5),
    ("0.5-1", 1.0),
    ("1-2", 2.0),
    ("2-4", 4.0),
    ("4-8", 8.0),
)
_MEGAPIXEL_BUCKET_OVERFLOW = "8+"

# Frames whose input size could not be determined. Recording them keeps the sum
# of bucket frames equal to the row's processed_frames, so downstream consumers
# can always reconcile the two.
MEGAPIXEL_BUCKET_UNKNOWN = "unknown"

MeasuredImageInput = Tuple[Optional[Tuple[int, int]], Optional[int]]

_measured_image_input: ContextVar[Optional[MeasuredImageInput]] = ContextVar(
    "usage_measured_image_input",
    default=None,
)


def megapixels_from_hw(height: int, width: int) -> float:
    return (max(height, 0) * max(width, 0)) / 1_000_000.0


def megapixel_bucket_for_hw(height: int, width: int) -> str:
    megapixels = megapixels_from_hw(height=height, width=width)
    for bucket_name, upper_bound in _MEGAPIXEL_BUCKET_UPPER_BOUNDS:
        if megapixels <= upper_bound:
            return bucket_name
    return _MEGAPIXEL_BUCKET_OVERFLOW


def build_megapixel_buckets(
    *,
    height: Optional[int],
    width: Optional[int],
    frames: int,
    execution_duration: float,
) -> Dict[str, Dict[str, Union[int, float]]]:
    if frames <= 0:
        return {}

    if height and width and height > 0 and width > 0:
        bucket = megapixel_bucket_for_hw(height=height, width=width)
    else:
        bucket = MEGAPIXEL_BUCKET_UNKNOWN
    return {
        bucket: {
            "processed_frames": int(frames),
            "execution_duration": float(execution_duration),
        }
    }


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _get_pixel_values(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("pixel_values")
    pixel_values = getattr(value, "pixel_values", None)
    if pixel_values is not None:
        return pixel_values
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter("pixel_values")
        except (TypeError, ValueError, KeyError):
            return None
    return None


def get_tensor_batch_size(tensor: Any) -> Optional[int]:
    pixel_values = _get_pixel_values(tensor)
    if pixel_values is not None and pixel_values is not tensor:
        return get_tensor_batch_size(pixel_values)

    shape = getattr(tensor, "shape", None)
    try:
        if shape is None or len(shape) < 4:
            return None
        return _as_positive_int(shape[0])
    except TypeError:
        return None


def _read_metadata_key(metadata: Any, key: str) -> Any:
    """Read a key from preprocess metadata, dict-like or attribute-style."""
    if isinstance(metadata, dict):
        return metadata.get(key)
    getter = getattr(metadata, "get", None)
    if callable(getter):
        try:
            value = getter(key)
        except (TypeError, ValueError, KeyError):
            value = None
        if value is not None:
            return value
    return getattr(metadata, key, None)


def _hw_pair(height: Any, width: Any) -> Optional[Tuple[int, int]]:
    height = _as_positive_int(height)
    width = _as_positive_int(width)
    if height is None or width is None:
        return None
    return height, width


def _first_dims_pair(dims: Any) -> Optional[Tuple[Any, Any]]:
    """The leading two-element pair, unwrapping a per-image sequence.

    Dims arrive as one pair per image even for a single image, so the first
    element is unwrapped when it is itself a pair. A bare pair is taken as-is.
    """
    if not isinstance(dims, (tuple, list)) or not dims:
        return None
    if isinstance(dims[0], (tuple, list)):
        dims = dims[0]
    if not isinstance(dims, (tuple, list)) or len(dims) != 2:
        return None
    return dims[0], dims[1]


def _hw_from_original_size(metadata: Any) -> Optional[Tuple[int, int]]:
    """(height, width) from an ``inference_models`` per-image metadata record."""
    record = metadata
    if not hasattr(record, "original_size"):
        try:
            record = metadata[0]
        except (TypeError, KeyError, IndexError):
            return None
    original_size = getattr(record, "original_size", None)
    if original_size is None:
        return None
    return _hw_pair(
        getattr(original_size, "height", None),
        getattr(original_size, "width", None),
    )


def parse_image_input_hw(metadata: Any) -> Optional[Tuple[int, int]]:
    """Native (height, width) of the image a preprocess call was handed.

    Reads whichever of the four conventions the model family uses, in
    descending order of how explicit they are. A batch is attributed to its
    first image.

    Args:
        metadata: The metadata returned alongside the preprocessed tensor.

    Returns:
        (height, width) before any resize, or None when no convention yielded a
        usable pair.
    """
    if metadata is None:
        return None

    pair = _first_dims_pair(_read_metadata_key(metadata, "img_dims"))
    if pair is not None:
        image_hw = _hw_pair(pair[0], pair[1])
        if image_hw is not None:
            return image_hw

    pair = _first_dims_pair(_read_metadata_key(metadata, "image_dims"))
    if pair is not None:
        image_hw = _hw_pair(pair[1], pair[0])
        if image_hw is not None:
            return image_hw

    image_hw = _hw_from_original_size(metadata)
    if image_hw is not None:
        return image_hw

    # A bare per-image sequence of (height, width), with no key to name it.
    if not isinstance(metadata, dict):
        pair = _first_dims_pair(metadata)
        if pair is not None:
            return _hw_pair(pair[0], pair[1])

    return None


def count_inference_images(image: Any) -> int:
    """Number of images the caller asked the model to process.

    A single array counts as one image even when it carries a batch dimension;
    the request is the authority on how many frames were asked for.
    """
    if image is None:
        return 0
    if isinstance(image, (list, tuple)):
        return len(image)
    return 1


def clear_measured_image_input() -> None:
    """Reset every per-call measurement published for the usage decorator.

    Called at the start of a model call so that a measurement published by an
    earlier call cannot be attributed to this one.
    """
    _measured_image_input.set(None)
    clear_measured_predict_duration()


def record_measured_image_input(
    image_hw: Optional[Tuple[int, int]],
    *,
    frames: Optional[int] = None,
) -> None:
    """Publish the caller's image size for the usage decorator to read.

    Args:
        image_hw: Native (height, width), or None when preprocess recorded
            neither dims convention.
        frames: Batch size of the preprocessed tensor, used only as a frame
            count fallback for calls whose images are not introspectable.
    """
    try:
        measured_hw = None
        if image_hw is not None:
            height = _as_positive_int(image_hw[0])
            width = _as_positive_int(image_hw[1])
            if height is not None and width is not None:
                measured_hw = (height, width)

        _measured_image_input.set((measured_hw, frames))
    except Exception:
        pass


def consume_measured_image_input() -> MeasuredImageInput:
    """Read and clear the image size published by the current call.

    Clearing on read keeps a stale value from leaking into a later call that did
    not publish one, which would otherwise attribute it to the wrong resolution.
    """
    measured = _measured_image_input.get()
    if measured is None:
        return None, None
    _measured_image_input.set(None)
    return measured
