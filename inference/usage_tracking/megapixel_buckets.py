"""Megapixel bucketing for model usage telemetry.

Buckets record the post-preprocess / fixed model input size, which is the
resolution the model actually ran at, rather than the native upload size. Bucket
maps are merge-friendly sums; averages are derived downstream.

The measured input size is published through a :class:`~contextvars.ContextVar`
rather than an attribute on the model. Model instances are shared across the
server's worker threads, so an attribute would let one request overwrite the
size and frame count of another request that is still running.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple, Union

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

MeasuredModelInput = Tuple[Optional[Tuple[int, int]], Optional[int]]

_measured_model_input: ContextVar[Optional[MeasuredModelInput]] = ContextVar(
    "usage_measured_model_input",
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


def _image_size_to_hw(value: Any) -> Optional[Tuple[int, int]]:
    """Normalize an ``image_size``-style attribute to (height, width).

    Backends express it either as a single edge length for square inputs or as
    an explicit (height, width) pair.
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            return None
        height = _as_positive_int(value[0])
        width = _as_positive_int(value[1])
        if height is None or width is None:
            return None
        return height, width
    size = _as_positive_int(value)
    if size is None:
        return None
    return size, size


def get_fixed_model_input_hw(model: Any) -> Optional[Tuple[int, int]]:
    """Return (height, width) when the model has a fixed numeric input size."""
    height = _as_positive_int(getattr(model, "img_size_h", None))
    width = _as_positive_int(getattr(model, "img_size_w", None))
    if height is not None and width is not None:
        return height, width

    for attr in ("image_size", "img_size"):
        size = _image_size_to_hw(getattr(model, attr, None))
        if size is not None:
            return size

    # Adapters wrap backends that expose image_size / _image_size.
    for attr in ("_model", "sam", "sam_model", "owlv2"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        for size_attr in ("image_size", "_image_size", "img_size"):
            size = _image_size_to_hw(getattr(inner, size_attr, None))
            if size is not None:
                return size
        nested = getattr(inner, "_model", None)
        if nested is not None:
            for size_attr in ("image_size", "_image_size", "img_size"):
                size = _image_size_to_hw(getattr(nested, size_attr, None))
                if size is not None:
                    return size

    environment = getattr(model, "environment", None)
    if isinstance(environment, dict):
        resolution = environment.get("RESOLUTION")
        if isinstance(resolution, (list, tuple)) and resolution:
            resolution = resolution[0]
        resolution = _as_positive_int(resolution)
        if resolution is not None:
            return resolution, resolution

        preproc = environment.get("PREPROCESSING")
        if isinstance(preproc, str):
            try:
                preproc = json.loads(preproc)
            except ValueError:
                preproc = None
        if isinstance(preproc, dict):
            resize = preproc.get("resize") or {}
            if isinstance(resize, dict):
                height = _as_positive_int(resize.get("height"))
                width = _as_positive_int(resize.get("width"))
                if height is not None and width is not None:
                    return height, width

    preproc = getattr(model, "preproc", None)
    if isinstance(preproc, dict):
        resize = preproc.get("resize") or {}
        if isinstance(resize, dict):
            height = _as_positive_int(resize.get("height"))
            width = _as_positive_int(resize.get("width"))
            if height is not None and width is not None:
                return height, width

    return None


def get_tensor_spatial_hw(tensor: Any) -> Optional[Tuple[int, int]]:
    """Best-effort spatial (height, width) from a preprocessed model tensor."""
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    if len(dims) < 2:
        return None

    # NCHW / CHW
    if len(dims) >= 3 and dims[-3] in (1, 3, 4):
        height = dims[-2]
        width = dims[-1]
        if height > 0 and width > 0:
            return height, width

    # NHWC / HWC
    if dims[-1] in (1, 3, 4):
        height = dims[-3] if len(dims) >= 3 else dims[-2]
        width = dims[-2] if len(dims) >= 3 else dims[-1]
        if height > 0 and width > 0:
            return height, width

    height = dims[-2]
    width = dims[-1]
    if height > 0 and width > 0:
        return height, width
    return None


def get_tensor_batch_size(tensor: Any) -> Optional[int]:
    shape = getattr(tensor, "shape", None)
    try:
        if shape is None or len(shape) < 4:
            return None
        return _as_positive_int(shape[0])
    except TypeError:
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


def clear_measured_model_input() -> None:
    _measured_model_input.set(None)


def record_measured_model_input(preprocessed_tensor: Any) -> None:
    """Publish post-preprocess spatial metadata for the usage decorator to read."""
    try:
        _measured_model_input.set(
            (
                get_tensor_spatial_hw(preprocessed_tensor),
                get_tensor_batch_size(preprocessed_tensor),
            )
        )
    except Exception:
        pass


def record_measured_model_hw(
    *,
    height: int,
    width: int,
    frames: Optional[int] = None,
) -> None:
    """Publish an explicit input size (used by SAM request entrypoints)."""
    try:
        _measured_model_input.set(
            (
                (int(height), int(width)),
                max(int(frames), 1) if frames else None,
            )
        )
    except Exception:
        pass


def consume_measured_model_input() -> MeasuredModelInput:
    """Read and clear the input size published by the current call.

    Clearing on read keeps a stale value from leaking into a later call that did
    not publish one, which would otherwise attribute it to the wrong resolution.
    """
    measured = _measured_model_input.get()
    if measured is None:
        return None, None
    _measured_model_input.set(None)
    return measured


def resolve_model_input_hw(
    model: Any,
    measured_hw: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """Fixed model input size when there is one, otherwise the observed size."""
    if model is not None:
        fixed_hw = get_fixed_model_input_hw(model)
        if fixed_hw is not None:
            return fixed_hw
    if (
        isinstance(measured_hw, tuple)
        and len(measured_hw) == 2
        and _as_positive_int(measured_hw[0]) is not None
        and _as_positive_int(measured_hw[1]) is not None
    ):
        return int(measured_hw[0]), int(measured_hw[1])
    return None


def record_sam_model_input(model: Any, request: Any = None) -> None:
    """Publish the SAM encoder input size for usage telemetry.

    SAM entrypoints decorate ``infer_from_request`` rather than
    ``BaseInference.infer``, so they publish their input size explicitly. The
    model's fixed encoder size is preferred over native upload resolution.
    """
    clear_measured_model_input()
    input_hw = resolve_model_input_hw(model)
    if input_hw is None:
        return
    record_measured_model_hw(
        height=input_hw[0],
        width=input_hw[1],
        frames=count_inference_images(getattr(request, "image", None)),
    )
