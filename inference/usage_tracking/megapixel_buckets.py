"""Billable megapixel bucketing for model usage telemetry.

Billing uses post-preprocess / fixed model input size (not native upload size).
Bucket maps are merge-friendly sums; averages are derived downstream.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Inclusive upper bounds in megapixels. Final bucket catches everything above.
_MEGAPIXEL_BUCKET_UPPER_BOUNDS: Tuple[Tuple[str, float], ...] = (
    ("0-0.25", 0.25),
    ("0.25-0.5", 0.5),
    ("0.5-1", 1.0),
    ("1-2", 2.0),
    ("2-4", 4.0),
)
_MEGAPIXEL_BUCKET_OVERFLOW = "4+"


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
    height: int,
    width: int,
    frames: int,
    execution_duration: float,
) -> Dict[str, Dict[str, Union[int, float]]]:
    if frames <= 0 or height <= 0 or width <= 0:
        return {}

    bucket = megapixel_bucket_for_hw(height=height, width=width)
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


def _square_image_size(value: Any) -> Optional[Tuple[int, int]]:
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
        square = _square_image_size(getattr(model, attr, None))
        if square is not None:
            return square

    # SAM adapters wrap torch backends that expose image_size / _image_size.
    for attr in ("_model", "sam", "sam_model"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        for size_attr in ("image_size", "_image_size", "img_size"):
            square = _square_image_size(getattr(inner, size_attr, None))
            if square is not None:
                return square
        nested = getattr(inner, "_model", None)
        if nested is not None:
            for size_attr in ("image_size", "_image_size", "img_size"):
                square = _square_image_size(getattr(nested, size_attr, None))
                if square is not None:
                    return square

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
                import json

                preproc = json.loads(preproc)
            except Exception:
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
    except Exception:
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
    if shape is None or len(shape) < 4:
        return None
    return _as_positive_int(shape[0])


def count_inference_images(image: Any) -> int:
    if image is None:
        return 0
    if isinstance(image, (list, tuple)):
        return len(image)
    return 1


def stamp_billable_model_input(model: Any, preprocessed_tensor: Any) -> None:
    """Store post-preprocess spatial metadata for the usage decorator to read."""
    try:
        model._usage_billable_input_hw = get_tensor_spatial_hw(preprocessed_tensor)
        model._usage_billable_frames = get_tensor_batch_size(preprocessed_tensor)
    except Exception:
        pass


def stamp_billable_model_hw(
    model: Any,
    *,
    height: int,
    width: int,
    frames: int = 1,
) -> None:
    """Stamp explicit billable spatial size (used by SAM request entrypoints)."""
    try:
        model._usage_billable_input_hw = (int(height), int(width))
        model._usage_billable_frames = max(int(frames), 1)
    except Exception:
        pass


def clear_billable_model_input(model: Any) -> None:
    for attr in (
        "_usage_billable_input_hw",
        "_usage_billable_frames",
    ):
        try:
            setattr(model, attr, None)
        except Exception:
            pass


def billable_hw_from_model(model: Any) -> Optional[Tuple[int, int]]:
    fixed_hw = get_fixed_model_input_hw(model)
    if fixed_hw is not None:
        return fixed_hw
    stamped = getattr(model, "_usage_billable_input_hw", None)
    if (
        isinstance(stamped, Sequence)
        and len(stamped) == 2
        and _as_positive_int(stamped[0]) is not None
        and _as_positive_int(stamped[1]) is not None
    ):
        return int(stamped[0]), int(stamped[1])
    return None


def prepare_sam_usage_billing(model: Any, request: Any = None) -> None:
    """Clear prior stamps and record SAM encoder input size for usage billing.

    SAM entrypoints decorate ``infer_from_request`` rather than ``BaseInference.infer``,
    so they must stamp billable HW explicitly. Prefer the model's fixed encoder size
    (``image_size`` / nested backend) over native upload resolution.
    """
    clear_billable_model_input(model)
    frames = count_inference_images(
        getattr(request, "image", None) if request else None
    )
    if frames <= 0:
        frames = 1
    billable_hw = billable_hw_from_model(model)
    if billable_hw is None:
        model._usage_billable_frames = frames
        return
    stamp_billable_model_hw(
        model,
        height=billable_hw[0],
        width=billable_hw[1],
        frames=frames,
    )
