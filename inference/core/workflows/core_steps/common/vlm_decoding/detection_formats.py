"""Box coordinate formats used by VLM object-detection prompts.

Decoding a VLM detection answer depends on the *coordinate contract the
prompt asked for*, not on the model vendor. Two vendors that are prompted
for ``box_2d`` integers normalized to 0-1000 produce identical payloads and
must be decoded identically. This module therefore keys both the prompt
wording and the box conversion by format name, so a block only has to pick
the format that scores best for its model.

Every converter returns ``[x_min, y_min, x_max, y_max]`` in pixels of the
ORIGINAL image, or ``None`` when the entry carries no well-formed box (the
caller skips such entries instead of failing the whole response). All
formats clamp coordinates to their nominal range before scaling, so mildly
out-of-range model output stays inside the image.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from inference.core.workflows.core_steps.common.vlm_decoding.utils import (
    scale_confidence,
)

# ---------------------------------------------------------------------------
# Prompt templates, one per coordinate format.
#
# `{class_list}` is always available. `{width}`/`{height}` are available only
# for formats with `requires_upload_dimensions=True`, and refer to the
# dimensions of the image as actually uploaded to the model.
# ---------------------------------------------------------------------------

XYXY_ABSOLUTE_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners in absolute pixel coordinates "
    "of the {width}x{height} pixel image. "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

XYXY_0_1000_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners as integers between 0 and 1000, "
    "normalized to the image width (x) and height (y). "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

YXYX_0_1000_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [y_min, x_min, y_max, x_max]: integers '
    "between 0 and 1000, normalized to the image height and width. "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

XYXY_PERCENT_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the text label in the key "
    '"label" and the 2D bounding box in the key "box_2d". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max] as percentages '
    "of image width and height (floats between 0 and 100). "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

NAMED_0_1000_PROMPT_TEMPLATE = (
    "You are an object grounding expert. Detect all objects in this image "
    "matching these labels: {class_list}. Ensure the objects accurately "
    "match the request and do not miss any objects. For each object, answer "
    'in the format {{"label": "<name>", "x_min": <int>, "y_min": <int>, '
    '"x_max": <int>, "y_max": <int>}}. The coordinates should be in the '
    "0-1000 range. Return a JSON array of results. If you cannot find an "
    "object, omit it from the results."
)

# Coordinate scales.
NORMALIZED_0_1000_SCALE = 1000.0
PERCENT_SCALE = 100.0

# Entry shape vocabulary. Models drift between the prompted keys and their
# native grounding vocabulary, so every accepted alias is listed here once.
BOX_2D_KEYS = ("box_2d", "bbox_2d")
NAMED_BOX_FIELDS = ("x_min", "y_min", "x_max", "y_max")
LABEL_KEYS = ("label", "class_name", "class", "description")
DETECTIONS_WRAPPER_KEY = "detections"


def _clamp(value: float, upper: float) -> float:
    return min(max(value, 0.0), upper)


def _read_number(value: Any) -> Optional[float]:
    """Coerce a JSON value into a plain finite float, or ``None``.

    Numeric strings (``"12"``, ``"0.5"``) are accepted: models routinely
    quote coordinates and the per-vendor parsers this module replaces used
    a bare ``float()``, so rejecting them would be a regression.

    Bools, non-numeric strings and NaN/inf are rejected: ``json.loads``
    accepts bare ``NaN`` and a NaN would survive clamping straight into
    ``xyxy``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    elif not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _read_box_2d(entry: dict) -> Optional[List[float]]:
    """Read a 4-element ``box_2d``/``bbox_2d`` list from an entry."""
    for key in BOX_2D_KEYS:
        box = entry.get(key)
        if not isinstance(box, list) or len(box) != 4:
            continue
        values = [_read_number(value) for value in box]
        if any(value is None for value in values):
            continue
        return values
    return None


def _read_named_box(entry: dict) -> Optional[List[float]]:
    """Read flat ``x_min``/``y_min``/``x_max``/``y_max`` fields."""
    box: List[float] = []
    for field in NAMED_BOX_FIELDS:
        value = _read_number(entry.get(field))
        if value is None:
            return None
        box.append(value)
    return box


def _require_upload_dimensions(
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> None:
    if not upload_width or not upload_height:
        raise ValueError(
            "Box format requires the dimensions of the uploaded image to map "
            "coordinates back onto the original image, but none were provided."
        )


def _convert_xyxy_absolute(
    entry: dict,
    image_width: int,
    image_height: int,
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> Optional[List[float]]:
    _require_upload_dimensions(upload_width, upload_height)
    box = _read_box_2d(entry)
    if box is None:
        return _named_normalized_fallback(entry, image_width, image_height)
    scale_x = image_width / upload_width
    scale_y = image_height / upload_height
    x_min, y_min, x_max, y_max = box
    x_min = _clamp(x_min, upload_width)
    x_max = _clamp(x_max, upload_width)
    y_min = _clamp(y_min, upload_height)
    y_max = _clamp(y_max, upload_height)
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def _convert_xyxy_0_1000(
    entry: dict,
    image_width: int,
    image_height: int,
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> Optional[List[float]]:
    box = _read_box_2d(entry)
    if box is None:
        return None
    x_min, y_min, x_max, y_max = (
        _clamp(value, NORMALIZED_0_1000_SCALE) for value in box
    )
    scale_x = image_width / NORMALIZED_0_1000_SCALE
    scale_y = image_height / NORMALIZED_0_1000_SCALE
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def _convert_yxyx_0_1000(
    entry: dict,
    image_width: int,
    image_height: int,
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> Optional[List[float]]:
    box = _read_box_2d(entry)
    if box is None:
        return _named_normalized_fallback(entry, image_width, image_height)
    y_min, x_min, y_max, x_max = (
        _clamp(value, NORMALIZED_0_1000_SCALE) for value in box
    )
    scale_x = image_width / NORMALIZED_0_1000_SCALE
    scale_y = image_height / NORMALIZED_0_1000_SCALE
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def _convert_xyxy_percent(
    entry: dict,
    image_width: int,
    image_height: int,
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> Optional[List[float]]:
    box = _read_box_2d(entry)
    if box is None:
        return None
    x_min, y_min, x_max, y_max = (_clamp(value, PERCENT_SCALE) for value in box)
    scale_x = image_width / PERCENT_SCALE
    scale_y = image_height / PERCENT_SCALE
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def _named_normalized_fallback(
    entry: dict, image_width: int, image_height: int
) -> Optional[List[float]]:
    """Accept a legacy ``x_min``.. entry normalized to 0-1 for a ``box_2d`` format.

    The deprecated Gemini, OpenAI and Claude parsers took this shape whenever
    ``box_2d`` was missing (older block versions prompted for it), so the
    formats those blocks moved to keep the same tolerance.
    """
    return _convert_named_normalized(
        entry,
        image_width=image_width,
        image_height=image_height,
        upload_width=None,
        upload_height=None,
    )


def _convert_named_0_1000(
    entry: dict,
    image_width: int,
    image_height: int,
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> Optional[List[float]]:
    box = _read_named_box(entry)
    if box is None:
        return None
    x_min, y_min, x_max, y_max = (
        _clamp(value, NORMALIZED_0_1000_SCALE) for value in box
    )
    scale_x = image_width / NORMALIZED_0_1000_SCALE
    scale_y = image_height / NORMALIZED_0_1000_SCALE
    return [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]


def _convert_named_normalized(
    entry: dict,
    image_width: int,
    image_height: int,
    upload_width: Optional[int],
    upload_height: Optional[int],
) -> Optional[List[float]]:
    box = _read_named_box(entry)
    if box is None:
        return None
    x_min, y_min, x_max, y_max = (_clamp(value, 1.0) for value in box)
    return [
        x_min * image_width,
        y_min * image_height,
        x_max * image_width,
        y_max * image_height,
    ]


BoxConverter = Callable[
    [dict, int, int, Optional[int], Optional[int]], Optional[List[float]]
]


@dataclass(frozen=True)
class DetectionBoxFormat:
    """A box coordinate contract: how to ask for it and how to decode it.

    Attributes:
        name: Stable identifier, used as the block-facing format value.
        prompt_template: Object-detection prompt wording for this contract,
            with a ``{class_list}`` placeholder and, when
            ``requires_upload_dimensions`` is set, ``{width}``/``{height}``.
            ``None`` for a contract whose prompt is rendered elsewhere - see
            ``named_normalized`` below.
        requires_upload_dimensions: Whether decoding needs the dimensions
            of the image as uploaded (true for absolute-pixel contracts).
        converter: Callable implementing the coordinate conversion.
    """

    name: str
    prompt_template: Optional[str]
    requires_upload_dimensions: bool
    converter: BoxConverter

    def to_pixel_xyxy(
        self,
        entry: dict,
        image_width: int,
        image_height: int,
        upload_width: Optional[int] = None,
        upload_height: Optional[int] = None,
    ) -> Optional[List[float]]:
        """Convert one detection entry into original-image pixel ``xyxy``.

        Args:
            entry: Raw detection entry from the model's JSON answer.
            image_width: Original image width in pixels.
            image_height: Original image height in pixels.
            upload_width: Width of the uploaded image, for absolute formats.
            upload_height: Height of the uploaded image, for absolute formats.

        Returns:
            ``[x_min, y_min, x_max, y_max]`` in original-image pixels, or
            ``None`` when the entry carries no well-formed box.

        Raises:
            ValueError: If the format needs upload dimensions and none were
                given.
        """
        return self.converter(
            entry,
            image_width,
            image_height,
            upload_width,
            upload_height,
        )


DETECTION_BOX_FORMATS: Dict[str, DetectionBoxFormat] = {
    "xyxy_absolute": DetectionBoxFormat(
        name="xyxy_absolute",
        prompt_template=XYXY_ABSOLUTE_PROMPT_TEMPLATE,
        requires_upload_dimensions=True,
        converter=_convert_xyxy_absolute,
    ),
    "xyxy_0_1000": DetectionBoxFormat(
        name="xyxy_0_1000",
        prompt_template=XYXY_0_1000_PROMPT_TEMPLATE,
        requires_upload_dimensions=False,
        converter=_convert_xyxy_0_1000,
    ),
    "yxyx_0_1000": DetectionBoxFormat(
        name="yxyx_0_1000",
        prompt_template=YXYX_0_1000_PROMPT_TEMPLATE,
        requires_upload_dimensions=False,
        converter=_convert_yxyx_0_1000,
    ),
    "xyxy_percent": DetectionBoxFormat(
        name="xyxy_percent",
        prompt_template=XYXY_PERCENT_PROMPT_TEMPLATE,
        requires_upload_dimensions=False,
        converter=_convert_xyxy_percent,
    ),
    "named_0_1000": DetectionBoxFormat(
        name="named_0_1000",
        prompt_template=NAMED_0_1000_PROMPT_TEMPLATE,
        requires_upload_dimensions=False,
        converter=_convert_named_0_1000,
    ),
    "named_normalized": DetectionBoxFormat(
        name="named_normalized",
        # No template here: the live wording for this contract is the legacy
        # OpenRouter/OpenAI system message built by
        # `common/openrouter.py::_prepare_object_detection_prompt`, which the
        # blocks on this format call instead of `build_object_detection_prompt`.
        prompt_template=None,
        requires_upload_dimensions=False,
        converter=_convert_named_normalized,
    ),
}

BoxFormatName = Literal[
    "xyxy_absolute",
    "xyxy_0_1000",
    "yxyx_0_1000",
    "xyxy_percent",
    "named_0_1000",
    "named_normalized",
]


def get_detection_box_format(box_format: str) -> DetectionBoxFormat:
    """Look up a registered box format.

    Args:
        box_format: Format name.

    Returns:
        The registered :class:`DetectionBoxFormat`.

    Raises:
        ValueError: If the name is not registered.
    """
    if box_format not in DETECTION_BOX_FORMATS:
        raise ValueError(
            f"Unknown detection box format: {box_format}. "
            f"Supported formats: {sorted(DETECTION_BOX_FORMATS)}."
        )
    return DETECTION_BOX_FORMATS[box_format]


def build_object_detection_prompt(
    box_format: str,
    classes: List[str],
    upload_width: Optional[int] = None,
    upload_height: Optional[int] = None,
) -> str:
    """Render the object-detection prompt for a box format.

    Args:
        box_format: Registered box format name.
        classes: Class names the model may predict.
        upload_width: Width of the image as uploaded, required for formats
            asking for absolute pixel coordinates.
        upload_height: Height of the image as uploaded, same requirement.

    Returns:
        The rendered prompt text.

    Raises:
        ValueError: If the format is unknown, carries no prompt template, or
            requires upload dimensions that were not provided.
    """
    detection_format = get_detection_box_format(box_format)
    if detection_format.prompt_template is None:
        raise ValueError(
            f"Box format {box_format} carries no prompt template - its prompt "
            f"is rendered elsewhere, see `DETECTION_BOX_FORMATS`."
        )
    if detection_format.requires_upload_dimensions:
        _require_upload_dimensions(upload_width, upload_height)
    return detection_format.prompt_template.format(
        class_list=", ".join(classes),
        width=upload_width,
        height=upload_height,
    )


def extract_detection_entries(parsed: Any) -> List[dict]:
    """Pull the list of detection entries out of a parsed JSON payload.

    Accepts a bare list of entries, a ``{"detections": [...]}`` wrapper, or
    a single bare entry object. Non-dict members are skipped, but a
    non-empty list holding no entry objects at all (for example a bare
    coordinate list lifted out of the model's reasoning text) is rejected,
    so it surfaces as ``error_status`` instead of "no objects found".

    Args:
        parsed: JSON payload extracted from the VLM output.

    Returns:
        List of raw detection entry dicts.

    Raises:
        ValueError: If the payload matches none of the accepted shapes.
    """
    if isinstance(parsed, list):
        return _entries_from_list(parsed)
    if isinstance(parsed, dict):
        detections = parsed.get(DETECTIONS_WRAPPER_KEY)
        if isinstance(detections, list):
            return _entries_from_list(detections)
        if _looks_like_detection_entry(parsed):
            return [parsed]
    raise ValueError("Unexpected object detection response format")


def _entries_from_list(items: list) -> List[dict]:
    entries = [entry for entry in items if isinstance(entry, dict)]
    if items and not entries:
        raise ValueError(
            "Object detection response is a list without any detection entries"
        )
    return entries


def _looks_like_detection_entry(entry: dict) -> bool:
    if all(field in entry for field in NAMED_BOX_FIELDS):
        return True
    return any(key in entry for key in BOX_2D_KEYS)


def get_detection_class_name(entry: dict) -> str:
    """Resolve the class label of a detection entry.

    Args:
        entry: Raw detection entry.

    Returns:
        The first non-empty label found under the accepted key aliases, or
        ``"unknown"``.
    """
    for key in LABEL_KEYS:
        value = entry.get(key)
        if value is None:
            continue
        label = str(value)
        if label:
            return label
    return "unknown"


def get_detection_confidence(entry: dict) -> float:
    """Resolve the confidence of a detection entry.

    VLMs are not asked for calibrated confidences by every format, so a
    missing or unusable value defaults to ``1.0``.

    Args:
        entry: Raw detection entry.

    Returns:
        Confidence clamped into ``[0.0, 1.0]``.
    """
    value = _read_number(entry.get("confidence"))
    if value is None or not math.isfinite(value):
        return 1.0
    return scale_confidence(value)
