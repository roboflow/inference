"""Decode vision-language model answers into native predictions.

Port of V1 ``core_steps/common/vlm_decoding`` and ``common/vlm_json``. Every
VLM block and the VLM parser blocks share these decoders, so one answer shape
is read one way everywhere. Detection decoding is keyed by the box coordinate
format the prompt asked for, not by the model vendor::

    answer (str, or an already decoded dict/list)
      └─ extract_json     first ```json fence, any fence, whole text,
                          value sequence, outermost [...] / {...}
           ├─ classification   {"class_name", "confidence"}  single label
           │                   {"predicted_classes": [...]}  multi-label
           ├─ object-detection entries -> box format -> image pixels
           └─ instance-segmentation polygons -> COCO RLE, no dense mask

Each decoder returns ``(error_status, predictions)``. ``error_status`` is
``True`` and ``predictions`` ``None`` when the answer cannot be read; a model
answer never raises. A wrong call does raise ``ValueError``: unknown box
format, invalid or duplicate ``classes``, or missing upload size.

Predictions are native carriers on the image's device:

=========================  ===============================================
task                       predictions
=========================  ===============================================
classification             ``ClassificationPrediction`` or
                           ``MultiLabelClassificationPrediction``
object-detection           ``Detections`` with a row identity
instance-segmentation      ``InstanceDetections`` with ``InstancesRLEMasks``
=========================  ===============================================

``classes[i]`` has class id ``i``. A label outside ``classes`` keeps its own
name: detections give it class id ``-1``; classifications append it after
``classes``, because their confidence vector is indexed by class id (as V1
tensor mode). Every detection row stores its label under ``class`` in
``bboxes_metadata``; ``image_metadata`` adds the image's frame, ``class_names``
and ``prediction_type``. A classification records the image's parent row.

Differences from V1:

* A decoded ``dict`` or ``list`` answer is accepted as is, besides text.
* A wrong call raises instead of returning ``error_status``.
* Detections carry a row identity named by ``inference_id`` instead of
  per-row ``detection_id``, ``inference_id`` and V1 serializer style keys.
"""

import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    with_classification_parent,
    with_row_identity,
)
from roboflow_workflows.v2.engine.declaration import Output

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

logger = logging.getLogger(__name__)

Decoded = Tuple[bool, Any]
SizeWH = Tuple[int, int]

CLASSIFICATION_TASKS = ("classification", "multi-label-classification")
DETECTION_TASKS = ("object-detection",)
SEGMENTATION_TASKS = ("instance-segmentation",)

CLASSIFICATION_PREDICTION_TYPE = "classification"
DETECTION_PREDICTION_TYPE = "object-detection"
SEGMENTATION_PREDICTION_TYPE = "instance-segmentation"
CLASS_NAME_KEY = "class"
UNKNOWN_LABEL = "unknown"

# Prompt templates, one per box format. ``{class_list}`` is always filled;
# ``{width}``/``{height}`` are the uploaded image's size, for absolute formats.
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
INSTANCE_SEGMENTATION_PROMPT_TEMPLATE = (
    "Segment all objects in this image. "
    'Output JSON with the key "segmentations" holding a list where each entry '
    'contains the outline polygon in the key "polygon" and the text label in '
    'the key "label". '
    'The "polygon" value must be a flat list [x1, y1, x2, y2, ...] of the '
    "polygon vertices in order, in absolute pixel coordinates of the "
    "{width}x{height} pixel image. "
    "Use as many vertices as you need to trace the object outline accurately. "
    "Only use these labels: {class_list}"
)

# Entry vocabulary: the prompted key first, then names models drift to.
BOX_2D_KEYS = ("box_2d", "bbox_2d")
NAMED_BOX_FIELDS = ("x_min", "y_min", "x_max", "y_max")
LABEL_KEYS = ("label", "class_name", "class", "description")
POLYGON_KEYS = ("polygon", "segmentation", "points", "mask")
CLASS_ENTRY_LABEL_KEYS = ("class", "class_name", "label")
DETECTIONS_WRAPPER_KEY = "detections"
SEGMENTATIONS_WRAPPER_KEY = "segmentations"

NORMALIZED_0_1000_SCALE = 1000.0
PERCENT_SCALE = 100.0
MINIMUM_POLYGON_VERTICES = 3
MINIMUM_POLYGON_AREA = 1.0
# V1 WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES default; bounds RLE work.
MAX_POLYGON_VERTICES = 500

# Upload sizes: VLM blocks resize before upload, and decoders map the answer's
# absolute pixels back with the same arithmetic.
DETECTION_MAX_EDGE_PIXELS = 2048
ANTHROPIC_DETECTION_MAX_EDGE_PIXELS = 2576
ANTHROPIC_DETECTION_MAX_IMAGE_TOKENS = 4784
ANTHROPIC_IMAGE_TILE_PIXELS = 28

_JSON_FENCE = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)```")
_FENCE_LINE = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", flags=re.MULTILINE)
_SEQUENCE_SEPARATOR = re.compile(r"^[\s,]*$")
_SEQUENCE_TAIL = re.compile(r"^[\s,]*\]?[\s,]*$")

BoxFormatName = Literal[
    "xyxy_absolute",
    "xyxy_0_1000",
    "yxyx_0_1000",
    "xyxy_percent",
    "named_0_1000",
    "named_normalized",
]
BoxConverter = Callable[[Mapping, SizeWH, Optional[SizeWH]], Optional[List[float]]]


@dataclass(frozen=True)
class DetectionBoxFormat:
    """A box coordinate contract: how to ask for it and how to read it.

    Args:
        name: Format name, the value blocks expose.
        prompt_template: Detection prompt with ``{class_list}`` and, for
            absolute formats, ``{width}``/``{height}``. ``None`` when the
            prompt is the calling block's own wording.
        requires_upload_size: Whether reading boxes needs the uploaded size.
        converter: ``(entry, image_size_wh, upload_size_wh)`` to
            ``[x_min, y_min, x_max, y_max]`` image pixels, or ``None`` for an
            entry without a well-formed box.
        reads_confidence: Whether an entry's ``confidence`` is read. Formats
            whose V1 parsers forced ``1.0`` keep doing so.
    """

    name: str
    prompt_template: Optional[str]
    requires_upload_size: bool
    converter: BoxConverter
    reads_confidence: bool = True


def extract_json(
    raw_output: Any, *, salvage_truncated_detections: bool = False
) -> Tuple[bool, Any]:
    """Read the JSON payload of a model answer leniently.

    Text is tried in order: the first ```json fence, the first fence of any
    language, the whole text, the text without fence lines, a sequence of
    top-level values (JSON Lines, ``{...}, {...}`` or ``[...]\\n[...]``, one
    stray trailing ``]`` allowed), then the outermost ``[...]`` or ``{...}``.
    With ``salvage_truncated_detections``, complete named-box objects of an
    answer cut off at the token limit are kept when all of that fails.

    Args:
        raw_output: Answer text, or an already decoded dict or list, which is
            returned as is.
        salvage_truncated_detections: Whether to salvage complete
            ``{"x_min", "y_min", "x_max", "y_max", ...}`` objects. Only for
            detection; salvaged boxes are not a classification.

    Returns:
        ``(error_status, payload)``; ``payload`` is a dict or list, or
        ``None`` when ``error_status`` is ``True``.
    """
    if isinstance(raw_output, (Mapping, list)):
        return False, raw_output
    if not isinstance(raw_output, str):
        return True, None

    try:
        payload = _parse_json_text(raw_output)
        if payload is None and salvage_truncated_detections:
            payload = _named_box_objects(raw_output) or None
    except (RecursionError, ValueError) as error:
        # json is not total: deep nesting and huge integers raise.
        logger.warning("Could not parse JSON of a VLM answer: %r", error)
        return True, None

    if payload is None:
        logger.warning("Could not find JSON in a VLM answer")
        return True, None

    return False, payload


def decode_classification(
    raw_output: Any, *, image: ImageData, classes: Sequence[str]
) -> Decoded:
    """Decode a single-label or multi-label classification answer.

    ``{"class_name": ..., "confidence": ...}`` is a single-label answer;
    ``{"predicted_classes": [{"class": ..., "confidence": ...}, ...]}`` is a
    multi-label answer, as is a bare list of such entries (``class_name`` and
    ``label`` accepted, ``confidence`` defaults to 1). A repeated label keeps
    its highest confidence. Confidences are clamped to ``[0, 1]``.

    Args:
        raw_output: Answer text, or an already decoded dict or list.
        image: The image the model classified.
        classes: Class names; ``classes[i]`` has class id ``i``.

    Returns:
        ``(error_status, predictions)``: a ``ClassificationPrediction`` of one
        image with the full confidence vector, or a
        ``MultiLabelClassificationPrediction``; ``None`` when unreadable.

    Raises:
        ValueError: When ``classes`` is not a list of unique strings.
    """
    classes = checked_classes(classes)
    error_status, payload = extract_json(raw_output)
    if error_status:
        return True, None

    try:
        payload = _classification_payload(payload)
        if "class_name" in payload and "confidence" in payload:
            prediction = _single_label(payload, image=image, classes=classes)
        elif "predicted_classes" in payload:
            prediction = _multi_label(payload, image=image, classes=classes)
        else:
            raise ValueError(
                "answer has neither class_name and confidence nor predicted_classes"
            )
    except (KeyError, TypeError, ValueError) as error:
        logger.warning("Could not decode a VLM classification answer: %s", error)
        return True, None

    prediction = with_classification_parent(prediction, image.parent_row)

    return False, prediction


def decode_object_detections(
    raw_output: Any,
    *,
    image: ImageData,
    classes: Sequence[str],
    box_format: str,
    inference_id: str,
    upload_size_wh: Optional[SizeWH] = None,
) -> Decoded:
    """Decode a detection answer in the given box format into image pixels.

    Entries come from a bare list, a ``{"detections": [...]}`` wrapper or one
    bare entry. An entry without a well-formed box is skipped; an answer in
    which no entry matches the format is unreadable, not empty. Swapped
    corners are reordered and boxes are rounded to whole pixels.

    Args:
        raw_output: Answer text, or an already decoded dict or list.
        image: The image the model saw, at its original size.
        classes: Class names; a label outside them gets class id ``-1``.
        box_format: A ``DETECTION_BOX_FORMATS`` name.
        inference_id: Namespace of the result's row identity.
        upload_size_wh: ``(width, height)`` of the image as uploaded; required
            by absolute formats.

    Returns:
        ``(error_status, predictions)``: ``Detections`` on the image's device,
        possibly empty, or ``None`` when unreadable.

    Raises:
        ValueError: On an unknown format, invalid ``classes`` or a missing
            ``upload_size_wh`` the format needs.
    """
    detection_format = detection_box_format(box_format)
    classes = checked_classes(classes)
    if detection_format.requires_upload_size:
        upload_size_wh = checked_upload_size(upload_size_wh, box_format=box_format)

    error_status, payload = extract_json(raw_output, salvage_truncated_detections=True)
    if error_status:
        return True, None

    try:
        rows = _detection_rows(
            _detection_entries(payload),
            detection_format=detection_format,
            image_size_wh=_size_wh(image),
            upload_size_wh=upload_size_wh,
        )
    except ValueError as error:
        logger.warning(
            "Could not decode a VLM %s detection answer: %s", box_format, error
        )
        return True, None

    detections = _detections(rows, image=image, classes=classes)
    identified = with_row_identity(detections, namespace=inference_id)

    return False, identified


def decode_instance_segmentations(
    raw_output: Any,
    *,
    image: ImageData,
    classes: Sequence[str],
    inference_id: str,
    upload_size_wh: SizeWH,
    max_polygon_vertices: int = MAX_POLYGON_VERTICES,
) -> Decoded:
    """Decode outline polygons in uploaded-image pixels into RLE instances.

    Entries come from a list, a ``{"segmentations": [...]}`` or
    ``{"detections": [...]}`` wrapper, or one bare entry. A polygon is a flat
    ``[x1, y1, ...]`` list, ``[[x, y], ...]`` or ``[{"x", "y"}, ...]``. It is
    clipped to the upload, scaled to the image and rounded, then encoded to
    COCO RLE; no dense mask is allocated. Polygons that are malformed, longer
    than ``max_polygon_vertices``, or enclose no area are skipped; an answer
    whose entries are all skipped is unreadable. Confidence is 1.

    Args:
        raw_output: Answer text, or an already decoded dict or list.
        image: The image the model saw, at its original size.
        classes: Class names; a label outside them gets class id ``-1``.
        inference_id: Namespace of the result's row identity.
        upload_size_wh: ``(width, height)`` of the image as uploaded.
        max_polygon_vertices: Longest accepted polygon.

    Returns:
        ``(error_status, predictions)``: ``InstanceDetections`` with
        ``InstancesRLEMasks`` on the image's device, or ``None``.

    Raises:
        ValueError: On invalid ``classes`` or ``upload_size_wh``.
    """
    classes = checked_classes(classes)
    upload_size_wh = checked_upload_size(
        upload_size_wh, box_format="instance-segmentation"
    )
    error_status, payload = extract_json(raw_output)
    if error_status:
        return True, None

    try:
        rows, masks = _segmentation_rows(
            _segmentation_entries(payload),
            image_size_wh=_size_wh(image),
            upload_size_wh=upload_size_wh,
            max_polygon_vertices=max_polygon_vertices,
        )
    except ValueError as error:
        logger.warning("Could not decode a VLM segmentation answer: %s", error)
        return True, None

    detections = _detections(
        rows,
        image=image,
        classes=classes,
        masks=masks,
        prediction_type=SEGMENTATION_PREDICTION_TYPE,
    )
    identified = with_row_identity(detections, namespace=inference_id)

    return False, identified


def decode_vlm_output(
    task_type: str,
    raw_output: Any,
    *,
    image: ImageData,
    classes: Optional[Sequence[str]],
    inference_id: str,
    box_format: Optional[str] = None,
    upload_size_wh: Optional[SizeWH] = None,
) -> Decoded:
    """Decode an answer according to the task the VLM block ran.

    Args:
        task_type: The block's task. Tasks outside ``CLASSIFICATION_TASKS``,
            ``DETECTION_TASKS`` and ``SEGMENTATION_TASKS`` decode nothing.
        raw_output: Answer text, or an already decoded dict or list.
        image: The image the model saw, at its original size.
        classes: Class names; required by every decoding task.
        inference_id: Namespace of the row identity of detections.
        box_format: Box format of a detection task.
        upload_size_wh: ``(width, height)`` of the upload, for absolute box
            formats and segmentation.

    Returns:
        ``(error_status, predictions)``; ``(False, None)`` for a task that
        decodes nothing.

    Raises:
        ValueError: When the task needs ``classes``, ``box_format`` or
            ``upload_size_wh`` and it is missing or invalid.
    """
    decodes = task_type in (CLASSIFICATION_TASKS + DETECTION_TASKS + SEGMENTATION_TASKS)
    if not decodes:
        return False, None
    if classes is None:
        raise ValueError(f"Decoding a VLM {task_type} answer needs classes")

    if task_type in CLASSIFICATION_TASKS:
        decoded = decode_classification(raw_output, image=image, classes=classes)
    elif task_type in SEGMENTATION_TASKS:
        decoded = decode_instance_segmentations(
            raw_output,
            image=image,
            classes=classes,
            inference_id=inference_id,
            upload_size_wh=upload_size_wh,
        )
    else:
        if box_format is None:
            raise ValueError(f"Decoding a VLM {task_type} answer needs box_format")
        decoded = decode_object_detections(
            raw_output,
            image=image,
            classes=classes,
            box_format=box_format,
            inference_id=inference_id,
            upload_size_wh=upload_size_wh,
        )

    return decoded


def vlm_prediction_outputs(
    task_type: str, *, source: str = "image"
) -> Dict[str, Output]:
    """Declare the decoded outputs of a VLM block for its task.

    Args:
        task_type: The block's task.
        source: The block's image field, the outputs' source.

    Returns:
        ``predictions`` typed by the task, ``error_status`` and
        ``inference_id``. A task that decodes nothing keeps V1's detection and
        classification union for ``predictions``, which is then always
        ``None``.
    """
    if task_type in DETECTION_TASKS:
        kinds = (OBJECT_DETECTION_PREDICTION_KIND,)
    elif task_type in SEGMENTATION_TASKS:
        # RLE first, as decoded; dense-kind consumers still connect.
        kinds = (
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        )
    elif task_type in CLASSIFICATION_TASKS:
        kinds = (CLASSIFICATION_PREDICTION_KIND,)
    else:
        kinds = (OBJECT_DETECTION_PREDICTION_KIND, CLASSIFICATION_PREDICTION_KIND)

    outputs = {
        "predictions": Output(
            *kinds,
            source=source,
            description="Decoded prediction; None when the answer is unreadable.",
        ),
        "error_status": Output(
            BOOLEAN_KIND,
            source=source,
            description="True when the answer could not be decoded.",
        ),
        "inference_id": Output(
            STRING_KIND,
            source=source,
            description="Fresh id of this result; namespace of its row identity.",
        ),
    }

    return outputs


def build_object_detection_prompt(
    box_format: str,
    *,
    classes: Sequence[str],
    upload_size_wh: Optional[SizeWH] = None,
) -> str:
    """Render the detection prompt of a box format.

    Args:
        box_format: A ``DETECTION_BOX_FORMATS`` name with a prompt template.
        classes: Class names the model may answer with.
        upload_size_wh: ``(width, height)`` of the upload; required by
            absolute formats.

    Returns:
        The prompt text.

    Raises:
        ValueError: On an unknown format, a format without a template,
            invalid ``classes`` or a missing ``upload_size_wh``.
    """
    detection_format = detection_box_format(box_format)
    if detection_format.prompt_template is None:
        raise ValueError(
            f"Box format {box_format!r} has no prompt template; the block that "
            "uses it writes its own prompt"
        )
    classes = checked_classes(classes)
    width, height = None, None
    if detection_format.requires_upload_size:
        width, height = checked_upload_size(upload_size_wh, box_format=box_format)

    prompt = detection_format.prompt_template.format(
        class_list=", ".join(classes), width=width, height=height
    )

    return prompt


def build_instance_segmentation_prompt(
    *, classes: Sequence[str], upload_size_wh: SizeWH
) -> str:
    """Render the instance-segmentation prompt.

    Args:
        classes: Class names the model may answer with.
        upload_size_wh: ``(width, height)`` of the upload.

    Returns:
        The prompt text.

    Raises:
        ValueError: On invalid ``classes`` or ``upload_size_wh``.
    """
    classes = checked_classes(classes)
    width, height = checked_upload_size(
        upload_size_wh, box_format="instance-segmentation"
    )
    prompt = INSTANCE_SEGMENTATION_PROMPT_TEMPLATE.format(
        class_list=", ".join(classes), width=width, height=height
    )

    return prompt


def detection_box_format(box_format: str) -> DetectionBoxFormat:
    """Return a registered box format.

    Args:
        box_format: Format name.

    Returns:
        The ``DetectionBoxFormat``.

    Raises:
        ValueError: When the name is not in ``DETECTION_BOX_FORMATS``.
    """
    if box_format not in DETECTION_BOX_FORMATS:
        raise ValueError(
            f"Unknown detection box format {box_format!r}; use one of "
            f"{sorted(DETECTION_BOX_FORMATS)}"
        )

    return DETECTION_BOX_FORMATS[box_format]


def checked_classes(classes: Any) -> List[str]:
    """Check a class list: unique strings; ``classes[i]`` has id ``i``.

    Args:
        classes: List or tuple of class names.

    Returns:
        The names as a new list.

    Raises:
        ValueError: On another container, a non-string or a repeated name.
    """
    if not isinstance(classes, (list, tuple)):
        raise ValueError(
            f"classes must be a list of strings, got {type(classes).__name__}"
        )
    not_strings = [name for name in classes if not isinstance(name, str)]
    if not_strings:
        raise ValueError(f"classes must be strings, got {not_strings!r}")
    repeated = sorted({name for name in classes if classes.count(name) > 1})
    if repeated:
        raise ValueError(f"classes repeat {repeated}; each name needs one class id")

    checked = list(classes)

    return checked


def checked_upload_size(upload_size_wh: Any, *, box_format: str) -> SizeWH:
    """Check an upload size: two positive integers, width first.

    Args:
        upload_size_wh: ``(width, height)`` of the uploaded image.
        box_format: What needs the size, for the error message.

    Returns:
        ``(width, height)`` as ints.

    Raises:
        ValueError: When the size is missing or not two positive integers.
    """
    valid = (
        isinstance(upload_size_wh, (tuple, list))
        and len(upload_size_wh) == 2
        and all(
            isinstance(side, int) and not isinstance(side, bool) and side > 0
            for side in upload_size_wh
        )
    )
    if not valid:
        raise ValueError(
            f"{box_format} needs upload_size_wh, the (width, height) of the "
            f"uploaded image as positive integers, got {upload_size_wh!r}"
        )

    size_wh = (int(upload_size_wh[0]), int(upload_size_wh[1]))

    return size_wh


def max_edge_upload_size(size_wh: SizeWH, *, max_edge: int) -> SizeWH:
    """Size of an image downscaled so its longest edge is at most ``max_edge``.

    Never upscales; keeps the aspect ratio. V1 OpenAI blocks upload at
    ``DETECTION_MAX_EDGE_PIXELS``.

    Args:
        size_wh: ``(width, height)`` of the original image.
        max_edge: Longest allowed edge.

    Returns:
        ``(width, height)`` of the upload.
    """
    width, height = size_wh
    if max(width, height) <= max_edge:
        return (width, height)

    if width >= height:
        scaled = (max_edge, max(round(height * max_edge / width), 1))
    else:
        scaled = (max(round(width * max_edge / height), 1), max_edge)

    return scaled


def anthropic_upload_size(size_wh: SizeWH) -> SizeWH:
    """Size Claude resizes an image to before reading it.

    Follows Anthropic's vision coordinate documentation as V1 does: edges
    padded to 28-pixel tiles stay within 2576 pixels and the tile count
    within 4784. Never upscales.

    Args:
        size_wh: ``(width, height)`` of the original image.

    Returns:
        ``(width, height)`` Claude reads, the size its pixel answers use.
    """
    width, height = size_wh
    if _fits_anthropic(width, height):
        return (width, height)
    if height > width:
        resized_height, resized_width = anthropic_upload_size((height, width))
        return (resized_width, resized_height)

    # Widest width whose aspect-kept height still fits.
    aspect_ratio = width / height
    low, high = 1, width
    while low + 1 < high:
        middle = (low + high) // 2
        if _fits_anthropic(middle, max(round(middle / aspect_ratio), 1)):
            low = middle
        else:
            high = middle

    resized = (low, max(round(low / aspect_ratio), 1))

    return resized


def scale_confidence(value: Any) -> float:
    """Clamp a confidence into ``[0, 1]``.

    Args:
        value: Number or numeric string.

    Returns:
        The clamped float.

    Raises:
        ValueError: On a non-finite value; ``json`` accepts ``NaN``.
        TypeError: On a value ``float`` cannot read.
    """
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError(f"confidence is not a finite number: {value!r}")

    clamped = min(max(confidence, 0.0), 1.0)

    return clamped


def entry_label(entry: Mapping) -> str:
    """Return an entry's first non-empty label, else ``"unknown"``.

    Args:
        entry: One detection or segmentation entry.

    Returns:
        The text of the first of ``LABEL_KEYS`` that is present and not empty.
    """
    for key in LABEL_KEYS:
        value = entry.get(key)
        if value is not None and str(value):
            return str(value)

    return UNKNOWN_LABEL


# --- JSON text ---------------------------------------------------------------


def _parse_json_text(raw: str) -> Optional[Any]:
    candidates = []
    json_fences = _JSON_FENCE.findall(raw)
    if json_fences:
        candidates.append(json_fences[0])
    any_fences = _ANY_FENCE.findall(raw)
    if any_fences:
        candidates.append(any_fences[0])
    candidates.extend([raw, _FENCE_LINE.sub("", raw)])
    for candidate in dict.fromkeys(candidates):
        payload = _json_container(candidate)
        if payload is not None:
            return payload

    stripped = _FENCE_LINE.sub("", raw)
    sequence = _value_sequence(stripped)
    if sequence is not None:
        return sequence

    for opening, closing in (("[", "]"), ("{", "}")):
        start, stop = stripped.find(opening), stripped.rfind(closing)
        if start != -1 and stop > start:
            payload = _json_container(stripped[start : stop + 1])
            if payload is not None:
                return payload

    return None


def _json_container(text: str) -> Optional[Any]:
    try:
        parsed = json.loads(text)
    except ValueError:
        # Includes JSONDecodeError and the int-to-str digit limit.
        return None

    container = parsed if isinstance(parsed, (dict, list)) else None

    return container


def _value_sequence(text: str) -> Optional[list]:
    # Succeeds only when the whole text is objects or arrays separated by
    # whitespace or commas, so garbage around fragments is never a success.
    decoder = json.JSONDecoder()
    entries: list = []
    found = 0
    index = 0
    while index < len(text):
        starts = [p for p in (text.find("{", index), text.find("[", index)) if p != -1]
        if not starts:
            break
        start = min(starts)
        if not _SEQUENCE_SEPARATOR.match(text[index:start]):
            return None
        try:
            value, index = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict):
            entries.append(value)
        elif isinstance(value, list):
            entries.extend(value)
        else:
            return None
        found += 1

    if not found or not _SEQUENCE_TAIL.match(text[index:]):
        return None

    return entries


def _named_box_objects(text: str) -> List[dict]:
    decoder = json.JSONDecoder()
    entries = []
    index = 0
    while (start := text.find("{", index)) != -1:
        try:
            entry, index = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(entry, dict) and all(key in entry for key in NAMED_BOX_FIELDS):
            entries.append(entry)

    return entries


# --- classification ----------------------------------------------------------


def _classification_payload(payload: Any) -> Mapping:
    if isinstance(payload, Mapping):
        return payload
    if not payload:
        raise ValueError("answer is an empty list")

    entries = []
    for entry in payload:
        labels = [
            entry.get(key)
            for key in CLASS_ENTRY_LABEL_KEYS
            if isinstance(entry, Mapping) and entry.get(key) is not None
        ]
        if not labels:
            raise ValueError(f"list entry {entry!r} has no class label")
        entries.append({"class": labels[0], "confidence": entry.get("confidence", 1.0)})

    wrapped = {"predicted_classes": entries}

    return wrapped


def _single_label(
    payload: Mapping, *, image: ImageData, classes: List[str]
) -> ClassificationPrediction:
    top = str(payload["class_name"])
    names = classes if top in classes else classes + [top]
    confidence = [0.0] * len(names)
    confidence[names.index(top)] = scale_confidence(payload["confidence"])

    prediction = ClassificationPrediction(
        class_id=torch.tensor(
            [names.index(top)], dtype=torch.int64, device=image.device
        ),
        confidence=torch.tensor([confidence], dtype=torch.float32, device=image.device),
        images_metadata=[_classification_metadata(image, names=names)],
    )

    return prediction


def _multi_label(
    payload: Mapping, *, image: ImageData, classes: List[str]
) -> MultiLabelClassificationPrediction:
    predicted: Dict[str, float] = {}
    for entry in payload["predicted_classes"]:
        label = str(entry["class"])
        confidence = scale_confidence(entry["confidence"])
        predicted[label] = max(confidence, predicted.get(label, confidence))

    names = classes + [label for label in predicted if label not in classes]
    confidence = [predicted.get(name, 0.0) for name in names]
    class_ids = [names.index(label) for label in predicted]

    prediction = MultiLabelClassificationPrediction(
        class_ids=torch.tensor(class_ids, dtype=torch.int64, device=image.device),
        confidence=torch.tensor(confidence, dtype=torch.float32, device=image.device),
        image_metadata=_classification_metadata(image, names=names),
    )

    return prediction


def _classification_metadata(image: ImageData, *, names: List[str]) -> Dict[str, Any]:
    metadata = {
        **image.prediction_metadata(),
        "class_names": dict(enumerate(names)),
        "prediction_type": CLASSIFICATION_PREDICTION_TYPE,
    }

    return metadata


# --- detection ---------------------------------------------------------------


def _detection_entries(payload: Any) -> List[Mapping]:
    if isinstance(payload, Mapping):
        wrapped = payload.get(DETECTIONS_WRAPPER_KEY)
        if isinstance(wrapped, list):
            return _mapping_entries(wrapped)
        named = all(key in payload for key in NAMED_BOX_FIELDS)
        if named or any(key in payload for key in BOX_2D_KEYS):
            return [payload]
        raise ValueError("answer is neither a list of detections nor one detection")

    entries = _mapping_entries(payload)

    return entries


def _mapping_entries(items: list) -> List[Mapping]:
    entries = [entry for entry in items if isinstance(entry, Mapping)]
    if items and not entries:
        raise ValueError("answer is a list without entry objects")

    return entries


def _detection_rows(
    entries: List[Mapping],
    *,
    detection_format: DetectionBoxFormat,
    image_size_wh: SizeWH,
    upload_size_wh: Optional[SizeWH],
) -> List[Tuple[List[float], str, float]]:
    rows = []
    for entry in entries:
        box = detection_format.converter(entry, image_size_wh, upload_size_wh)
        if box is None:
            logger.warning(
                "Skipping a VLM detection without a well-formed %s box: %r",
                detection_format.name,
                entry,
            )
            continue
        x_0, y_0, x_1, y_1 = box
        ordered = [min(x_0, x_1), min(y_0, y_1), max(x_0, x_1), max(y_0, y_1)]
        confidence = (
            _entry_confidence(entry) if detection_format.reads_confidence else 1.0
        )
        rows.append((ordered, entry_label(entry), confidence))

    if entries and not rows:
        # An empty result would read as "nothing found"; the model answered
        # in another coordinate contract.
        raise ValueError(
            f"none of {len(entries)} entries has a {detection_format.name} box"
        )

    return rows


def _entry_confidence(entry: Mapping) -> float:
    value = _read_number(entry.get("confidence"))
    confidence = 1.0 if value is None else scale_confidence(value)

    return confidence


def _detections(
    rows: List[Tuple[List[float], str, float]],
    *,
    image: ImageData,
    classes: List[str],
    masks: Optional[List[bytes]] = None,
    prediction_type: str = DETECTION_PREDICTION_TYPE,
) -> Any:
    device = image.device
    ids_by_name = {name: class_id for class_id, name in enumerate(classes)}
    boxes = torch.tensor([box for box, _, _ in rows], dtype=torch.float64)
    fields = {
        "xyxy": boxes.reshape(-1, 4).round().to(torch.float32).to(device),
        "class_id": torch.tensor(
            [ids_by_name.get(label, -1) for _, label, _ in rows],
            dtype=torch.int64,
            device=device,
        ),
        "confidence": torch.tensor(
            [confidence for _, _, confidence in rows],
            dtype=torch.float32,
            device=device,
        ),
        "image_metadata": {
            **image.prediction_metadata(),
            "class_names": dict(enumerate(classes)),
            "prediction_type": prediction_type,
        },
        "bboxes_metadata": [{CLASS_NAME_KEY: label} for _, label, _ in rows],
    }
    if masks is None:
        return Detections(**fields)

    height, width = image.size_hw
    rle = InstancesRLEMasks(image_size=(height, width), masks=masks)
    instances = InstanceDetections(mask=rle, **fields)

    return instances


# --- box formats -------------------------------------------------------------


def _read_number(value: Any) -> Optional[float]:
    # Numeric strings are read: models quote coordinates. Bools and NaN/inf
    # are not; json accepts NaN, which would survive clamping.
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    elif not isinstance(value, (int, float)):
        return None

    number = float(value)
    if not math.isfinite(number):
        return None

    return number


def _read_box_2d(entry: Mapping) -> Optional[List[float]]:
    for key in BOX_2D_KEYS:
        box = entry.get(key)
        if not isinstance(box, list) or len(box) != 4:
            continue
        values = [_read_number(value) for value in box]
        if all(value is not None for value in values):
            return values

    return None


def _read_named_box(entry: Mapping) -> Optional[List[float]]:
    values = [_read_number(entry.get(key)) for key in NAMED_BOX_FIELDS]
    box = None if any(value is None for value in values) else values

    return box


def _scaled(
    box: List[float], *, limit_wh: Tuple[float, float], scale_wh: Tuple[float, float]
) -> List[float]:
    # Clamp each coordinate to its nominal range, then scale to image pixels.
    limit_x, limit_y = limit_wh
    scale_x, scale_y = scale_wh
    x_min, y_min, x_max, y_max = box
    scaled = [
        min(max(x_min, 0.0), limit_x) * scale_x,
        min(max(y_min, 0.0), limit_y) * scale_y,
        min(max(x_max, 0.0), limit_x) * scale_x,
        min(max(y_max, 0.0), limit_y) * scale_y,
    ]

    return scaled


def _normalized(
    box: Optional[List[float]], *, size_wh: SizeWH, scale: float
) -> Optional[List[float]]:
    if box is None:
        return None

    width, height = size_wh
    scaled = _scaled(
        box, limit_wh=(scale, scale), scale_wh=(width / scale, height / scale)
    )

    return scaled


def _convert_xyxy_absolute(
    entry: Mapping, size_wh: SizeWH, upload_size_wh: Optional[SizeWH]
) -> Optional[List[float]]:
    box = _read_box_2d(entry)
    if box is None:
        # V1 Gemini, OpenAI and Claude parsers took 0-1 named fields when
        # box_2d was missing; older block versions prompted for them.
        return _convert_named_normalized(entry, size_wh, None)

    (width, height), (upload_width, upload_height) = size_wh, upload_size_wh
    scaled = _scaled(
        box,
        limit_wh=(upload_width, upload_height),
        scale_wh=(width / upload_width, height / upload_height),
    )

    return scaled


def _convert_xyxy_0_1000(
    entry: Mapping, size_wh: SizeWH, upload_size_wh: Optional[SizeWH]
) -> Optional[List[float]]:
    converted = _normalized(
        _read_box_2d(entry), size_wh=size_wh, scale=NORMALIZED_0_1000_SCALE
    )

    return converted


def _convert_yxyx_0_1000(
    entry: Mapping, size_wh: SizeWH, upload_size_wh: Optional[SizeWH]
) -> Optional[List[float]]:
    box = _read_box_2d(entry)
    if box is None:
        return _convert_named_normalized(entry, size_wh, None)

    y_min, x_min, y_max, x_max = box
    converted = _normalized(
        [x_min, y_min, x_max, y_max], size_wh=size_wh, scale=NORMALIZED_0_1000_SCALE
    )

    return converted


def _convert_xyxy_percent(
    entry: Mapping, size_wh: SizeWH, upload_size_wh: Optional[SizeWH]
) -> Optional[List[float]]:
    converted = _normalized(_read_box_2d(entry), size_wh=size_wh, scale=PERCENT_SCALE)

    return converted


def _convert_named_0_1000(
    entry: Mapping, size_wh: SizeWH, upload_size_wh: Optional[SizeWH]
) -> Optional[List[float]]:
    converted = _normalized(
        _read_named_box(entry), size_wh=size_wh, scale=NORMALIZED_0_1000_SCALE
    )

    return converted


def _convert_named_normalized(
    entry: Mapping, size_wh: SizeWH, upload_size_wh: Optional[SizeWH]
) -> Optional[List[float]]:
    converted = _normalized(_read_named_box(entry), size_wh=size_wh, scale=1.0)

    return converted


DETECTION_BOX_FORMATS: Dict[str, DetectionBoxFormat] = {
    "xyxy_absolute": DetectionBoxFormat(
        name="xyxy_absolute",
        prompt_template=XYXY_ABSOLUTE_PROMPT_TEMPLATE,
        requires_upload_size=True,
        converter=_convert_xyxy_absolute,
    ),
    "xyxy_0_1000": DetectionBoxFormat(
        name="xyxy_0_1000",
        prompt_template=XYXY_0_1000_PROMPT_TEMPLATE,
        requires_upload_size=False,
        converter=_convert_xyxy_0_1000,
        reads_confidence=False,
    ),
    "yxyx_0_1000": DetectionBoxFormat(
        name="yxyx_0_1000",
        prompt_template=YXYX_0_1000_PROMPT_TEMPLATE,
        requires_upload_size=False,
        converter=_convert_yxyx_0_1000,
    ),
    "xyxy_percent": DetectionBoxFormat(
        name="xyxy_percent",
        prompt_template=XYXY_PERCENT_PROMPT_TEMPLATE,
        requires_upload_size=False,
        converter=_convert_xyxy_percent,
    ),
    "named_0_1000": DetectionBoxFormat(
        name="named_0_1000",
        prompt_template=NAMED_0_1000_PROMPT_TEMPLATE,
        requires_upload_size=False,
        converter=_convert_named_0_1000,
        reads_confidence=False,
    ),
    # The prompt of this V1 OpenRouter/OpenAI contract is the block's own
    # system message, so it has no template here.
    "named_normalized": DetectionBoxFormat(
        name="named_normalized",
        prompt_template=None,
        requires_upload_size=False,
        converter=_convert_named_normalized,
    ),
}


# --- segmentation ------------------------------------------------------------


def _segmentation_entries(payload: Any) -> List[Mapping]:
    if isinstance(payload, Mapping):
        for wrapper in (SEGMENTATIONS_WRAPPER_KEY, DETECTIONS_WRAPPER_KEY):
            if isinstance(payload.get(wrapper), list):
                return _mapping_entries(payload[wrapper])
        if any(key in payload for key in POLYGON_KEYS):
            return [payload]
        raise ValueError(
            "answer is neither a list of segmentations nor one segmentation"
        )

    entries = _mapping_entries(payload)

    return entries


def _segmentation_rows(
    entries: List[Mapping],
    *,
    image_size_wh: SizeWH,
    upload_size_wh: SizeWH,
    max_polygon_vertices: int,
) -> Tuple[List[Tuple[List[float], str, float]], List[bytes]]:
    (width, height), (upload_width, upload_height) = image_size_wh, upload_size_wh
    rows, masks = [], []
    for entry in entries:
        polygon = _read_polygon(entry)
        if polygon is None or len(polygon) > max_polygon_vertices:
            # The entry is not logged: it may be the oversized payload.
            logger.warning(
                "Skipping VLM segmentation %r: no polygon of 3 to %d vertices",
                entry_label(entry),
                max_polygon_vertices,
            )
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, upload_width) * (
            width / upload_width
        )
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, upload_height) * (
            height / upload_height
        )
        vertices = polygon.round()
        if _polygon_area(vertices) < MINIMUM_POLYGON_AREA:
            logger.warning("Skipping VLM segmentation %r: no area", entry_label(entry))
            continue
        rle = mask_utils.frPyObjects([vertices.ravel().tolist()], height, width)[0]
        if mask_utils.area(rle) == 0:
            logger.warning(
                "Skipping VLM segmentation %r: no pixels", entry_label(entry)
            )
            continue
        x, y, box_width, box_height = mask_utils.toBbox(rle).tolist()
        rows.append(([x, y, x + box_width, y + box_height], entry_label(entry), 1.0))
        masks.append(rle["counts"])

    if entries and not rows:
        raise ValueError(f"none of {len(entries)} entries has a usable polygon")

    return rows, masks


def _read_polygon(entry: Mapping) -> Optional[np.ndarray]:
    for key in POLYGON_KEYS:
        raw = entry.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        vertices = _read_vertices(raw)
        if vertices is not None and len(vertices) >= MINIMUM_POLYGON_VERTICES:
            return np.array(vertices, dtype=float)

    return None


def _read_vertices(raw: list) -> Optional[List[List[float]]]:
    if all(isinstance(item, Mapping) for item in raw):
        pairs = [
            (_read_number(item.get("x")), _read_number(item.get("y"))) for item in raw
        ]
    elif all(isinstance(item, list) for item in raw):
        if any(len(item) != 2 for item in raw):
            return None
        pairs = [(_read_number(x), _read_number(y)) for x, y in raw]
    else:
        if len(raw) % 2:
            return None
        values = [_read_number(value) for value in raw]
        pairs = list(zip(values[0::2], values[1::2]))

    if any(x is None or y is None for x, y in pairs):
        return None

    vertices = [[x, y] for x, y in pairs]

    return vertices


def _polygon_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    area = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    return area


# --- sizes -------------------------------------------------------------------


def _size_wh(image: ImageData) -> SizeWH:
    height, width = image.size_hw

    return (width, height)


def _fits_anthropic(width: int, height: int) -> bool:
    tile = ANTHROPIC_IMAGE_TILE_PIXELS
    tiles_x, tiles_y = math.ceil(width / tile), math.ceil(height / tile)
    fits = (
        tiles_x * tile <= ANTHROPIC_DETECTION_MAX_EDGE_PIXELS
        and tiles_y * tile <= ANTHROPIC_DETECTION_MAX_EDGE_PIXELS
        and tiles_x * tiles_y <= ANTHROPIC_DETECTION_MAX_IMAGE_TOKENS
    )

    return fits


__all__ = [
    "ANTHROPIC_DETECTION_MAX_EDGE_PIXELS",
    "ANTHROPIC_DETECTION_MAX_IMAGE_TOKENS",
    "ANTHROPIC_IMAGE_TILE_PIXELS",
    "BOX_2D_KEYS",
    "BoxFormatName",
    "CLASSIFICATION_TASKS",
    "DETECTION_BOX_FORMATS",
    "DETECTION_MAX_EDGE_PIXELS",
    "DETECTION_TASKS",
    "DetectionBoxFormat",
    "INSTANCE_SEGMENTATION_PROMPT_TEMPLATE",
    "LABEL_KEYS",
    "MAX_POLYGON_VERTICES",
    "NAMED_0_1000_PROMPT_TEMPLATE",
    "NAMED_BOX_FIELDS",
    "POLYGON_KEYS",
    "SEGMENTATION_TASKS",
    "XYXY_0_1000_PROMPT_TEMPLATE",
    "XYXY_ABSOLUTE_PROMPT_TEMPLATE",
    "XYXY_PERCENT_PROMPT_TEMPLATE",
    "YXYX_0_1000_PROMPT_TEMPLATE",
    "anthropic_upload_size",
    "build_instance_segmentation_prompt",
    "build_object_detection_prompt",
    "checked_classes",
    "checked_upload_size",
    "decode_classification",
    "decode_instance_segmentations",
    "decode_object_detections",
    "decode_vlm_output",
    "detection_box_format",
    "entry_label",
    "extract_json",
    "max_edge_upload_size",
    "scale_confidence",
    "vlm_prediction_outputs",
]
