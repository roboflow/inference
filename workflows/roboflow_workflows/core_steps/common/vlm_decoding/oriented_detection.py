"""Turn a VLM oriented-object-detection answer into native OBB detections.

The prompt asks for one rotated rectangle per object as its centre, its two side
lengths and the direction of its main axis, in absolute pixel coordinates of the
uploaded image. Models follow that contract more reliably than a list of corners
(no corner-order convention to explain) and it maps one-to-one onto the native
OBB tuple ``[cx, cy, width, height, theta]`` of ``common/oriented.py``: the
answer's ``length`` is the tuple's ordered ``width`` (the side along the main
axis, and along the front for a class with a front), its ``width`` is the
tuple's ``height``, and ``angle_deg`` is the clockwise angle of the main axis in
image coordinates.

``class_heading`` says which classes have a front. A class with a front reports
the direction its front points to over the full circle; a class without one
reports its long axis, so its orientation is only defined modulo 180 degrees.
Decoding mirrors ``detections.py``: entries are read leniently and the geometry
is scaled back onto the original image. The result is the native OBB carrier
(ordered corners, per-row front flag and the frozen ``class_heading``), the same
one ``roboflow_core/roboflow_oriented_object_detection_model@v1`` produces, so
the visualization, filters and the native serializer all apply to it. VLMs give
no confidence, so every detection has confidence ``1.0``.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import supervision as sv
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.oriented import (
    native_obb_from_inference,
    native_obb_to_tensor,
)
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from roboflow_workflows.core_steps.common.vlm_decoding.detection_formats import (
    _read_number,
    get_detection_class_name,
)
from roboflow_workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_json,
)
from roboflow_workflows.core_steps.common.vlm_decoding.utils import create_classes_index
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

logger = get_logger(__name__)

ORIENTED_OBJECT_DETECTION_TASK = "oriented-object-detection"
ORIENTED_DETECTIONS_WRAPPER_KEY = "oriented_detections"
ORIENTED_DETECTION_CONFIDENCE = 1.0
ENTRY_NUMBER_FIELDS = ("cx", "cy", "length", "width", "angle_deg")
"""Numeric fields of an answer entry; every entry also carries a ``label``."""

CLASS_HEADING_DESCRIPTION = (
    "Which classes have a front, as a mapping of class name to `true` (the class has "
    "a front) or `false`. For a class with a front the model reports the direction "
    "the object's front points to, over the full circle, and the predictions keep it "
    "as the front of the oriented box. A class that is `false` or missing has no "
    "front: its boxes are undirected and follow the long axis. The mapping, completed "
    "for every class in `classes`, is the `class_heading` of the predictions."
)
CLASS_HEADING_EXAMPLE = {"forklift": True, "scissor-lift": False}

ORIENTED_DETECTION_PROMPT_TEMPLATE = (
    "Detect all objects in this image and return an oriented bounding box for each "
    "one: the smallest rotated rectangle that fully encloses the object. "
    "Image coordinates: the origin is the top-left corner, x grows to the right and "
    "y grows downward, and the image is {width}x{height} pixels. "
    'Output JSON with the key "oriented_detections" holding a list where each entry '
    'has the text label in the key "label" and these numbers: '
    '"cx" and "cy", the center of the rectangle in absolute pixel coordinates; '
    '"length", the size of the rectangle along the object\'s main axis, in pixels; '
    '"width", its size across that axis, in pixels; '
    '"angle_deg", the direction of the main axis in degrees, measured clockwise from '
    "the +x axis (0 points right, 90 points down, 180 or -180 points left, -90 points up)."
    "{front_rules} "
    "Only use these labels: {class_list}"
)
FRONT_RULE_TEMPLATE = (
    " Classes with a front: {names}. For them the main axis points toward the front "
    "of the object, so angle_deg is the direction the front points to, in the range "
    "(-180, 180], and length is the size of the rectangle along that direction."
)
NO_FRONT_RULE_TEMPLATE = (
    " Classes without a front: {names}. For them the main axis is the longer side of "
    "the rectangle, so angle_deg is its direction in the range (-90, 90] and length "
    "is at least width."
)


def resolve_class_heading(
    class_heading: Optional[Dict[str, Any]],
    *,
    labels: List[str],
) -> Dict[str, bool]:
    """Complete the front mapping for the labels the model may predict.

    Names outside ``labels`` are ignored, so a project-wide mapping can be
    passed unchanged.

    Args:
        class_heading (Optional[Dict[str, Any]]): Class name to ``True`` (has a
            front) / ``False``, or ``None`` when no class has a front.
        labels (List[str]): Class labels of the predictions, in class-id order.

    Returns:
        Mapping from every label to a plain ``bool``; labels the input does
        not name have no front.

    Raises:
        ValueError: If the mapping is not a dictionary of class names to
            booleans.
    """
    if class_heading is None:
        return {label: False for label in labels}

    if not isinstance(class_heading, dict):
        raise ValueError("`class_heading` must map class names to true or false")

    for name, has_front in class_heading.items():
        if not isinstance(name, str) or type(has_front) is not bool:
            raise ValueError(
                "`class_heading` must map class names to true or false, "
                f"got {name!r}: {has_front!r}"
            )

    resolved_heading = {label: class_heading.get(label, False) for label in labels}

    return resolved_heading


def build_oriented_detection_prompt(
    classes: List[str],
    *,
    class_heading: Dict[str, bool],
    upload_width: int,
    upload_height: int,
) -> str:
    """Render the oriented-object-detection prompt.

    Args:
        classes (List[str]): Class labels the model may predict.
        class_heading (Dict[str, bool]): Mapping from each label to whether it
            has a front, as ``resolve_class_heading`` returns it.
        upload_width (int): Width of the image as uploaded to the model.
        upload_height (int): Height of the image as uploaded to the model.

    Returns:
        The rendered prompt text.

    Raises:
        ValueError: If the class list is empty or the upload dimensions are
            missing - the contract asks for absolute pixel coordinates, so the
            prompt must name the frame.
    """
    if not classes:
        raise ValueError("Oriented object detection requires at least one class")

    if not upload_width or not upload_height:
        raise ValueError(
            "Oriented-object-detection prompt requires the dimensions of the "
            "uploaded image, but none were provided."
        )

    with_front = [name for name in classes if class_heading.get(name, False)]
    without_front = [name for name in classes if not class_heading.get(name, False)]
    front_rules = ""
    if with_front:
        front_rules += FRONT_RULE_TEMPLATE.format(names=", ".join(with_front))
    if without_front:
        front_rules += NO_FRONT_RULE_TEMPLATE.format(names=", ".join(without_front))

    prompt = ORIENTED_DETECTION_PROMPT_TEMPLATE.format(
        width=upload_width,
        height=upload_height,
        front_rules=front_rules,
        class_list=", ".join(classes),
    )

    return prompt


def build_oriented_json_schema(labels: List[str]) -> dict:
    """Build the strict JSON schema (OpenAI structured outputs) of the answer.

    Args:
        labels (List[str]): Class labels the model may predict.

    Returns:
        The schema, with ``label`` constrained to ``labels``.
    """
    properties = {"label": {"type": "string", "enum": list(dict.fromkeys(labels))}}
    properties.update({field: {"type": "number"} for field in ENTRY_NUMBER_FIELDS})

    return {
        "type": "object",
        "properties": {
            ORIENTED_DETECTIONS_WRAPPER_KEY: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": [ORIENTED_DETECTIONS_WRAPPER_KEY],
        "additionalProperties": False,
    }


def build_oriented_response_schema(labels: List[str]) -> dict:
    """Build the Gemini response schema of the answer.

    Args:
        labels (List[str]): Class labels the model may predict.

    Returns:
        The OpenAPI-style schema Gemini's ``response_schema`` takes, with
        ``label`` constrained to ``labels``.
    """
    properties = {"label": {"type": "STRING", "enum": list(dict.fromkeys(labels))}}
    properties.update({field: {"type": "NUMBER"} for field in ENTRY_NUMBER_FIELDS})

    return {
        "type": "OBJECT",
        "properties": {
            ORIENTED_DETECTIONS_WRAPPER_KEY: {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": list(properties),
                    "propertyOrdering": list(properties),
                },
            }
        },
        "required": [ORIENTED_DETECTIONS_WRAPPER_KEY],
    }


def decode_oriented_detections(
    raw_output: str,
    *,
    image: WorkflowImageData,
    labels: List[str],
    class_heading: Dict[str, bool],
    upload_width: int,
    upload_height: int,
) -> Tuple[bool, Optional[Any]]:
    """Decode a raw VLM answer into native OBB detections in original-image pixels.

    Never raises: any failure is reported through ``error_status`` and
    logged, so a malformed model answer cannot take down a workflow run.

    Args:
        raw_output (str): Raw string produced by the model.
        image (WorkflowImageData): Workflow image the detections refer to
            (original resolution).
        labels (List[str]): Class labels used to map entries onto class ids;
            entries with another label are dropped, since a native prediction
            needs a class with a known front.
        class_heading (Dict[str, bool]): Mapping from each label to whether it
            has a front.
        upload_width (int): Width of the image as uploaded; the geometry is
            expressed in that frame.
        upload_height (int): Height of the image as uploaded, same requirement.

    Returns:
        Tuple of ``(error_status, detections)``; ``detections`` is ``None``
        when ``error_status`` is ``True``. Under
        ``ENABLE_TENSOR_DATA_REPRESENTATION`` it is the tensor carrier,
        otherwise ``sv.Detections``.
    """
    error_status, parsed_data = extract_json(raw_output)
    if error_status:
        return True, None

    try:
        detections = build_oriented_detections(
            parsed_data,
            image=image,
            labels=labels,
            class_heading=class_heading,
            upload_width=upload_width,
            upload_height=upload_height,
        )
        if ENABLE_TENSOR_DATA_REPRESENTATION:
            detections = native_obb_to_tensor(detections)
    except Exception as error:
        logger.warning(
            "Could not decode VLM oriented-object-detection output. "
            "Error type: %s. Details: %s",
            error.__class__.__name__,
            error,
        )
        return True, None

    return False, detections


def build_oriented_detections(
    parsed_data: object,
    *,
    image: WorkflowImageData,
    labels: List[str],
    class_heading: Dict[str, bool],
    upload_width: int,
    upload_height: int,
) -> sv.Detections:
    """Build native OBB ``sv.Detections`` from an already-parsed JSON payload.

    Args:
        parsed_data (object): JSON payload extracted from the VLM output.
        image (WorkflowImageData): Workflow image the detections refer to.
        labels (List[str]): Class labels used to map entries onto class ids.
        class_heading (Dict[str, bool]): Mapping from each label to whether it
            has a front.
        upload_width (int): Width of the image as uploaded.
        upload_height (int): Height of the image as uploaded.

    Returns:
        Native OBB detections in the original image's coordinate space.

    Raises:
        ValueError: If the payload shape is not recognised, the upload
            dimensions are missing, or entries were present but none was
            well-formed - i.e. the model answered in a different contract
            (a partial mismatch keeps skipping the offending entries).
    """
    if not upload_width or not upload_height:
        raise ValueError("The dimensions of the uploaded image are required")

    entries = extract_oriented_entries(parsed_data)
    image_height, image_width = image._read_shape_without_materialization()
    scale_x, scale_y = image_width / upload_width, image_height / upload_height
    size_scale = (scale_x + scale_y) / 2
    label_to_id = create_classes_index(classes=labels)

    rows = []
    for entry in entries:
        row = _read_entry(
            entry,
            label_to_id=label_to_id,
            class_heading=class_heading,
            scale_x=scale_x,
            scale_y=scale_y,
            size_scale=size_scale,
        )
        if row is None:
            logger.warning(
                "Skipping VLM oriented detection entry that is not well formed: %r",
                entry,
            )
            continue

        rows.append(row)

    if entries and not rows:
        # Every entry was skipped: the model answered in another contract.
        # Reporting that as "nothing detected" would hide it, so fail instead
        # and let `decode_oriented_detections` surface `error_status=True`.
        raise ValueError(
            f"none of {len(entries)} oriented detection entries was well formed"
        )

    native_response = {
        "task_type": ORIENTED_OBJECT_DETECTION_TASK,
        "image": {"width": image_width, "height": image_height},
        "class_heading": {label: bool(class_heading.get(label)) for label in labels},
        "predictions": rows,
    }
    detections = native_obb_from_inference(native_response)
    detections = attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )

    return detections


def extract_oriented_entries(parsed: Any) -> List[dict]:
    """Pull the list of oriented entries out of a parsed JSON payload.

    Accepts a bare list of entries or an ``{"oriented_detections": [...]}``
    wrapper. Non-dict members are skipped, but a non-empty list holding no
    entry objects at all is rejected, so it surfaces as ``error_status``
    instead of "no objects found".

    Args:
        parsed (Any): JSON payload extracted from the VLM output.

    Returns:
        List of raw entry dicts.

    Raises:
        ValueError: If the payload matches none of the accepted shapes.
    """
    if isinstance(parsed, dict):
        parsed = parsed.get(ORIENTED_DETECTIONS_WRAPPER_KEY)

    if not isinstance(parsed, list):
        raise ValueError("Unexpected oriented object detection response format")

    entries = [entry for entry in parsed if isinstance(entry, dict)]
    if parsed and not entries:
        raise ValueError(
            "Oriented object detection response is a list without any entries"
        )

    return entries


def _read_entry(
    entry: dict,
    *,
    label_to_id: Dict[str, int],
    class_heading: Dict[str, bool],
    scale_x: float,
    scale_y: float,
    size_scale: float,
) -> Optional[dict]:
    """Build one native prediction row from an answer entry, or ``None``."""
    label = get_detection_class_name(entry)
    if label not in label_to_id:
        return None

    numbers = {field: _read_number(entry.get(field)) for field in ENTRY_NUMBER_FIELDS}
    if any(value is None for value in numbers.values()):
        return None

    if numbers["length"] <= 0 or numbers["width"] <= 0:
        return None

    has_front = bool(class_heading.get(label))
    length, width, theta = _normalize_orientation(
        length=numbers["length"],
        width=numbers["width"],
        angle_deg=numbers["angle_deg"],
        has_front=has_front,
    )
    row = {
        "obb": [
            numbers["cx"] * scale_x,
            numbers["cy"] * scale_y,
            length * size_scale,
            width * size_scale,
            theta,
        ],
        "class": label,
        "class_id": label_to_id[label],
        "confidence": ORIENTED_DETECTION_CONFIDENCE,
        "heading": has_front,
        "detection_id": str(uuid4()),
    }

    return row


def _normalize_orientation(
    *,
    length: float,
    width: float,
    angle_deg: float,
    has_front: bool,
) -> Tuple[float, float, float]:
    """Express a reported rectangle in the native ordered-axes convention.

    A class with a front keeps its sides and its full-circle angle, wrapped
    into ``[-pi, pi)``. A class without one is undirected, so the same footprint
    is written with the longer side first (a swap adds a quarter turn) and the
    angle wrapped into ``[-pi/2, pi/2)``. Returns ``(length, width, theta)``
    with ``theta`` in radians.
    """
    theta = math.radians(angle_deg)
    if has_front:
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        return length, width, theta

    if length < width:
        length, width = width, length
        theta += math.pi / 2
    theta = (theta + math.pi / 2) % math.pi - math.pi / 2

    return length, width, theta
