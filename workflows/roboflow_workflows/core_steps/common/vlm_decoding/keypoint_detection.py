"""Keypoint detection with a VLM: describe each skeleton, constrain the answer, decode it.

The request names the skeleton of every class: its keypoint names, the edges between
them and, optionally, where each keypoint sits on a reference instance (a template).
A response schema constrains the answer to one list of instances per class. Each
instance has a Gemini-native ``box_2d`` (``[y_min, x_min, y_max, x_max]`` normalized to
0-1000) and one property per keypoint, holding ``{"point": [y, x], "occluded": bool}``
(normalized to 0-1000) or ``null`` when the keypoint is outside the image or cannot be
placed.

The decoded prediction carries its keypoints the way keypoint models do (the
``keypoints_*`` fields of ``sv.Detections.data``), so keypoint consumers need no
VLM-specific code. A VLM reports no confidence, so a visible keypoint gets
``VISIBLE_KEYPOINT_CONFIDENCE`` and one the model marks occluded (hidden, its position
estimated) ``OCCLUDED_KEYPOINT_CONFIDENCE``. A keypoint the model could not place is
left out of its instance.
"""

import copy
import math
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import BaseModel, Field, ValidationError, model_validator
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.keypoints import MAX_KEYPOINT_SLOTS
from roboflow_workflows.core_steps.common.utils import (
    add_inference_keypoints_to_sv_detections,
    attach_parents_coordinates_to_sv_detections,
)
from roboflow_workflows.core_steps.common.vlm_decoding.detection_formats import (
    get_detection_box_format,
)
from roboflow_workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_json,
)
from roboflow_workflows.core_steps.common.vlm_decoding.tensor_native import (
    to_tensor_native_predictions,
)
from roboflow_workflows.core_steps.common.vlm_decoding.utils import create_classes_index
from roboflow_workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    PREDICTION_TYPE_KEY,
)
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from supervision.config import CLASS_NAME_DATA_FIELD

logger = get_logger(__name__)

PREDICTION_TYPE = "keypoint-detection"
KEYPOINT_DETECTION_TASK = "keypoint-detection"
VISIBLE_KEYPOINT_CONFIDENCE = 1.0
OCCLUDED_KEYPOINT_CONFIDENCE = 0.5
NORMALIZED_SCALE = 1000.0
BOX_FORMAT = "yxyx_0_1000"

KEYPOINT_SCHEMA_DESCRIPTION = (
    "The skeleton of every class, keyed by class name (keypoint-detection only). Each "
    'skeleton is `{"keypoints": [...], "edges": [[i, j], ...], "template": [[x, y], '
    "...]}`: the keypoint names in skeleton order, the pairs of connected keypoints as "
    "0-based indices into `keypoints`, and where each keypoint sits on a reference "
    "instance, as (x, y) fractions of its bounding box from the top-left corner (`null` "
    "for a keypoint without a position). `edges` and `template` are optional. Every "
    "class in `classes` needs a skeleton."
)
KEYPOINT_SCHEMA_EXAMPLE = {
    "license-plate": {
        "keypoints": ["TL", "TR", "BR", "BL"],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "template": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    }
}

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class KeypointSkeleton(BaseModel):
    """The skeleton of one class: what the model locates on each of its instances."""

    keypoints: List[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        max_length=MAX_KEYPOINT_SLOTS,
        description="Keypoint names in skeleton order; a keypoint's position in this "
        "list is its keypoint class id.",
        examples=[["TL", "TR", "BR", "BL"]],
    )
    edges: Optional[List[Tuple[int, int]]] = Field(
        default=None,
        description="Pairs of connected keypoints, as 0-based indices into "
        "`keypoints`.",
        examples=[[[0, 1], [1, 2], [2, 3], [3, 0]]],
    )
    template: Optional[List[Optional[Tuple[FiniteFloat, FiniteFloat]]]] = Field(
        default=None,
        description="Where each keypoint sits on a reference instance, as (x, y) "
        "fractions of its bounding box from the top-left corner; one entry per "
        "keypoint, `null` for a keypoint without a position.",
        examples=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
    )

    @model_validator(mode="after")
    def _validate_references(self) -> "KeypointSkeleton":
        count = len(self.keypoints)
        for first, second in self.edges or []:
            if not (0 <= first < count and 0 <= second < count):
                raise ValueError(
                    f"edge [{first}, {second}] points outside the {count} keypoints"
                )

        if self.template is not None and len(self.template) != count:
            raise ValueError(
                f"template has {len(self.template)} positions for {count} keypoints"
            )

        return self

    @property
    def answer_keys(self) -> List[str]:
        """The property name of each keypoint in the answer, in skeleton order.

        A name is its own key. A name that repeats takes a numbered key from its
        second occurrence on (``"wheel"``, ``"wheel (2)"``), because the answer needs
        one property per keypoint.
        """
        keys: List[str] = []
        taken = set()
        for name in self.keypoints:
            key = name
            occurrence = 1
            while key in taken:
                occurrence += 1
                key = f"{name} ({occurrence})"
            taken.add(key)
            keys.append(key)

        return keys


KeypointSchema = Dict[str, KeypointSkeleton]


def resolve_keypoint_skeletons(
    keypoint_schema: Any,
    *,
    classes: Optional[Sequence[str]],
) -> Dict[str, KeypointSkeleton]:
    """Parse a keypoint schema and pick the skeletons of the classes to detect.

    Args:
        keypoint_schema (Any): Manifest literal or resolved selector value: a mapping
            of class name to skeleton, as dicts or ``KeypointSkeleton`` objects.
        classes (Optional[Sequence[str]]): Classes to detect, in order. ``None`` or an
            empty list selects every class of the schema.

    Returns:
        Dict[str, KeypointSkeleton]: The skeleton of each class to detect, in the
            order of ``classes`` (schema order without them), each class once.

    Raises:
        ValueError: If the schema is missing or malformed, or a class has no skeleton.
    """
    if not isinstance(keypoint_schema, dict) or not keypoint_schema:
        raise ValueError(
            "`keypoint_schema` must map each class name to its skeleton for the "
            f"{KEYPOINT_DETECTION_TASK} task"
        )

    skeletons: Dict[str, KeypointSkeleton] = {}
    for class_name, skeleton in keypoint_schema.items():
        if isinstance(skeleton, KeypointSkeleton):
            skeletons[class_name] = skeleton
            continue
        try:
            skeletons[class_name] = KeypointSkeleton.model_validate(skeleton)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
                for detail in error.errors()
            )
            raise ValueError(
                f"`keypoint_schema` has an invalid skeleton for {class_name!r}: "
                f"{problems}"
            ) from error

    requested = list(dict.fromkeys(classes)) if classes else list(skeletons)
    missing = [class_name for class_name in requested if class_name not in skeletons]
    if missing:
        raise ValueError(f"`keypoint_schema` has no skeleton for classes: {missing}")

    resolved = {class_name: skeletons[class_name] for class_name in requested}

    return resolved


def build_keypoint_detection_prompt(skeletons: Dict[str, KeypointSkeleton]) -> str:
    """Write the instructions that go with the image.

    Args:
        skeletons (Dict[str, KeypointSkeleton]): The skeleton of each class to detect.

    Returns:
        str: The prompt: the answer's layout, then each class's keypoints, edges and,
            when it has one, its template.
    """
    lines = [
        "Find every instance of these classes in this image and locate its keypoints: "
        f"{', '.join(skeletons)}.",
        "",
        "Return JSON with one key per class, holding a list with one entry per "
        "instance of that class, or an empty list when the image has none. Each entry "
        "has:",
        '- "box_2d": the bounding box as [y_min, x_min, y_max, x_max], integers '
        "between 0 and 1000 normalized to the image height and width.",
        '- "keypoints": one key per keypoint of the class skeleton. Each value is '
        '{"point": [y, x], "occluded": bool}, with y and x integers between 0 and '
        "1000 normalized to the image height and width, or null when the keypoint is "
        'outside the image or its position cannot be estimated. Set "occluded" to '
        "true when the keypoint is hidden (by another object or by the instance "
        "itself) but you can still estimate where it is.",
    ]
    for class_name, skeleton in skeletons.items():
        lines.extend(_describe_skeleton(class_name, skeleton=skeleton))

    prompt = "\n".join(lines)

    return prompt


def _describe_skeleton(class_name: str, *, skeleton: KeypointSkeleton) -> List[str]:
    keys = skeleton.answer_keys
    lines = [
        "",
        f"Skeleton of a {class_name}:",
        f"Keypoints: {', '.join(keys)}",
    ]
    if skeleton.edges:
        edges = ", ".join(
            f"{keys[first]}-{keys[second]}" for first, second in skeleton.edges
        )
        lines.append(f"Edges (connected keypoints): {edges}")

    positions = [
        f"{key} ({position[0]:.2f}, {position[1]:.2f})"
        for key, position in zip(keys, skeleton.template or [])
        if position is not None
    ]
    if positions:
        lines.append(
            f"Template: the position of each keypoint on a reference {class_name} "
            "drawn inside its bounding box, as (x, y) fractions of the box width and "
            f"height from its top-left corner: {', '.join(positions)}."
        )

    return lines


def build_keypoint_response_schema(skeletons: Dict[str, KeypointSkeleton]) -> dict:
    """Build the response schema that constrains the answer.

    Args:
        skeletons (Dict[str, KeypointSkeleton]): The skeleton of each class to detect.

    Returns:
        dict: An object with one required list of instances per class. An instance
            requires its ``box_2d`` and one nullable property per keypoint.
    """
    class_names = list(skeletons)
    schema = {
        "type": "OBJECT",
        "properties": {
            class_name: {
                "type": "ARRAY",
                "items": _instance_schema(skeleton),
            }
            for class_name, skeleton in skeletons.items()
        },
        "required": class_names,
        "propertyOrdering": class_names,
    }

    return schema


def _instance_schema(skeleton: KeypointSkeleton) -> dict:
    keys = skeleton.answer_keys
    keypoint = {
        "type": "OBJECT",
        "nullable": True,
        "properties": {
            "point": {
                "type": "ARRAY",
                "items": {"type": "INTEGER"},
                "minItems": 2,
                "maxItems": 2,
            },
            "occluded": {"type": "BOOLEAN"},
        },
        "required": ["point", "occluded"],
        "propertyOrdering": ["point", "occluded"],
    }

    return {
        "type": "OBJECT",
        "properties": {
            "box_2d": {
                "type": "ARRAY",
                "items": {"type": "INTEGER", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
            "keypoints": {
                "type": "OBJECT",
                "properties": {key: copy.deepcopy(keypoint) for key in keys},
                "required": keys,
                "propertyOrdering": keys,
            },
        },
        "required": ["box_2d", "keypoints"],
        "propertyOrdering": ["box_2d", "keypoints"],
    }


def decode_keypoint_detections(
    raw_output: str,
    *,
    image: WorkflowImageData,
    skeletons: Dict[str, KeypointSkeleton],
    classes: Optional[List[str]],
    inference_id: str,
) -> Tuple[bool, Any]:
    """Decode a keypoint answer into keypoint detections in original-image pixels.

    Never raises: a malformed answer is reported through the error flag, so it cannot
    take down a workflow run. An invalid instance is skipped, a keypoint with an
    unusable point is left out of its instance, and an answer whose instances are all
    invalid is an error.

    Args:
        raw_output (str): The model's JSON answer.
        image (WorkflowImageData): Image the answer refers to.
        skeletons (Dict[str, KeypointSkeleton]): The skeleton of each class the
            request asked for, as ``resolve_keypoint_skeletons`` returns them.
        classes (Optional[List[str]]): Class names the block was given; a class's
            position in them is its class id. Defaults to the skeletons' classes.
        inference_id (str): Identifier attached to every instance.

    Returns:
        Tuple[bool, Any]: The error flag and the keypoint detections (``None`` on
            error), converted to the tensor carrier when that representation is on.
    """
    error, parsed = extract_json(raw_output)
    if error:
        return True, None

    try:
        detections = _build_keypoint_detections(
            parsed,
            image=image,
            skeletons=skeletons,
            classes=classes or list(skeletons),
            inference_id=inference_id,
        )
        predictions = to_tensor_native_predictions(
            predictions=detections,
            image=image,
            classes=classes or list(skeletons),
            inference_id=inference_id,
        )
    except Exception as error:
        logger.warning(
            "Could not decode VLM keypoint-detection output. Error type: %s. "
            "Details: %s",
            error.__class__.__name__,
            error,
        )
        return True, None

    return False, predictions


def _build_keypoint_detections(
    parsed: Any,
    *,
    image: WorkflowImageData,
    skeletons: Dict[str, KeypointSkeleton],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    if not isinstance(parsed, dict):
        raise ValueError("The answer is not an object keyed by class")

    image_height, image_width = image._read_shape_without_materialization()
    class_name2id = create_classes_index(classes=classes)
    box_format = get_detection_box_format(BOX_FORMAT)

    xyxy, class_id, class_name, keypoints = [], [], [], []
    entries_count = 0
    for name, skeleton in skeletons.items():
        entries = parsed.get(name)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"The instances of {name!r} are not a list")

        keys = skeleton.answer_keys
        for entry in entries:
            entries_count += 1
            box = (
                box_format.to_pixel_xyxy(
                    entry=entry,
                    image_width=image_width,
                    image_height=image_height,
                )
                if isinstance(entry, dict)
                else None
            )
            if box is None or not isinstance(entry.get("keypoints"), dict):
                logger.warning(
                    "Skipping VLM keypoint instance without a box: %r", entry
                )
                continue

            x_0, y_0, x_1, y_1 = box
            xyxy.append([min(x_0, x_1), min(y_0, y_1), max(x_0, x_1), max(y_0, y_1)])
            class_id.append(class_name2id.get(name, -1))
            class_name.append(name)
            keypoints.append(
                _read_keypoints(
                    entry["keypoints"],
                    names=skeleton.keypoints,
                    keys=keys,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

    if entries_count and not xyxy:
        raise ValueError(f"None of the {entries_count} instances has a valid box")

    count = len(xyxy)
    data = {
        CLASS_NAME_DATA_FIELD: np.array(class_name) if count else np.empty(0),
        IMAGE_DIMENSIONS_KEY: np.array([[image_height, image_width]] * count),
        INFERENCE_ID_KEY: np.array([inference_id] * count),
        DETECTION_ID_KEY: np.array([str(uuid4()) for _ in range(count)]),
        PREDICTION_TYPE_KEY: np.array([PREDICTION_TYPE] * count),
    }
    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=float).round(0) if count else np.empty((0, 4)),
        # The prompt asks for no instance confidence; downstream filters see 1.0.
        confidence=np.ones(count) if count else np.empty(0),
        class_id=np.array(class_id).astype(int) if count else np.empty(0),
        mask=None,
        tracker_id=None,
        data=data,
    )
    detections = add_inference_keypoints_to_sv_detections(
        inference_prediction=[
            {KEYPOINTS_KEY_IN_INFERENCE_RESPONSE: instance_keypoints}
            for instance_keypoints in keypoints
        ],
        detections=detections,
    )
    if count == 0:
        # Per-row data is empty, so keep the frame size where
        # `empty_detections_with_image_metadata` keeps it.
        detections.metadata[IMAGE_DIMENSIONS_KEY] = [image_height, image_width]

    detections = attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )

    return detections


def _read_keypoints(
    raw: dict,
    *,
    names: List[str],
    keys: List[str],
    image_width: int,
    image_height: int,
) -> List[dict]:
    """The instance's placed keypoints, in skeleton order, as keypoint models list them."""
    keypoints = []
    for keypoint_class_id, (name, key) in enumerate(zip(names, keys)):
        point = _read_point(raw.get(key))
        if point is None:
            continue

        y, x = point
        occluded = raw[key].get("occluded") is True
        keypoints.append(
            {
                "x": x * image_width / NORMALIZED_SCALE,
                "y": y * image_height / NORMALIZED_SCALE,
                KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE: keypoint_class_id,
                KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE: name,
                KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE: (
                    OCCLUDED_KEYPOINT_CONFIDENCE
                    if occluded
                    else VISIBLE_KEYPOINT_CONFIDENCE
                ),
            }
        )

    return keypoints


def _read_point(value: Any) -> Optional[Tuple[float, float]]:
    """A keypoint's ``[y, x]``, or None when it is null, malformed or off the image."""
    if not isinstance(value, dict):
        return None

    point = value.get("point")
    if not isinstance(point, list) or len(point) != 2:
        return None

    coordinates = []
    for coordinate in point:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            or not 0 <= coordinate <= NORMALIZED_SCALE
        ):
            return None
        coordinates.append(float(coordinate))

    y, x = coordinates

    return y, x
