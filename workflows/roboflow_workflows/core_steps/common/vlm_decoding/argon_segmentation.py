"""Decode Argon normalized YX components and holes into RLE instance masks."""

import math
from typing import Any, List, Tuple
from uuid import uuid4

import numpy as np
import supervision as sv
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from roboflow_workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_json,
)
from roboflow_workflows.core_steps.common.vlm_decoding.segmentation import (
    extract_segmentation_entries,
)
from roboflow_workflows.core_steps.common.vlm_decoding.tensor_native import (
    to_tensor_native_predictions,
)
from roboflow_workflows.core_steps.common.vlm_decoding.utils import create_classes_index
from roboflow_workflows.environment import (
    WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES,
)
from roboflow_workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
)
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from supervision.config import CLASS_NAME_DATA_FIELD

logger = get_logger(__name__)
PREDICTION_TYPE = "instance-segmentation"

ARGON_SEGMENTATION_PROMPT = "Segment every visible instance of these classes: {class_list}. Treat each distinct visible physical object as one instance. Segment visible surfaces only; do not infer hidden portions. Exclude cast shadows, reflections and background. Return only a JSON list. Each item must contain label (use the exact class name), box_2d [ymin,xmin,ymax,xmax] normalized to 0-1000 relative to the whole image, and mask, a list of components. Each component has outer, a polygon list of [y,x] vertices, and holes, a list of polygon lists of [y,x] vertices; all coordinates are normalized to 0-1000 relative to the WHOLE image. Follow the visible object boundary closely. Do not output bounding rectangles as masks. Keep separate physical objects separate. If one physical object has separate visible components because of occlusion, put each visible component in a separate component object inside that same object mask list; do not connect across the occluder. For each component, outer encloses its visible region. Use holes for enclosed openings or enclosed occluders that are not part of the object; use an empty holes list when none exist. All mask points use [y,x], vertical first and horizontal second. Subtract holes only from their own component, then union the components of one physical object."

ARGON_SEGMENTATION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "box_2d": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "mask": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "outer": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "minItems": 3,
                        },
                        "holes": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "minItems": 3,
                            },
                        },
                    },
                    "required": ["outer", "holes"],
                },
            },
        },
        "required": ["label", "box_2d", "mask"],
    },
}


def _read_ring(raw: object) -> list:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("Each mask boundary requires at least three YX points")
    points = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Mask vertices must be [y, x] pairs")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1000
            for v in point
        ):
            raise ValueError("Mask coordinates must be finite numbers in [0, 1000]")
        points.append(point)
    return points


def _read_components(entry: dict) -> list:
    raw = entry.get("mask")
    if not isinstance(raw, list) or not raw:
        raise ValueError("An instance requires at least one mask component")
    components = []
    vertex_count = 0
    for component in raw:
        if not isinstance(component, dict) or not isinstance(
            component.get("holes"), list
        ):
            raise ValueError("Each component requires outer and holes")
        outer = _read_ring(component.get("outer"))
        holes = [_read_ring(hole) for hole in component["holes"]]
        vertex_count += len(outer) + sum(len(hole) for hole in holes)
        if vertex_count > WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES:
            raise ValueError("Instance exceeds the segmentation vertex limit")
        components.append((outer, holes))
    return components


def _components_to_rle(components: list, *, width: int, height: int) -> dict:
    union = Image.new("1", (width, height))
    for outer, holes in components:
        component = Image.new("1", (width, height))
        draw = ImageDraw.Draw(component)
        draw.polygon([(x * width / 1000, y * height / 1000) for y, x in outer], fill=1)
        for hole in holes:
            draw.polygon(
                [(x * width / 1000, y * height / 1000) for y, x in hole], fill=0
            )
        union.paste(1, mask=component)
    encoded = mask_utils.encode(np.asfortranarray(np.asarray(union, dtype=np.uint8)))
    return encoded


def decode_argon_segmentations(
    raw_output: str,
    *,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
) -> Tuple[bool, Any]:
    """Decode Argon multipart masks, preserving holes and instance identity.

    Args:
        raw_output (str): Model JSON with normalized YX mask components.
        image (WorkflowImageData): Original image and parent metadata.
        classes (List[str]): Requested labels used to assign class IDs.
        inference_id (str): Identifier attached to decoded predictions.

    Returns:
        Tuple[bool, Any]: Error flag and RLE predictions,
            converted to the tensor carrier when enabled. Invalid instances are
            skipped; wholly malformed nonempty responses return an error.
    """
    error, parsed = extract_json(raw_output)
    if error:
        return True, None

    try:
        entries = extract_segmentation_entries(parsed)
        image_height, image_width = image._read_shape_without_materialization()
        class_name2id = create_classes_index(classes=classes)
        xyxy, rle_masks, class_id, class_name = [], [], [], []
        for entry in entries:
            try:
                label = entry.get("label")
                if not isinstance(label, str):
                    raise ValueError("Instance requires a text label")
                components = _read_components(entry)
                rle = _components_to_rle(
                    components, width=image_width, height=image_height
                )
                if mask_utils.area(rle) == 0:
                    raise ValueError("Instance covers no pixels")
            except (ValueError, TypeError, KeyError) as error:
                logger.warning("Skipping invalid Argon mask: %s", error)
                continue
            x, y, w, h = mask_utils.toBbox(rle)
            xyxy.append([x, y, x + w, y + h])
            rle_masks.append(rle)
            class_id.append(class_name2id.get(label, -1))
            class_name.append(label)
        if entries and not xyxy:
            raise ValueError("No valid Argon masks in nonempty response")
        count = len(xyxy)
        data = {
            CLASS_NAME_DATA_FIELD: np.array(class_name) if count else np.empty(0),
            IMAGE_DIMENSIONS_KEY: np.array([[image_height, image_width]] * count),
            INFERENCE_ID_KEY: np.array([inference_id] * count),
            DETECTION_ID_KEY: np.array([str(uuid4()) for _ in range(count)]),
            PREDICTION_TYPE_KEY: np.array([PREDICTION_TYPE] * count),
            RLE_MASK_KEY_IN_SV_DETECTIONS: np.array(rle_masks, dtype=object),
        }
        detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=float) if count else np.empty((0, 4)),
            # The prompt asks for no confidence; downstream filters see 1.0.
            confidence=np.ones(count) if count else np.empty(0),
            class_id=np.array(class_id).astype(int) if count else np.empty(0),
            mask=None,
            tracker_id=None,
            data=data,
        )
        if count == 0:
            # Per-row data is empty, so keep the frame size where
            # `empty_detections_with_image_metadata` keeps it.
            detections.metadata[IMAGE_DIMENSIONS_KEY] = [image_height, image_width]
        detections = attach_parents_coordinates_to_sv_detections(
            detections=detections,
            image=image,
        )
        predictions = to_tensor_native_predictions(
            predictions=detections,
            image=image,
            classes=classes,
            inference_id=inference_id,
        )
        return False, predictions
    except Exception as error:
        logger.warning("Could not decode Argon segmentation: %s", error)
        return True, None
