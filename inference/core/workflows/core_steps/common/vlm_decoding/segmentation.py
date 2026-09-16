"""Turn a VLM instance-segmentation answer into ``sv.Detections`` with masks.

The prompt asks for one polygon per instance as a flat ``[x1, y1, x2, y2, ...]``
vertex list in absolute pixel coordinates of the uploaded image, which is the
contract that scored best for GPT-6 Astra on the vlm-exam mask ground truth
(absolute pixels of the ORIGINAL image beat 0-1 floats on cost and 0-1000 ints
on small objects, and pre-downscaling the upload cost mask AP - hence no
``DETECTION_MAX_EDGE_PIXELS`` resize on this path). Decoding mirrors
``detections.py``: entries are read leniently and vertices are scaled back
onto the original image. Each polygon is then encoded straight to COCO RLE
(``pycocotools.frPyObjects``) - no dense mask is ever materialised, so memory
stays proportional to the run count however many instances the model returns
- and the result carries ``mask=None`` plus ``data["rle_mask"]``, the
RLE-first contract of ``segment_anything3@v3`` / ``instance_segmentation@v4``
that the visualization, crop, upload blocks and the Auto Label worker already
consume (they decode lazily, one instance at a time, only where pixels are
needed).
"""

import logging
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import supervision as sv
from pycocotools import mask as mask_utils
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.env import WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.core_steps.common.vlm_decoding.detection_formats import (
    DETECTIONS_WRAPPER_KEY,
    _read_number,
    get_detection_class_name,
)
from inference.core.workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_json,
)
from inference.core.workflows.core_steps.common.vlm_decoding.utils import (
    create_classes_index,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

logger = logging.getLogger(__name__)

PREDICTION_TYPE = "instance-segmentation"

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

# Entry shape vocabulary, mirroring `detection_formats.BOX_2D_KEYS`: the
# prompted key first, then the names models drift to.
POLYGON_KEYS = ("polygon", "segmentation", "points", "mask")
SEGMENTATIONS_WRAPPER_KEY = "segmentations"
MINIMUM_POLYGON_VERTICES = 3
MINIMUM_POLYGON_AREA = 1.0
"""Smallest enclosed area (in original-image px²) a polygon must have."""


def build_instance_segmentation_prompt(
    classes: List[str],
    upload_width: int,
    upload_height: int,
) -> str:
    """Render the instance-segmentation prompt.

    Args:
        classes: Class names the model may predict.
        upload_width: Width of the image as uploaded to the model.
        upload_height: Height of the image as uploaded to the model.

    Returns:
        The rendered prompt text.

    Raises:
        ValueError: If the upload dimensions are missing - the contract asks
            for absolute pixel coordinates, so the prompt must name the frame.
    """
    if not upload_width or not upload_height:
        raise ValueError(
            "Instance-segmentation prompt requires the dimensions of the "
            "uploaded image, but none were provided."
        )
    return INSTANCE_SEGMENTATION_PROMPT_TEMPLATE.format(
        class_list=", ".join(classes),
        width=upload_width,
        height=upload_height,
    )


def decode_instance_segmentations(
    raw_output: str,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
    upload_width: Optional[int] = None,
    upload_height: Optional[int] = None,
) -> Tuple[bool, Optional[sv.Detections]]:
    """Decode a raw VLM answer into RLE-masked detections in original-image pixels.

    Never raises: any failure is reported through ``error_status`` and
    logged, so a malformed model answer cannot take down a workflow run.

    Args:
        raw_output: Raw string produced by the model.
        image: Workflow image the instances refer to (original resolution).
        classes: Class names used to map labels onto class ids; labels
            outside the list are kept with ``class_id == -1``.
        inference_id: Identifier attached to every parsed instance.
        upload_width: Width of the image as uploaded; the polygon vertices
            are expressed in that frame.
        upload_height: Height of the image as uploaded, same requirement.

    Returns:
        Tuple of ``(error_status, detections)``; ``detections`` is ``None``
        when ``error_status`` is ``True``.
    """
    error_status, parsed_data = extract_json(raw_output)
    if error_status:
        return True, None
    try:
        detections = build_instance_segmentations(
            parsed_data=parsed_data,
            image=image,
            classes=classes,
            inference_id=inference_id,
            upload_width=upload_width,
            upload_height=upload_height,
        )
        return False, detections
    except Exception as error:
        logger.warning(
            "Could not decode VLM instance-segmentation output. "
            "Error type: %s. Details: %s",
            error.__class__.__name__,
            error,
        )
        return True, None


def build_instance_segmentations(
    parsed_data: object,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
    upload_width: Optional[int] = None,
    upload_height: Optional[int] = None,
) -> sv.Detections:
    """Build RLE-masked ``sv.Detections`` from an already-parsed JSON payload.

    Every entry is validated first (well-formed polygon, at most
    ``WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES`` vertices, non-zero
    enclosed area); the survivors are encoded polygon -> COCO RLE at the original image
    resolution without rasterising a dense mask, and their bounding box is
    taken from the RLE (``toBbox``) so box and mask always agree.

    Args:
        parsed_data: JSON payload extracted from the VLM output.
        image: Workflow image the instances refer to.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to every parsed instance.
        upload_width: Width of the image as uploaded.
        upload_height: Height of the image as uploaded.

    Returns:
        Detections with ``mask=None`` and one COCO RLE (``{"size", "counts"}``,
        ``counts`` as bytes) per instance under ``data["rle_mask"]``, in the
        original image's coordinate space.

    Raises:
        ValueError: If the payload shape is not recognised, the upload
            dimensions are missing, or NO entry carried a usable polygon
            (a partial failure keeps skipping the offending entries).
    """
    if classes is None:
        raise ValueError("Class list is required to decode instance segmentations")
    if not upload_width or not upload_height:
        raise ValueError(
            "Instance-segmentation decoding requires the dimensions of the "
            "uploaded image to map polygon vertices back onto the original "
            "image, but none were provided."
        )
    entries = extract_segmentation_entries(parsed_data)
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image._read_shape_without_materialization()
    scale_x = image_width / upload_width
    scale_y = image_height / upload_height

    # Validate every entry before encoding anything, so nothing is allocated
    # for entries that end up skipped.
    valid: List[tuple] = []
    for entry in entries:
        polygon = read_polygon(entry)
        if polygon is None:
            logger.warning(
                "Skipping VLM segmentation entry without a well-formed polygon: %r",
                entry,
            )
            continue
        if len(polygon) > WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES:
            # The RLE encoder allocates memory proportional to the outline
            # length, so an oversized answer is dropped before encoding. The
            # entry itself is not logged - it is the oversized payload.
            logger.warning(
                "Skipping VLM segmentation entry labelled %r: its polygon has "
                "%d vertices, above the limit of %d set by "
                "WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES",
                get_detection_class_name(entry),
                len(polygon),
                WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES,
            )
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0.0, upload_width) * scale_x
        polygon[:, 1] = np.clip(polygon[:, 1], 0.0, upload_height) * scale_y
        vertices = polygon.round()
        if _polygon_area(vertices) < MINIMUM_POLYGON_AREA:
            # Collinear, repeated or clipped-away vertices enclose no area; the
            # serialiser would drop such an instance silently, so drop it here
            # where it can be logged.
            logger.warning(
                "Skipping VLM segmentation entry whose polygon encloses no area: %r",
                entry,
            )
            continue
        valid.append((entry, vertices))

    if entries and not valid:
        # Every entry was skipped: the model answered in a shape other than
        # the prompted one. An empty prediction would be indistinguishable
        # from "nothing found", so fail and let the caller surface
        # `error_status=True`.
        raise ValueError(
            f"none of {len(entries)} segmentation entries carried a usable polygon"
        )

    xyxy, rle_masks, class_id, class_name = [], [], [], []
    for entry, vertices in valid:
        rle = polygon_to_rle(
            vertices, image_width=image_width, image_height=image_height
        )
        if mask_utils.area(rle) == 0:
            logger.warning(
                "Skipping VLM segmentation entry whose polygon covers no pixels: %r",
                entry,
            )
            continue
        x, y, w, h = mask_utils.toBbox(rle)
        xyxy.append([x, y, x + w, y + h])
        rle_masks.append(rle)
        label = get_detection_class_name(entry)
        class_id.append(class_name2id.get(label, -1))
        class_name.append(label)

    if valid and not xyxy:
        raise ValueError(
            f"none of {len(entries)} segmentation entries covered any pixels"
        )

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
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )


def polygon_to_rle(vertices: np.ndarray, image_width: int, image_height: int) -> dict:
    """Encode one ``(N, 2)`` pixel polygon as a COCO RLE without rasterising it.

    Args:
        vertices: Polygon vertices in original-image pixels.
        image_width: Original image width.
        image_height: Original image height.

    Returns:
        ``{"size": [height, width], "counts": bytes}`` as pycocotools emits it.
    """
    flat = [vertices.astype(float).ravel().tolist()]
    return mask_utils.frPyObjects(flat, image_height, image_width)[0]


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace area of an ``(N, 2)`` vertex array."""
    x, y = vertices[:, 0], vertices[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def extract_segmentation_entries(parsed: Any) -> List[dict]:
    """Pull the list of segmentation entries out of a parsed JSON payload.

    Accepts a bare list of entries, a ``{"segmentations": [...]}`` wrapper
    (or the detection wrapper key, which models fall back to), or a single
    bare entry. A non-empty list holding no entry objects at all is
    rejected so it surfaces as ``error_status`` instead of "no objects".

    Args:
        parsed: JSON payload extracted from the VLM output.

    Returns:
        List of raw segmentation entry dicts.

    Raises:
        ValueError: If the payload matches none of the accepted shapes.
    """
    if isinstance(parsed, list):
        return _entries_from_list(parsed)
    if isinstance(parsed, dict):
        for wrapper_key in (SEGMENTATIONS_WRAPPER_KEY, DETECTIONS_WRAPPER_KEY):
            entries = parsed.get(wrapper_key)
            if isinstance(entries, list):
                return _entries_from_list(entries)
        if any(key in parsed for key in POLYGON_KEYS):
            return [parsed]
    raise ValueError("Unexpected instance segmentation response format")


def _entries_from_list(items: list) -> List[dict]:
    entries = [entry for entry in items if isinstance(entry, dict)]
    if items and not entries:
        raise ValueError("Instance segmentation response is a list without any entries")
    return entries


def read_polygon(entry: dict) -> Optional[np.ndarray]:
    """Read the polygon of an entry as an ``(N, 2)`` float array.

    The prompt asks for a flat ``[x1, y1, x2, y2, ...]`` list; ``[[x, y], ...]``
    pairs and ``[{"x": .., "y": ..}, ...]`` points are accepted too, since
    models drift between the three. Anything with fewer than three
    vertices, an odd flat length or a non-numeric coordinate is rejected.

    Args:
        entry: Raw segmentation entry.

    Returns:
        Vertices in the uploaded image's pixel frame, or ``None``.
    """
    for key in POLYGON_KEYS:
        raw = entry.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        vertices = _read_vertices(raw)
        if vertices is not None and len(vertices) >= MINIMUM_POLYGON_VERTICES:
            return np.array(vertices, dtype=float)
    return None


def _read_vertices(raw: list) -> Optional[List[List[float]]]:
    if all(isinstance(item, dict) for item in raw):
        pairs = [
            (_read_number(item.get("x")), _read_number(item.get("y"))) for item in raw
        ]
    elif all(isinstance(item, list) for item in raw):
        if any(len(item) != 2 for item in raw):
            return None
        pairs = [(_read_number(item[0]), _read_number(item[1])) for item in raw]
    else:
        if len(raw) % 2:
            return None
        values = [_read_number(value) for value in raw]
        pairs = list(zip(values[0::2], values[1::2]))
    if any(x is None or y is None for x, y in pairs):
        return None
    return [[x, y] for x, y in pairs]
