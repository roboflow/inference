"""Turn a VLM object-detection answer into ``sv.Detections``."""

from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.core_steps.common.vlm_decoding.detection_formats import (
    extract_detection_entries,
    get_detection_box_format,
    get_detection_class_name,
    get_detection_confidence,
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
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

PREDICTION_TYPE = "object-detection"


def decode_object_detections(
    raw_output: str,
    box_format: str,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
    upload_width: Optional[int] = None,
    upload_height: Optional[int] = None,
) -> Tuple[bool, Optional[sv.Detections]]:
    """Decode a raw VLM answer into detections in original-image pixels.

    Never raises: any failure is reported through ``error_status`` and
    logged, so a malformed model answer cannot take down a workflow run.

    Args:
        raw_output: Raw string produced by the model.
        box_format: Registered box coordinate format the prompt asked for.
        image: Workflow image the detections refer to (original resolution).
        classes: Class names used to map labels onto class ids; labels
            outside the list are kept with ``class_id == -1``.
        inference_id: Identifier attached to every parsed detection.
        upload_width: Width of the image as uploaded, required by formats
            using absolute pixel coordinates.
        upload_height: Height of the image as uploaded, same requirement.

    Returns:
        Tuple of ``(error_status, detections)``; ``detections`` is ``None``
        when ``error_status`` is ``True``.
    """
    error_status, parsed_data = extract_json(raw_output)
    if error_status:
        return True, None
    try:
        detections = build_detections(
            parsed_data=parsed_data,
            box_format=box_format,
            image=image,
            classes=classes,
            inference_id=inference_id,
            upload_width=upload_width,
            upload_height=upload_height,
        )
        return False, detections
    except Exception as error:
        logger.warning(
            "Could not decode VLM object-detection output for box format %s. "
            "Error type: %s. Details: %s",
            box_format,
            error.__class__.__name__,
            error,
        )
        return True, None


def build_detections(
    parsed_data: object,
    box_format: str,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
    upload_width: Optional[int] = None,
    upload_height: Optional[int] = None,
) -> sv.Detections:
    """Build ``sv.Detections`` from an already-parsed JSON payload.

    Args:
        parsed_data: JSON payload extracted from the VLM output.
        box_format: Registered box coordinate format.
        image: Workflow image the detections refer to.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to every parsed detection.
        upload_width: Width of the image as uploaded, for absolute formats.
        upload_height: Height of the image as uploaded, for absolute formats.

    Returns:
        Detections in the original image's coordinate space.

    Raises:
        ValueError: If the format is unknown, the payload shape is not
            recognised, required upload dimensions are missing, or the model
            answered in a different coordinate contract - i.e. NO entry
            matched the configured format (a partial mismatch keeps skipping
            the offending entries).
    """
    detection_format = get_detection_box_format(box_format)
    if classes is None:
        raise ValueError("Class list is required to decode object detections")
    entries = extract_detection_entries(parsed_data)
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image._read_shape_without_materialization()

    xyxy, class_id, class_name, confidence = [], [], [], []
    for entry in entries:
        box = detection_format.to_pixel_xyxy(
            entry=entry,
            image_width=image_width,
            image_height=image_height,
            upload_width=upload_width,
            upload_height=upload_height,
        )
        if box is None:
            logger.warning(
                "Skipping VLM detection entry without a well-formed %s box: %r",
                box_format,
                entry,
            )
            continue
        xyxy.append(box)
        label = get_detection_class_name(entry)
        class_id.append(class_name2id.get(label, -1))
        class_name.append(label)
        confidence.append(get_detection_confidence(entry))

    if entries and not xyxy:
        # Every entry was skipped: the model answered in a coordinate contract
        # other than the configured one. Reporting that as an empty prediction
        # is indistinguishable from "nothing detected", so fail instead and let
        # `decode_object_detections` surface `error_status=True`.
        raise ValueError(
            f"none of {len(entries)} detection entries matched box format "
            f"{box_format}"
        )

    xyxy = np.array(xyxy).round(0) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    detection_ids = np.array([str(uuid4()) for _ in range(len(xyxy))])
    dimensions = np.array([[image_height, image_width]] * len(xyxy))
    inference_ids = np.array([inference_id] * len(xyxy))
    prediction_type = np.array([PREDICTION_TYPE] * len(xyxy))
    data = {
        CLASS_NAME_DATA_FIELD: class_name,
        IMAGE_DIMENSIONS_KEY: dimensions,
        INFERENCE_ID_KEY: inference_ids,
        DETECTION_ID_KEY: detection_ids,
        PREDICTION_TYPE_KEY: prediction_type,
    }
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=None,
        tracker_id=None,
        data=data,
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )
