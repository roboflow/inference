from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import supervision as sv

from inference.core import logger
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_INFERENCE_RESPONSE,
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_INFERENCE_RESPONSE,
    AREA_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CONFIDENCE_KEY,
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PARENT_ORIGIN_KEY,
    PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY,
    POLYGON_KEY_IN_INFERENCE_RESPONSE,
    POLYGON_KEY_IN_SV_DETECTIONS,
    RLE_MASK_KEY_IN_INFERENCE_RESPONSE,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    ROOT_PARENT_ORIGIN_KEY,
    SMOOTHED_SPEED_KEY_IN_INFERENCE_RESPONSE,
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_INFERENCE_RESPONSE,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_INFERENCE_RESPONSE,
    SPEED_KEY_IN_SV_DETECTIONS,
    TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
    TRACKER_ID_KEY,
    VELOCITY_KEY_IN_INFERENCE_RESPONSE,
    VELOCITY_KEY_IN_SV_DETECTIONS,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ParentOrigin,
    VideoMetadata,
    WorkflowImageData,
)

MIN_SECRET_LENGTH_TO_REVEAL_PREFIX = 8
MIN_POLYGON_POINT_COUNT = 3


def serialise_sv_detections(detections: sv.Detections) -> dict:
    serialized_detections = []
    image_dimensions = None
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        detection_dict = {}
        image_dimensions = data.get(IMAGE_DIMENSIONS_KEY)

        # Avoid converting the whole xyxy array to a python list when unnecessary.
        # Extract numeric components directly which handles both sequences and numpy arrays.
        x1 = float(xyxy[0])
        y1 = float(xyxy[1])
        x2 = float(xyxy[2])
        y2 = float(xyxy[3])

        detection_dict[WIDTH_KEY] = abs(x2 - x1)
        detection_dict[HEIGHT_KEY] = abs(y2 - y1)
        detection_dict[X_KEY] = x1 + detection_dict[WIDTH_KEY] / 2
        detection_dict[Y_KEY] = y1 + detection_dict[HEIGHT_KEY] / 2

        detection_dict[CONFIDENCE_KEY] = float(confidence)
        detection_dict[CLASS_ID_KEY] = int(class_id)
        if mask is not None:
            if (
                POLYGON_KEY_IN_SV_DETECTIONS in data
                and data[POLYGON_KEY_IN_SV_DETECTIONS] is not None
                and len(data[POLYGON_KEY_IN_SV_DETECTIONS]) > 2
            ):
                polygon = data[POLYGON_KEY_IN_SV_DETECTIONS]
            else:
                polygon = mask_to_polygon(mask=mask)
            if polygon is None:
                # ignoring the whole instance
                continue
            # Use list comprehension to build polygon point dicts more efficiently.
            detection_dict[POLYGON_KEY] = [
                {X_KEY: float(x), Y_KEY: float(y)} for x, y in polygon
            ]
        if tracker_id is not None:
            detection_dict[TRACKER_ID_KEY] = int(tracker_id)
        detection_dict[CLASS_NAME_KEY] = str(data["class_name"])
        detection_dict[DETECTION_ID_KEY] = str(data[DETECTION_ID_KEY])
        if PATH_DEVIATION_KEY_IN_SV_DETECTIONS in data:
            detection_dict[PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE] = data[
                PATH_DEVIATION_KEY_IN_SV_DETECTIONS
            ]
        if TIME_IN_ZONE_KEY_IN_SV_DETECTIONS in data:
            detection_dict[TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE] = data[
                TIME_IN_ZONE_KEY_IN_SV_DETECTIONS
            ]
        if POLYGON_KEY_IN_SV_DETECTIONS in data:
            detection_dict[POLYGON_KEY_IN_INFERENCE_RESPONSE] = (
                data[POLYGON_KEY_IN_SV_DETECTIONS]
                .astype(float)
                .round()
                .astype(int)
                .tolist()
            )
        if (
            BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS in data
        ):
            detection_dict[BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS
            ]

        if PARENT_ID_KEY in data:
            detection_dict[PARENT_ID_KEY] = str(data[PARENT_ID_KEY])

        # Add parent origin metadata if detection is based on a crop/slice
        if (
            PARENT_ID_KEY in data
            and ROOT_PARENT_ID_KEY in data
            and str(data[PARENT_ID_KEY]) != str(data[ROOT_PARENT_ID_KEY])
        ):
            _attach_parent_metadata_to_detection_dict(
                detection_dict=detection_dict,
                data=data,
                coordinates_key=PARENT_COORDINATES_KEY,
                dimensions_key=PARENT_DIMENSIONS_KEY,
                origin_key=PARENT_ORIGIN_KEY,
            )

            detection_dict[ROOT_PARENT_ID_KEY] = str(data[ROOT_PARENT_ID_KEY])

            _attach_parent_metadata_to_detection_dict(
                detection_dict=detection_dict,
                data=data,
                coordinates_key=ROOT_PARENT_COORDINATES_KEY,
                dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
                origin_key=ROOT_PARENT_ORIGIN_KEY,
            )

        if (
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in data
        ):
            kp_class_id = data[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS]
            kp_class_name = data[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS]
            kp_confidence = data[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS]
            kp_xy = data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
            # Build keypoints list using list comprehension and bounded length to match zip behaviour.
            n_kp = min(
                len(kp_class_id),
                len(kp_class_name),
                len(kp_confidence),
                len(kp_xy),
            )
            detection_dict[KEYPOINTS_KEY_IN_INFERENCE_RESPONSE] = [
                {
                    "class_id": int(kp_class_id[i]),
                    "class": str(kp_class_name[i]),
                    "confidence": float(kp_confidence[i]),
                    "x": float(kp_xy[i][0]),
                    "y": float(kp_xy[i][1]),
                }
                for i in range(n_kp)
            ]
        if DETECTED_CODE_KEY in data:
            detection_dict[DETECTED_CODE_KEY] = data[DETECTED_CODE_KEY]
        if VELOCITY_KEY_IN_SV_DETECTIONS in data:
            detection_dict[VELOCITY_KEY_IN_INFERENCE_RESPONSE] = data[
                VELOCITY_KEY_IN_SV_DETECTIONS
            ].tolist()
        if SPEED_KEY_IN_SV_DETECTIONS in data:
            detection_dict[SPEED_KEY_IN_INFERENCE_RESPONSE] = data[
                SPEED_KEY_IN_SV_DETECTIONS
            ].astype(float)
        if SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS in data:
            detection_dict[SMOOTHED_VELOCITY_KEY_IN_INFERENCE_RESPONSE] = data[
                SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS
            ].tolist()
        if SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS in data:
            detection_dict[SMOOTHED_SPEED_KEY_IN_INFERENCE_RESPONSE] = data[
                SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS
            ].astype(float)
        if AREA_KEY_IN_SV_DETECTIONS in data:
            detection_dict[AREA_KEY_IN_INFERENCE_RESPONSE] = float(
                data[AREA_KEY_IN_SV_DETECTIONS]
            )
        if AREA_CONVERTED_KEY_IN_SV_DETECTIONS in data:
            detection_dict[AREA_CONVERTED_KEY_IN_INFERENCE_RESPONSE] = float(
                data[AREA_CONVERTED_KEY_IN_SV_DETECTIONS]
            )
        serialized_detections.append(detection_dict)
    image_metadata = {
        "width": None,
        "height": None,
    }  # TODO: this breaks the contract of
    # standard inference, but to fix that problem, we would need sv.Detections to provide
    # detection-level metadata.
    if image_dimensions is not None:
        image_metadata = {
            "width": image_dimensions[1].item(),
            "height": image_dimensions[0].item(),
        }
    return {"image": image_metadata, "predictions": serialized_detections}


def _attach_parent_metadata_to_detection_dict(
    detection_dict: dict,
    data: Dict[str, Union[np.ndarray, list]],
    coordinates_key: str,
    dimensions_key: str,
    origin_key: str,
) -> None:
    if coordinates_key in data and dimensions_key in data:
        parent_coords = data[coordinates_key]
        parent_dims = data[dimensions_key]
        if isinstance(parent_coords, np.ndarray):
            parent_coords = parent_coords.astype(float).round().astype(int).tolist()
        if isinstance(parent_dims, np.ndarray):
            parent_dims = parent_dims.astype(float).round().astype(int).tolist()
        detection_dict[origin_key] = ParentOrigin(
            offset_x=parent_coords[0],
            offset_y=parent_coords[1],
            width=parent_dims[1],
            height=parent_dims[0],
        ).model_dump()


def mask_to_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    # masks here should be predicted by instance segmentation
    # model and in theory can only present SINGLE!!! shape
    # our response schema for InstanceSegmentationPrediction
    # (see `inference.core.entities.responses.inference.InstanceSegmentationPrediction`)
    # FROM THE BEGINNING were allowing to serialise single polygon that belongs to mask,
    # no hierarchy respected and multiple polygons are not taken into account
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        # nothing found - instance should be rejected
        return None
    if len(contours) > 1:
        logger.warning(
            f"Detected instance segmentation that has {len(contours)} in the mask, which by convention "
            "should not happen. We are taking the first polygon to avoid exceptions, but if you see this "
            "warning - it may indicate that some model producing instance segmentation result works under "
            "different assumptions that models historically added into inference and that should be a signal "
            "to figure out more generic representation for instance seg predictions."
        )
    contour = np.squeeze(contours[0], axis=1)
    contour_padding = max(MIN_POLYGON_POINT_COUNT - contour.shape[0], 0)
    if contour_padding > 0:
        padding = np.repeat(
            np.expand_dims(contour[-1], axis=0), repeats=contour_padding, axis=0
        )
        return np.append(contour, padding, axis=0)
    return contour


def serialise_image(image: WorkflowImageData) -> Dict[str, Any]:
    result = {
        "type": "base64",
        "value": image.base64_image,
        "video_metadata": image.video_metadata.dict() if image.video_metadata else None,
    }

    parent_metadata = image.parent_metadata
    root_metadata = image.workflow_root_ancestor_metadata

    # Add parent origin metadata if image is a crop/slice
    if parent_metadata.parent_id != root_metadata.parent_id:
        result[PARENT_ID_KEY] = parent_metadata.parent_id
        result[PARENT_ORIGIN_KEY] = ParentOrigin.from_origin_coordinates_system(
            parent_metadata.origin_coordinates
        ).model_dump()

        result[ROOT_PARENT_ID_KEY] = root_metadata.parent_id
        result[ROOT_PARENT_ORIGIN_KEY] = ParentOrigin.from_origin_coordinates_system(
            root_metadata.origin_coordinates
        ).model_dump()

    return result


def serialize_video_metadata_kind(video_metadata: VideoMetadata) -> dict:
    return video_metadata.dict()


def serialize_wildcard_kind(value: Any) -> Any:
    if isinstance(value, WorkflowImageData):
        value = serialise_image(image=value)
    elif isinstance(value, dict):
        value = serialize_dict(elements=value)
    elif isinstance(value, list):
        value = serialize_list(elements=value)
    elif isinstance(value, sv.Detections):
        value = serialise_sv_detections(detections=value)
    elif isinstance(value, datetime):
        value = serialize_timestamp(timestamp=value)
    return value


def serialize_list(elements: List[Any]) -> List[Any]:
    result = []
    for element in elements:
        element = serialize_wildcard_kind(value=element)
        result.append(element)
    return result


def serialize_dict(elements: Dict[str, Any]) -> Dict[str, Any]:
    serialized_result = {}
    for key, value in elements.items():
        value = serialize_wildcard_kind(value=value)
        serialized_result[key] = value
    return serialized_result


def serialize_secret(secret: str) -> str:
    if len(secret) < MIN_SECRET_LENGTH_TO_REVEAL_PREFIX:
        return "*" * MIN_SECRET_LENGTH_TO_REVEAL_PREFIX
    prefix = secret[:2]
    infix = "*" * MIN_SECRET_LENGTH_TO_REVEAL_PREFIX
    suffix = secret[-2:]
    return f"{prefix}{infix}{suffix}"


def serialize_timestamp(timestamp: datetime) -> str:
    return timestamp.isoformat()


def serialise_rle_sv_detections(detections: sv.Detections) -> dict:
    rle_masks = detections.data.get(RLE_MASK_KEY_IN_SV_DETECTIONS)
    if rle_masks is None:
        raise ValueError(
            "No RLE masks found in detections.data['rle_mask']. "
            "This serializer requires RLE masks to be present."
        )

    result = serialise_sv_detections(detections=detections)

    for idx, detection_dict in enumerate(result["predictions"]):
        detection_dict.pop(POLYGON_KEY, None)

        if idx < len(rle_masks):
            rle = rle_masks[idx]
            if isinstance(rle.get("counts"), bytes):
                rle = {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
            detection_dict[RLE_MASK_KEY_IN_INFERENCE_RESPONSE] = rle

    return result
