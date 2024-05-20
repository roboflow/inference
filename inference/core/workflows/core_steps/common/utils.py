import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    KEYPOINTS_CLASS_ID_KEY,
    KEYPOINTS_CLASS_NAME_KEY,
    KEYPOINTS_CONFIDENCE_KEY,
    KEYPOINTS_KEY,
    KEYPOINTS_XY_KEY,
    LEFT_TOP_X_KEY,
    LEFT_TOP_Y_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
    PARENT_ID_KEY,
    WIDTH_KEY, ROOT_PARENT_COORDINATES_KEY, ROOT_PARENT_DIMENSIONS_KEY, PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY, PREDICTION_TYPE_KEY, ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.entities.base import Batch, WorkflowImageData, OriginCoordinatesSystem, \
    ParentImageMetadata


def load_core_model(
    model_manager: ModelManager,
    inference_request: Union[
        DoctrOCRInferenceRequest,
        ClipCompareRequest,
        CogVLMInferenceRequest,
        YOLOWorldInferenceRequest,
    ],
    core_model: str,
) -> str:
    version_id_field = f"{core_model}_version_id"
    core_model_id = (
        f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
    )
    model_manager.add_model(core_model_id, inference_request.api_key)
    return core_model_id


def attach_prediction_type_info(
    predictions: List[Dict[str, Any]],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[Dict[str, Any]]:
    for result in predictions:
        result[key] = prediction_type
    return predictions


def attach_prediction_type_info_to_sv_detections(
    predictions: List[sv.Detections],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[sv.Detections]:
    for prediction in predictions:
        prediction[key] = np.array([prediction_type] * len(prediction))
    return predictions


def convert_to_sv_detections(
    predictions: List[Dict[str, Union[List[Dict[str, Any]], Any]]],
    predictions_key: str = "predictions",
) -> List[sv.Detections]:
    batch_of_detections: List[sv.Detections] = []
    for p in predictions:
        detections = sv.Detections.from_inference(p)
        parent_ids = [d.get(PARENT_ID_KEY, "") for d in p[predictions_key]]
        detection_ids = [
            d.get(DETECTION_ID_KEY, str(uuid.uuid4)) for d in p[predictions_key]
        ]
        detections[DETECTION_ID_KEY] = np.array(detection_ids)
        detections[PARENT_ID_KEY] = np.array(parent_ids)
        batch_of_detections.append(detections)
    return batch_of_detections


def add_keypoints_to_detections(
    predictions: List[Dict[str, Union[List[Dict[str, Any]], Any]]],
    detections: sv.Detections,
):
    keypoints_class_names = []
    keypoints_class_ids = []
    keypoints_confidences = []
    keypoints_xy = []
    for p in predictions:
        keypoints = p.get(KEYPOINTS_KEY, [])
        keypoints_class_names.append(
            np.array([k[KEYPOINTS_CLASS_NAME_KEY] for k in keypoints])
        )
        keypoints_class_ids.append(
            np.array([k[KEYPOINTS_CLASS_ID_KEY] for k in keypoints])
        )
        keypoints_confidences.append(
            np.array(
                [k[KEYPOINTS_CONFIDENCE_KEY] for k in keypoints],
                dtype=np.float64,
            )
        )
        keypoints_xy.append(
            np.array([k[KEYPOINTS_XY_KEY] for k in keypoints], dtype=np.float64)
        )
    detections[KEYPOINTS_CLASS_NAME_KEY] = np.array(
        keypoints_class_names, dtype="object"
    )
    detections[KEYPOINTS_CLASS_ID_KEY] = np.array(keypoints_class_ids, dtype="object")
    detections[KEYPOINTS_CONFIDENCE_KEY] = np.array(
        keypoints_confidences, dtype="object"
    )
    detections[KEYPOINTS_XY_KEY] = np.array(keypoints_xy, dtype="object")


def attach_parent_info(
    images: List[Dict[str, Any]],
    predictions: List[Dict[str, Union[sv.Detections, Any]]],
    nested_key: Optional[str] = "predictions",
) -> List[Dict[str, Union[sv.Detections, Any]]]:
    for image, prediction in zip(images, predictions):
        prediction[PARENT_ID_KEY] = image[PARENT_ID_KEY]
        if nested_key is None:
            continue
        detections = prediction[nested_key]
        detections[PARENT_ID_KEY] = [image[PARENT_ID_KEY]] * len(detections)
    return predictions


def attach_parents_coordinates_to_list_of_detections(
    predictions: List[sv.Detections],
    images: List[WorkflowImageData],
) -> List[sv.Detections]:
    result = []
    for prediction, image in zip(predictions, images):
        result.append(attach_parents_coordinates_to_detections(
            detections=prediction,
            image=image,
        ))
    return result


def attach_parents_coordinates_to_detections(
    detections: sv.Detections,
    image: WorkflowImageData,
) -> sv.Detections:
    detections = attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=image.workflow_root_ancestor_metadata,
        parent_id_key=ROOT_PARENT_ID_KEY,
        coordinates_key=ROOT_PARENT_COORDINATES_KEY,
        dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
    )
    return attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=image.parent_metadata,
        parent_id_key=PARENT_ID_KEY,
        coordinates_key=PARENT_COORDINATES_KEY,
        dimensions_key=PARENT_DIMENSIONS_KEY,
    )


def attach_parent_coordinates_to_detections(
    detections: sv.Detections,
    parent_metadata: ParentImageMetadata,
    parent_id_key: str,
    coordinates_key: str,
    dimensions_key: str,
) -> sv.Detections:
    parent_coordinates_system = parent_metadata.origin_coordinates
    detections[parent_id_key] = np.array([parent_metadata.parent_id] * len(detections))
    coordinates = np.array([
        [parent_coordinates_system.left_top_x, parent_coordinates_system.left_top_y]
    ] * len(detections))
    detections[coordinates_key] = coordinates
    dimensions = np.array([
        [parent_coordinates_system.origin_height, parent_coordinates_system.origin_width]
    ] * len(detections))
    detections[dimensions_key] = dimensions
    return detections


def anchor_prediction_detections_in_parent_coordinates(
    image: List[Dict[str, Any]],
    predictions: List[Dict[str, Union[sv.Detections, Any]]],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> List[Dict[str, Union[sv.Detections, Any]]]:
    return [
        anchor_detections_in_parent_coordinates(
            image=image,
            prediction=prediction,
            image_metadata_key=image_metadata_key,
            detections_key=detections_key,
        )
        for image, prediction in zip(image, predictions)
    ]


def anchor_detections_in_parent_coordinates(
    image: Dict[str, Any],
    prediction: Dict[str, Union[sv.Detections, Any]],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
    keypoints_key: str = KEYPOINTS_KEY,
) -> Dict[str, Union[sv.Detections, Any]]:
    prediction[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"] = deepcopy(
        prediction[detections_key]
    )
    prediction[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = deepcopy(
        prediction[image_metadata_key]
    )
    if ORIGIN_COORDINATES_KEY not in image:
        return prediction
    shift_x, shift_y = (
        image[ORIGIN_COORDINATES_KEY][LEFT_TOP_X_KEY],
        image[ORIGIN_COORDINATES_KEY][LEFT_TOP_Y_KEY],
    )
    anchored_detections: sv.Detections = prediction[
        f"{detections_key}{PARENT_COORDINATES_SUFFIX}"
    ]
    anchored_detections.xyxy += [shift_x, shift_y, shift_x, shift_y]
    # TODO: assumed type
    if keypoints_key in anchored_detections.data:
        anchored_detections[keypoints_key] += [shift_x, shift_y]
    if anchored_detections.mask:
        origin_width = image[ORIGIN_COORDINATES_KEY][ORIGIN_SIZE_KEY][WIDTH_KEY]
        origin_height = image[ORIGIN_COORDINATES_KEY][ORIGIN_SIZE_KEY][HEIGHT_KEY]
        origin_mask_base = np.full((origin_height, origin_width), False)
        anchored_detections.mask = [
            origin_mask_base.copy() for _ in anchored_detections
        ]
        for anchored_mask, original_mask in zip(
            anchored_detections.mask, prediction[detections_key].mask
        ):
            mask_h, mask_w = original_mask.shape
            # TODO: instead of shifting mask we could store contours in data instead of storing mask (even if calculated)
            #       it would be faster to shift contours but at expense of having to remember to generate mask from contour when it's needed
            anchored_mask[shift_x : shift_x + mask_w, shift_y : shift_y + mask_h] = (
                original_mask
            )
    prediction[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = image[
        ORIGIN_COORDINATES_KEY
    ][ORIGIN_SIZE_KEY]
    return prediction


def filter_out_unwanted_classes_from_predictions_detections(
    predictions: List[Dict[str, Union[sv.Detections, Any]]],
    classes_to_accept: Optional[List[str]],
    detections_key: str = "predictions",
    class_name_key: str = "class_name",
) -> List[Dict[str, Union[sv.Detections, Any]]]:
    if not classes_to_accept:
        return predictions
    filtered_predictions = []
    for prediction in predictions:
        detections = prediction[detections_key]
        prediction[detections_key] = detections[
            np.isin(detections[class_name_key], classes_to_accept)
        ]
        filtered_predictions.append(prediction)
    return predictions


def filter_out_unwanted_classes_from_sv_detections(
    predictions: List[sv.Detections],
    classes_to_accept: Optional[List[str]],
) -> List[sv.Detections]:
    if not classes_to_accept:
        return predictions
    filtered_predictions = []
    for prediction in predictions:
        filtered_prediction = prediction[
            np.isin(prediction[CLASS_NAME_DATA_FIELD], classes_to_accept)
        ]
        filtered_predictions.append(filtered_prediction)
    return filtered_predictions


def extract_origin_size_from_images_batch(
    input_images: List[Union[dict, np.ndarray]],
    decoded_images: List[np.ndarray],
) -> List[Dict[str, int]]:
    result = []
    for input_image, decoded_image in zip(input_images, decoded_images):
        if isinstance(input_image, dict) and ORIGIN_COORDINATES_KEY in input_image:
            result.append(input_image[ORIGIN_COORDINATES_KEY][ORIGIN_SIZE_KEY])
        else:
            result.append(
                {HEIGHT_KEY: decoded_image.shape[0], WIDTH_KEY: decoded_image.shape[1]}
            )
    return result


# TODO: remove once fusion is migrated
def detection_to_xyxy(detection: dict) -> Tuple[int, int, int, int]:
    x_min = round(detection["x"] - detection[WIDTH_KEY] / 2)
    y_min = round(detection["y"] - detection[HEIGHT_KEY] / 2)
    x_max = round(x_min + detection[WIDTH_KEY])
    y_max = round(y_min + detection[HEIGHT_KEY])
    return x_min, y_min, x_max, y_max


def grab_batch_parameters(
    operations_parameters: Dict[str, Any],
    predictions: Batch[Optional[sv.Detections]],
) -> Dict[str, Any]:
    return {
        key: value.broadcast(n=len(predictions))
        for key, value in operations_parameters.items()
        if isinstance(value, Batch)
    }


def grab_non_batch_parameters(operations_parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in operations_parameters.items()
        if not isinstance(value, Batch)
    }
