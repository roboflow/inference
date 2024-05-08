from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.constants import (
    HEIGHT_KEY,
    LEFT_TOP_X_KEY,
    LEFT_TOP_Y_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
    PARENT_ID_KEY,
    WIDTH_KEY,
)


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
    key: str = "prediction_type",
) -> List[Dict[str, Any]]:
    for result in predictions:
        result[key] = prediction_type
    return predictions


def attach_parent_info(
    images: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    nested_key: Optional[str] = "predictions",
) -> List[Dict[str, Any]]:
    return [
        attach_parent_info_to_prediction(
            image=image, prediction=prediction, nested_key=nested_key
        )
        for image, prediction in zip(images, predictions)
    ]


def attach_parent_info_to_prediction(
    image: Dict[str, Any],
    prediction: Dict[str, Any],
    nested_key: Optional[str],
) -> Dict[str, Any]:
    prediction[PARENT_ID_KEY] = image[PARENT_ID_KEY]
    if nested_key is None:
        return prediction
    for detection in prediction[nested_key]:
        detection[PARENT_ID_KEY] = image[PARENT_ID_KEY]
    return prediction


def anchor_prediction_detections_in_parent_coordinates(
    image: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> List[Dict[str, Any]]:
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
    prediction: Dict[str, Any],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> Dict[str, Any]:
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
    for detection in prediction[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"]:
        detection["x"] += shift_x
        detection["y"] += shift_y
        for point in detection.get("points", []):
            point["x"] += shift_x
            point["y"] += shift_y
        for point in detection.get("keypoints", []):
            point["x"] += shift_x
            point["y"] += shift_y
    prediction[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = image[
        ORIGIN_COORDINATES_KEY
    ][ORIGIN_SIZE_KEY]
    return prediction


def filter_out_unwanted_classes_from_predictions_detections(
    predictions: List[Dict[str, Any]],
    classes_to_accept: Optional[List[str]],
    detections_key: str = "predictions",
    class_name_key: str = "class",
) -> List[Dict[str, Any]]:
    if classes_to_accept is None:
        return predictions
    classes_to_accept = set(classes_to_accept)
    results = []
    for prediction in predictions:
        filtered_image_result = deepcopy(prediction)
        filtered_image_result[detections_key] = [
            prediction
            for prediction in prediction[detections_key]
            if prediction[class_name_key] in classes_to_accept
        ]
        results.append(filtered_image_result)
    return results


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


def detection_to_xyxy(detection: dict) -> Tuple[int, int, int, int]:
    x_min = round(detection["x"] - detection[WIDTH_KEY] / 2)
    y_min = round(detection["y"] - detection[HEIGHT_KEY] / 2)
    x_max = round(x_min + detection[WIDTH_KEY])
    y_max = round(y_min + detection[HEIGHT_KEY])
    return x_min, y_min, x_max, y_max
