from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    HEIGHT_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
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
    api_key: Optional[str] = None,
) -> str:
    if api_key:
        inference_request.api_key = api_key
    version_id_field = f"{core_model}_version_id"
    core_model_id = (
        f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
    )
    model_manager.add_model(core_model_id, inference_request.api_key)
    return core_model_id


def attach_prediction_type_info(
    results: List[Dict[str, Any]],
    prediction_type: str,
    key: str = "prediction_type",
) -> List[Dict[str, Any]]:
    for result in results:
        result[key] = prediction_type
    return results


def attach_parent_info(
    image: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    nested_key: Optional[str] = "predictions",
) -> List[Dict[str, Any]]:
    return [
        attach_parent_info_to_image_detections(
            image=i, predictions=p, nested_key=nested_key
        )
        for i, p in zip(image, results)
    ]


def attach_parent_info_to_image_detections(
    image: Dict[str, Any],
    predictions: Dict[str, Any],
    nested_key: Optional[str],
) -> Dict[str, Any]:
    predictions["parent_id"] = image["parent_id"]
    if nested_key is None:
        return predictions
    for prediction in predictions[nested_key]:
        prediction["parent_id"] = image["parent_id"]
    return predictions


def anchor_detections_in_parent_coordinates(
    image: List[Dict[str, Any]],
    serialised_result: List[Dict[str, Any]],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> List[Dict[str, Any]]:
    return [
        anchor_image_detections_in_parent_coordinates(
            image=i,
            serialised_result=d,
            image_metadata_key=image_metadata_key,
            detections_key=detections_key,
        )
        for i, d in zip(image, serialised_result)
    ]


def anchor_image_detections_in_parent_coordinates(
    image: Dict[str, Any],
    serialised_result: Dict[str, Any],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> Dict[str, Any]:
    serialised_result[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"] = deepcopy(
        serialised_result[detections_key]
    )
    serialised_result[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = deepcopy(
        serialised_result[image_metadata_key]
    )
    if ORIGIN_COORDINATES_KEY not in image:
        return serialised_result
    shift_x, shift_y = (
        image[ORIGIN_COORDINATES_KEY][CENTER_X_KEY],
        image[ORIGIN_COORDINATES_KEY][CENTER_Y_KEY],
    )
    parent_left_top_x = round(shift_x - image[ORIGIN_COORDINATES_KEY][WIDTH_KEY] / 2)
    parent_left_top_y = round(shift_y - image[ORIGIN_COORDINATES_KEY][HEIGHT_KEY] / 2)
    for detection in serialised_result[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"]:
        detection["x"] += parent_left_top_x
        detection["y"] += parent_left_top_y
        for point in detection.get("points", []):
            point["x"] += parent_left_top_x
            point["y"] += parent_left_top_y
        for point in detection.get("keypoints", []):
            point["x"] += parent_left_top_x
            point["y"] += parent_left_top_y
    serialised_result[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = image[
        ORIGIN_COORDINATES_KEY
    ][ORIGIN_SIZE_KEY]
    return serialised_result


def filter_out_unwanted_classes(
    serialised_result: List[Dict[str, Any]],
    classes_to_accept: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if classes_to_accept is None:
        return serialised_result
    classes_to_accept = set(classes_to_accept)
    results = []
    for image_result in serialised_result:
        filtered_image_result = deepcopy(image_result)
        filtered_image_result["predictions"] = []
        for prediction in image_result["predictions"]:
            if prediction["class"] not in classes_to_accept:
                continue
            filtered_image_result["predictions"].append(prediction)
        results.append(filtered_image_result)
    return results


def extract_origin_size_from_images(
    input_images: List[Union[dict, np.ndarray]],
    decoded_images: List[np.ndarray],
) -> List[Dict[str, int]]:
    result = []
    for input_image, decoded_image in zip(input_images, decoded_images):
        if (
            issubclass(type(input_image), dict)
            and ORIGIN_COORDINATES_KEY in input_image
        ):
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
