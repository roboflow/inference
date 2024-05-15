import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv

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


def convert_to_sv_detections(
    predictions: List[Dict[str, Any]],
    predictions_key: str = "predictions",
    keypoints_key: str = "keypoints",
    detection_id_key: str = "detection_id",
    parent_id_key: str = PARENT_ID_KEY,
) -> List[Dict[str, Union[sv.Detections, Any]]]:
    converted_predictions: List[Dict[str, Union[sv.Detections, Any]]] = []
    for p in predictions:
        converted_prediction = deepcopy(p)
        detections = sv.Detections.from_inference(p)
        if any(keypoints_key in d for d in p[predictions_key]):
            # keypoints arrays may have different length for each detection hence "object" type is used
            detections[keypoints_key] = np.array(
                [
                    np.array(
                        [
                            [keypoint["x"], keypoint["y"]]
                            for keypoint in d[keypoints_key]
                        ]
                    )
                    if d.get(keypoints_key)
                    else np.array([])
                    for d in p[predictions_key]
                ],
                dtype="object",
            )
        if any(parent_id_key in d for d in p[predictions_key]):
            detections[parent_id_key] = np.array(
                [
                    d[parent_id_key]
                    if d.get(parent_id_key)
                    else None
                    for d in p[predictions_key]
                ]
            )
        detection_ids = [
            d[detection_id_key] if detection_id_key in d else str(uuid.uuid4)
            for d in p[predictions_key]
        ]
        detections[detection_id_key] = np.array(detection_ids)
        converted_prediction[predictions_key] = detections
        converted_predictions.append(converted_prediction)
    return converted_predictions


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
    keypoints_key: str = "keypoints",
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
        anchored_detections.mask = [origin_mask_base.copy() for _ in anchored_detections]
        for anchored_mask, original_mask in zip(anchored_detections.mask, prediction[detections_key].mask):
            mask_h, mask_w = original_mask.shape
            # TODO: instead of shifting mask we could store contours in data instead of storing mask (even if calculated)
            #       it would be faster to shift contours but at expense of having to remember to generate mask from contour when it's needed
            anchored_mask[shift_x : shift_x + mask_w, shift_y : shift_y + mask_h] = original_mask
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
    if classes_to_accept is None:
        return predictions
    filtered_predictions = []
    for prediction in predictions:
        filtered_prediction = deepcopy(prediction)
        filtered_detections = filtered_prediction[detections_key]
        filtered_prediction[detections_key] = filtered_detections[
            np.isin(filtered_detections[class_name_key], classes_to_accept)
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
