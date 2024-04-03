from typing import Any, Dict, List, Optional

from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    HOSTED_CORE_MODEL_URL,
    HOSTED_DETECT_URL,
    HOSTED_INSTANCE_SEGMENTATION_URL,
)
from inference.enterprise.workflows.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    HEIGHT_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
    WIDTH_KEY,
)


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
    serialised_result[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = image[
        ORIGIN_COORDINATES_KEY
    ][ORIGIN_SIZE_KEY]
    return serialised_result


ROBOFLOW_MODEL2HOSTED_ENDPOINT = {
    "ClassificationModel": HOSTED_CLASSIFICATION_URL,
    "MultiLabelClassificationModel": HOSTED_CLASSIFICATION_URL,
    "ObjectDetectionModel": HOSTED_DETECT_URL,
    "KeypointsDetectionModel": HOSTED_DETECT_URL,
    "InstanceSegmentationModel": HOSTED_INSTANCE_SEGMENTATION_URL,
    "OCRModel": HOSTED_CORE_MODEL_URL,
    "ClipComparison": HOSTED_CORE_MODEL_URL,
    "YoloWorld": HOSTED_CORE_MODEL_URL,
    "YoloWorldModel": HOSTED_CORE_MODEL_URL,
}


def resolve_model_api_url(step: StepInterface) -> str:
    if WORKFLOWS_REMOTE_API_TARGET != "hosted":
        return LOCAL_INFERENCE_API_URL
    return ROBOFLOW_MODEL2HOSTED_ENDPOINT[step.get_type()]
