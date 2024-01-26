import json
from typing import List, Tuple

from inference.core.active_learning.entities import (
    Prediction,
    PredictionFileType,
    PredictionType,
    SerialisedPrediction,
)
from inference.core.constants import (
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
)
from inference.core.exceptions import PredictionFormatNotSupported


def adjust_prediction_to_client_scaling_factor(
    prediction: dict, scaling_factor: float, prediction_type: PredictionType
) -> dict:
    if abs(scaling_factor - 1.0) < 1e-5:
        return prediction
    if "image" in prediction:
        prediction["image"] = {
            "width": round(prediction["image"]["width"] / scaling_factor),
            "height": round(prediction["image"]["height"] / scaling_factor),
        }
    if predictions_should_not_be_post_processed(
        prediction=prediction, prediction_type=prediction_type
    ):
        return prediction
    if prediction_type == INSTANCE_SEGMENTATION_TASK:
        prediction["predictions"] = (
            adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
                points_key="points",
            )
        )
    if prediction_type == OBJECT_DETECTION_TASK:
        prediction["predictions"] = (
            adjust_object_detection_predictions_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
            )
        )
    return prediction


def predictions_should_not_be_post_processed(
    prediction: dict, prediction_type: PredictionType
) -> bool:
    # excluding from post-processing classification output, stub-output and empty predictions
    return (
        "is_stub" in prediction
        or "predictions" not in prediction
        or CLASSIFICATION_TASK in prediction_type
        or len(prediction["predictions"]) == 0
    )


def adjust_object_detection_predictions_to_client_scaling_factor(
    predictions: List[dict],
    scaling_factor: float,
) -> List[dict]:
    result = []
    for prediction in predictions:
        prediction = adjust_bbox_coordinates_to_client_scaling_factor(
            bbox=prediction,
            scaling_factor=scaling_factor,
        )
        result.append(prediction)
    return result


def adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
    predictions: List[dict],
    scaling_factor: float,
    points_key: str,
) -> List[dict]:
    result = []
    for prediction in predictions:
        prediction = adjust_bbox_coordinates_to_client_scaling_factor(
            bbox=prediction,
            scaling_factor=scaling_factor,
        )
        prediction[points_key] = adjust_points_coordinates_to_client_scaling_factor(
            points=prediction[points_key],
            scaling_factor=scaling_factor,
        )
        result.append(prediction)
    return result


def adjust_bbox_coordinates_to_client_scaling_factor(
    bbox: dict,
    scaling_factor: float,
) -> dict:
    bbox["x"] = bbox["x"] / scaling_factor
    bbox["y"] = bbox["y"] / scaling_factor
    bbox["width"] = bbox["width"] / scaling_factor
    bbox["height"] = bbox["height"] / scaling_factor
    return bbox


def adjust_points_coordinates_to_client_scaling_factor(
    points: List[dict],
    scaling_factor: float,
) -> List[dict]:
    result = []
    for point in points:
        point["x"] = point["x"] / scaling_factor
        point["y"] = point["y"] / scaling_factor
        result.append(point)
    return result


def encode_prediction(
    prediction: Prediction,
    prediction_type: PredictionType,
) -> Tuple[SerialisedPrediction, PredictionFileType]:
    if CLASSIFICATION_TASK not in prediction_type:
        return json.dumps(prediction), "json"
    if "top" in prediction:
        return prediction["top"], "txt"
    raise PredictionFormatNotSupported(
        f"Prediction type or prediction format not supported."
    )
