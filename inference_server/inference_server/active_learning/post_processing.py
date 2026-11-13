"""Rescaling and encoding of predictions registered as annotations."""

import json
from typing import List, Tuple

from inference_server.active_learning.entities import (
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
    Prediction,
    PredictionFileType,
    PredictionFormatNotSupported,
    PredictionType,
    SerialisedPrediction,
)


def adjust_prediction_to_client_scaling_factor(
    prediction: dict, scaling_factor: float, prediction_type: PredictionType
) -> dict:
    """Rescale a prediction to the size of the image that gets registered.

    The prediction is modified in place.

    Args:
        prediction: Prediction in the response format of the server.
        scaling_factor: Ratio of the registered image height to the original.
        prediction_type: Task type of the model.

    Returns:
        The same prediction object, rescaled when the factor differs from 1.
    """
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
    """Tell whether a prediction carries no coordinates to rescale.

    Args:
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.

    Returns:
        True for classification, stub and empty predictions.
    """
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
    """Rescale the boxes of detections in place.

    Args:
        predictions: Detections with ``x``, ``y``, ``width`` and ``height``.
        scaling_factor: Factor the coordinates are divided by.

    Returns:
        The rescaled detections.
    """
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
    """Rescale the boxes and the points of detections in place.

    Args:
        predictions: Detections with a box and a list of points.
        scaling_factor: Factor the coordinates are divided by.
        points_key: Key of the points list in each detection.

    Returns:
        The rescaled detections.
    """
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
    """Rescale one box in place.

    Args:
        bbox: Mapping with ``x``, ``y``, ``width`` and ``height``.
        scaling_factor: Factor the coordinates are divided by.

    Returns:
        The same mapping, rescaled.
    """
    bbox["x"] = bbox["x"] / scaling_factor
    bbox["y"] = bbox["y"] / scaling_factor
    bbox["width"] = bbox["width"] / scaling_factor
    bbox["height"] = bbox["height"] / scaling_factor

    return bbox


def adjust_points_coordinates_to_client_scaling_factor(
    points: List[dict],
    scaling_factor: float,
) -> List[dict]:
    """Rescale points in place.

    Args:
        points: Mappings with ``x`` and ``y``.
        scaling_factor: Factor the coordinates are divided by.

    Returns:
        The rescaled points.
    """
    result = []
    for point in points:
        point["x"] = point["x"] / scaling_factor
        point["y"] = point["y"] / scaling_factor
        result.append(point)

    return result


def prediction_has_supported_annotation_format(
    prediction: Prediction,
    prediction_type: PredictionType,
) -> bool:
    """Tell whether a prediction can be encoded as an annotation.

    Args:
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.

    Returns:
        False for a classification prediction without a ``top`` class, which is
        what a multi-label prediction looks like; True otherwise.
    """
    return CLASSIFICATION_TASK not in prediction_type or "top" in prediction


def encode_prediction(
    prediction: Prediction,
    prediction_type: PredictionType,
) -> Tuple[SerialisedPrediction, PredictionFileType]:
    """Encode a prediction as the annotation file the platform accepts.

    Args:
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.

    Returns:
        The annotation content and its file type, ``json`` or ``txt``.

    Raises:
        PredictionFormatNotSupported: If the prediction is a classification
            without a ``top`` class.
    """
    if CLASSIFICATION_TASK not in prediction_type:
        return json.dumps(prediction), "json"

    if "top" in prediction:
        return prediction["top"], "txt"

    raise PredictionFormatNotSupported(
        "Prediction type or prediction format not supported."
    )
