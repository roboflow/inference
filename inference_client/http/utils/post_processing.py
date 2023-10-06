import base64
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from requests import Response

from inference_client.http.entities import (
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
    TaskType,
    VisualisationResponseFormat,
)
from inference_client.http.utils.encoding import (
    bytes_to_opencv_image,
    bytes_to_pillow_image,
    encode_base_64,
)

CONTENT_TYPE_HEADERS = ["content-type", "Content-Type"]
IMAGES_TRANSCODING_METHODS = {
    VisualisationResponseFormat.BASE64: encode_base_64,
    VisualisationResponseFormat.NUMPY: bytes_to_opencv_image,
    VisualisationResponseFormat.PILLOW: bytes_to_pillow_image,
}


def response_contains_jpeg_image(response: Response) -> bool:
    content_type = None
    for header_name in CONTENT_TYPE_HEADERS:
        if header_name in response.headers:
            content_type = response.headers[header_name]
            break
    if content_type is None:
        return False
    return "image/jpeg" in content_type


def transform_base64_visualisation(
    visualisation: str,
    expected_format: VisualisationResponseFormat,
) -> Union[str, np.ndarray, Image.Image]:
    visualisation_bytes = base64.b64decode(visualisation)
    return transform_visualisation_bytes(
        visualisation=visualisation_bytes, expected_format=expected_format
    )


def transform_visualisation_bytes(
    visualisation: bytes,
    expected_format: VisualisationResponseFormat,
) -> Union[str, np.ndarray, Image.Image]:
    if expected_format not in IMAGES_TRANSCODING_METHODS:
        raise NotImplementedError(
            f"Expected format: {expected_format} is not supported in terms of visualisations transcoding."
        )
    transcoding_method = IMAGES_TRANSCODING_METHODS[expected_format]
    return transcoding_method(visualisation)


def adjust_prediction_to_client_scaling_factor(
    prediction: dict,
    scaling_factor: Optional[float],
) -> dict:
    if scaling_factor is None:
        return prediction
    if "image" in prediction:
        prediction["image"] = {
            "width": round(prediction["image"]["width"] / scaling_factor),
            "height": round(prediction["image"]["height"] / scaling_factor),
        }
    if predictions_should_not_be_post_processed(prediction=prediction):
        return prediction
    if "points" in prediction["predictions"][0]:
        prediction[
            "predictions"
        ] = adjust_instance_segmentation_predictions_to_client_scaling_factor(
            predictions=prediction["predictions"],
            scaling_factor=scaling_factor,
        )
    elif "x" in prediction["predictions"][0] and "y" in prediction["predictions"][0]:
        prediction[
            "predictions"
        ] = adjust_object_detection_predictions_to_client_scaling_factor(
            predictions=prediction["predictions"],
            scaling_factor=scaling_factor,
        )
    return prediction


def predictions_should_not_be_post_processed(prediction: dict) -> bool:
    # excluding from post-processing classification output and empty predictions
    return (
        "predictions" not in prediction
        or not issubclass(type(prediction["predictions"]), list)
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


def adjust_instance_segmentation_predictions_to_client_scaling_factor(
    predictions: List[dict],
    scaling_factor: float,
) -> List[dict]:
    result = []
    for prediction in predictions:
        prediction = adjust_bbox_coordinates_to_client_scaling_factor(
            bbox=prediction,
            scaling_factor=scaling_factor,
        )
        prediction["points"] = adjust_segmentation_polygon_to_client_scaling_factor(
            points=prediction["points"],
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


def adjust_segmentation_polygon_to_client_scaling_factor(
    points: List[dict],
    scaling_factor: float,
) -> List[dict]:
    result = []
    for point in points:
        point["x"] = point["x"] / scaling_factor
        point["y"] = point["y"] / scaling_factor
        result.append(point)
    return result
