import base64
import itertools
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from requests import Response

from inference_sdk.http.entities import ModelDescription, VisualisationResponseFormat
from inference_sdk.http.utils.encoding import (
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


def decode_workflow_outputs(
    workflow_outputs: Dict[str, Any],
    expected_format: VisualisationResponseFormat,
) -> Dict[str, Any]:
    result = {}
    for key, value in workflow_outputs.items():
        if is_workflow_image(value=value):
            value = decode_workflow_output_image(
                value=value,
                expected_format=expected_format,
            )
        elif issubclass(type(value), dict):
            value = decode_workflow_outputs(
                workflow_outputs=value, expected_format=expected_format
            )
        elif issubclass(type(value), list):
            value = decode_workflow_output_list(
                elements=value,
                expected_format=expected_format,
            )
        result[key] = value
    return result


def decode_workflow_output_list(
    elements: List[Any],
    expected_format: VisualisationResponseFormat,
) -> List[Any]:
    result = []
    for element in elements:
        if is_workflow_image(value=element):
            element = decode_workflow_output_image(
                value=element,
                expected_format=expected_format,
            )
        elif issubclass(type(element), dict):
            element = decode_workflow_outputs(
                workflow_outputs=element, expected_format=expected_format
            )
        elif issubclass(type(element), list):
            element = decode_workflow_output_list(
                elements=element,
                expected_format=expected_format,
            )
        result.append(element)
    return result


def is_workflow_image(value: Any) -> bool:
    return issubclass(type(value), dict) and value.get("type") == "base64"


def decode_workflow_output_image(
    value: Dict[str, Any],
    expected_format: VisualisationResponseFormat,
) -> Dict[str, Any]:
    if expected_format is VisualisationResponseFormat.BASE64:
        return value
    if expected_format is VisualisationResponseFormat.NUMPY:
        value["type"] = "numpy_object"
    else:
        value["type"] = "pil"
    value["value"] = transform_base64_visualisation(
        visualisation=value["value"],
        expected_format=expected_format,
    )
    return value


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
    if scaling_factor is None or prediction.get("is_stub", False):
        return prediction
    if "image" in prediction:
        prediction["image"] = {
            "width": round(prediction["image"]["width"] / scaling_factor),
            "height": round(prediction["image"]["height"] / scaling_factor),
        }
    if predictions_should_not_be_post_processed(prediction=prediction):
        return prediction
    if "points" in prediction["predictions"][0]:
        prediction["predictions"] = (
            adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
                points_key="points",
            )
        )
    elif "keypoints" in prediction["predictions"][0]:
        prediction["predictions"] = (
            adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
                points_key="keypoints",
            )
        )
    elif "x" in prediction["predictions"][0] and "y" in prediction["predictions"][0]:
        prediction["predictions"] = (
            adjust_object_detection_predictions_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
            )
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


def combine_gaze_detections(
    detections: Union[dict, List[Union[dict, List[dict]]]]
) -> Union[dict, List[Dict]]:
    if not issubclass(type(detections), list):
        return detections
    detections = [e if issubclass(type(e), list) else [e] for e in detections]
    return list(itertools.chain.from_iterable(detections))


def combine_clip_embeddings(embeddings: Union[dict, List[dict]]) -> List[dict]:
    if issubclass(type(embeddings), list):
        result = []
        for e in embeddings:
            result.extend(combine_clip_embeddings(embeddings=e))
        return result
    frame_id = embeddings["frame_id"]
    time = embeddings["time"]
    if len(embeddings["embeddings"]) > 1:
        new_embeddings = [
            {"frame_id": frame_id, "time": time, "embeddings": [e]}
            for e in embeddings["embeddings"]
        ]
    else:
        new_embeddings = [embeddings]
    return new_embeddings


def filter_model_descriptions(
    descriptions: List[ModelDescription],
    model_id: str,
) -> Optional[ModelDescription]:
    matching_models = [d for d in descriptions if d.model_id == model_id]
    if len(matching_models) > 0:
        return matching_models[0]
    return None
