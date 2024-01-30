import base64
from io import BytesIO
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image, ImageChops
from requests import Response

from inference_sdk.http.entities import VisualisationResponseFormat
from inference_sdk.http.utils import post_processing
from inference_sdk.http.utils.post_processing import (
    adjust_bbox_coordinates_to_client_scaling_factor,
    adjust_object_detection_predictions_to_client_scaling_factor,
    adjust_points_coordinates_to_client_scaling_factor,
    adjust_prediction_to_client_scaling_factor,
    adjust_prediction_with_bbox_and_points_to_client_scaling_factor,
    decode_workflow_output_image,
    decode_workflow_outputs,
    is_workflow_image,
    response_contains_jpeg_image,
    transform_base64_visualisation,
)


def test_adjust_points_coordinates_to_client_scaling_factor() -> None:
    # given
    points = [{"x": 50, "y": 100}, {"x": 80, "y": 120}]

    # when
    result = adjust_points_coordinates_to_client_scaling_factor(
        points=points,
        scaling_factor=0.5,
    )

    # then
    assert result == [{"x": 100.0, "y": 200.0}, {"x": 160.0, "y": 240.0}]


def test_adjust_bbox_coordinates_to_client_scaling_factor() -> None:
    # given
    bbox = {
        "x": 50.0,
        "y": 100.0,
        "width": 100.0,
        "height": 150.0,
        "confidence": 0.9,
        "class": "A",
    }

    # when
    result = adjust_bbox_coordinates_to_client_scaling_factor(
        bbox=bbox,
        scaling_factor=0.5,
    )

    # then
    assert result == {
        "x": 100.0,
        "y": 200.0,
        "width": 200.0,
        "height": 300.0,
        "confidence": 0.9,
        "class": "A",
    }


def test_adjust_prediction_with_bbox_and_points_to_client_scaling_factor() -> None:
    # given
    predictions = [
        {
            "x": 50.0,
            "y": 100.0,
            "width": 100.0,
            "height": 150.0,
            "confidence": 0.9,
            "class": "A",
            "points": [{"x": 50, "y": 100}, {"x": 80, "y": 120}],
        }
    ]

    # when
    result = adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
        predictions=predictions,
        scaling_factor=0.5,
        points_key="points",
    )

    # then
    assert result == [
        {
            "x": 100.0,
            "y": 200.0,
            "width": 200.0,
            "height": 300.0,
            "confidence": 0.9,
            "class": "A",
            "points": [{"x": 100.0, "y": 200.0}, {"x": 160.0, "y": 240.0}],
        }
    ]


def test_adjust_object_detection_predictions_to_client_scaling_factor() -> None:
    # given
    predictions = [
        {
            "x": 50.0,
            "y": 100.0,
            "width": 100.0,
            "height": 150.0,
            "confidence": 0.9,
            "class": "A",
        }
    ]

    # when
    result = adjust_object_detection_predictions_to_client_scaling_factor(
        predictions=predictions,
        scaling_factor=0.5,
    )

    # then
    assert result == [
        {
            "x": 100.0,
            "y": 200.0,
            "width": 200.0,
            "height": 300.0,
            "confidence": 0.9,
            "class": "A",
        }
    ]


def test_response_contains_jpeg_image_when_content_type_headers_are_missing() -> None:
    # given
    response = Response()

    # when
    result = response_contains_jpeg_image(response=response)

    # then
    assert result is False


def test_response_contains_jpeg_image_when_content_type_headers_refer_to_image() -> (
    None
):
    # given
    response = Response()
    response.headers = {"content-type": "image/jpeg"}

    # when
    result = response_contains_jpeg_image(response=response)

    # then
    assert result is True


def test_response_contains_jpeg_image_when_content_type_headers_refer_to_another_format() -> (
    None
):
    # given
    response = Response()
    response.headers = {"content-type": "application/json"}

    # when
    result = response_contains_jpeg_image(response=response)

    # then
    assert result is False


def test_transform_base64_visualisation_when_result_requested_in_unknown_format() -> (
    None
):
    # when
    with pytest.raises(NotImplementedError):
        _ = transform_base64_visualisation(
            visualisation=base64.b64encode(b"dummy").decode("utf-8"),
            expected_format="UNKNOWN",  # type: ignore
        )


def test_transform_base64_visualisation_when_result_should_be_base64() -> None:
    # given
    payload = base64.b64encode(b"dummy").decode("utf-8")

    # when
    result = transform_base64_visualisation(
        visualisation=payload, expected_format=VisualisationResponseFormat.BASE64
    )

    # then
    assert result == payload


def test_transform_base64_visualisation_when_result_should_be_np_array() -> None:
    # given
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", image)
    image_bytes = np.array(img_encoded).tobytes()
    payload = base64.b64encode(image_bytes).decode("utf-8")

    # when
    result = transform_base64_visualisation(
        visualisation=payload, expected_format=VisualisationResponseFormat.NUMPY
    )

    # then
    assert result.shape == image.shape
    assert np.allclose(image, result)


def test_transform_base64_visualisation_when_result_should_be_pillow_image() -> None:
    # given
    image = Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        payload = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # when
    result = transform_base64_visualisation(
        visualisation=payload, expected_format=VisualisationResponseFormat.PILLOW
    )

    # then
    assert result.size == image.size
    difference = ImageChops.difference(image, result)
    assert difference.getbbox() is None


def test_adjust_prediction_to_client_scaling_factor_when_scaling_factor_is_not_enabled() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {"class": "Corn", "class_id": 1, "confidence": 0.4329},
            {"class": "Rice", "class_id": 0, "confidence": 0.3397},
        ],
        "top": "Corn",
        "confidence": 0.4329,
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=None
    )

    # then
    assert result is prediction


def test_adjust_prediction_to_client_scaling_factor_when_sstub_prediction_is_received() -> (
    None
):
    # given
    prediction = {
        "time": 0.0002442499971948564,
        "is_stub": True,
        "model_id": "instance-seg/0",
        "task_type": "instance-segmentation",
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=2.0
    )

    # then
    assert result is prediction


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_classification() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {"class": "Corn", "class_id": 1, "confidence": 0.4329},
            {"class": "Rice", "class_id": 0, "confidence": 0.3397},
        ],
        "top": "Corn",
        "confidence": 0.4329,
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    for key in [
        "visualization",
        "frame_id",
        "time",
        "predictions",
        "top",
        "confidence",
    ]:
        assert result[key] == prediction[key]


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_object_detection() -> (
    None
):
    # given
    prediction = {
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {
                "x": 21.0,
                "y": 128.0,
                "width": 42.0,
                "height": 48.0,
                "confidence": 0.9,
                "class": "wood-log",
                "class_id": 0,
            }
        ],
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == {
        "x": 42.0,
        "y": 256.0,
        "width": 84.0,
        "height": 96.0,
        "confidence": 0.9,
        "class": "wood-log",
        "class_id": 0,
    }


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_instance_segmentation() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {
                "x": 21.0,
                "y": 128.0,
                "width": 42.0,
                "height": 48.0,
                "confidence": 0.9,
                "class": "wood-log",
                "class_id": 0,
                "points": [{"x": 100.0, "y": 200.0}],
            }
        ],
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == {
        "x": 42.0,
        "y": 256.0,
        "width": 84.0,
        "height": 96.0,
        "confidence": 0.9,
        "class": "wood-log",
        "class_id": 0,
        "points": [{"x": 200.0, "y": 400.0}],
    }


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_keypoints_detection() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {
                "x": 21.0,
                "y": 128.0,
                "width": 42.0,
                "height": 48.0,
                "confidence": 0.9,
                "class": "wood-log",
                "class_id": 0,
                "keypoints": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "confidence": 0.92,
                        "class_id": 16,
                        "class": "right_ankle",
                    }
                ],
            }
        ],
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == {
        "x": 42.0,
        "y": 256.0,
        "width": 84.0,
        "height": 96.0,
        "confidence": 0.9,
        "class": "wood-log",
        "class_id": 0,
        "keypoints": [
            {
                "x": 200.0,
                "y": 400.0,
                "confidence": 0.92,
                "class_id": 16,
                "class": "right_ankle",
            }
        ],
    }


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_multi_label_classification() -> (
    None
):
    # given
    prediction = {
        "time": 0.14,
        "image": {"width": 224, "height": 224},
        "predictions": {"cat": {"confidence": 0.49}, "dog": {"confidence": 0.92}},
        "predicted_classes": ["cat", "dog"],
    }

    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5
    )

    # then
    assert result["image"] == {"width": 448, "height": 448}
    assert result["time"] == prediction["time"]
    assert result["predictions"] == prediction["predictions"]
    assert result["predicted_classes"] == prediction["predicted_classes"]


def test_is_workflow_image_when_workflow_image_given() -> None:
    # given
    image = {"type": "base64", "value": "base64_image_here"}

    # when
    result = is_workflow_image(value=image)

    # then
    assert result is True


@pytest.mark.parametrize(
    "value", [{"type": "url", "value": "https://some.com/image.jpg"}, "some", 3]
)
def test_is_workflow_image_when_not_a_workflow_image_given(value: Any) -> None:
    # when
    result = is_workflow_image(value=value)

    # then
    assert result is False


def test_decode_workflow_output_image_image_when_base64_image_expected() -> None:
    # given
    value = {"type": "base64", "value": "base64_image_here"}

    # when
    result = decode_workflow_output_image(
        value=value,
        expected_format=VisualisationResponseFormat.BASE64,
    )

    # then
    assert result == {
        "type": "base64",
        "value": "base64_image_here",
    }, "Input value should not be changed"


@mock.patch.object(post_processing, "transform_base64_visualisation")
def test_decode_workflow_output_image_when_numpy_image_expected(
    transform_base64_visualisation_mock: MagicMock,
) -> None:
    # given
    value = {"type": "base64", "value": "base64_image_here"}

    # when
    result = decode_workflow_output_image(
        value=value,
        expected_format=VisualisationResponseFormat.NUMPY,
    )

    # then
    assert result["type"] == "numpy_object", "Numpy object type should be output"
    transform_base64_visualisation_mock.assert_called_once_with(
        visualisation="base64_image_here",
        expected_format=VisualisationResponseFormat.NUMPY,
    )


@mock.patch.object(post_processing, "transform_base64_visualisation")
def test_decode_workflow_output_image_when_pil_image_expected(
    transform_base64_visualisation_mock: MagicMock,
) -> None:
    # given
    value = {"type": "base64", "value": "base64_image_here"}

    # when
    result = decode_workflow_output_image(
        value=value,
        expected_format=VisualisationResponseFormat.PILLOW,
    )

    # then
    assert result["type"] == "pil", "Pillow object type should be output"
    transform_base64_visualisation_mock.assert_called_once_with(
        visualisation="base64_image_here",
        expected_format=VisualisationResponseFormat.PILLOW,
    )


@mock.patch.object(post_processing, "transform_base64_visualisation", MagicMock())
def test_decode_workflow_outputs() -> None:
    # given
    workflow_outputs = {
        "some": "value",
        "other": {"type": "base64", "value": "base64_image_here"},
        "third": [1, {"type": "base64", "value": "base64_image_here"}],
    }

    # when
    result = decode_workflow_outputs(
        workflow_outputs=workflow_outputs,
        expected_format=VisualisationResponseFormat.NUMPY,
    )

    # then
    assert len(result) == 3, "Number of elements in dict cannot be altered"
    assert result["some"] == "value", "This value must not be changed"
    assert result["other"]["type"] == "numpy_object", "This element must be deserialize"
    assert result["third"][0] == 1, "This object cannot be mutated"
    assert (
        result["third"][1]["type"] == "numpy_object"
    ), "This element must be deserialize"
