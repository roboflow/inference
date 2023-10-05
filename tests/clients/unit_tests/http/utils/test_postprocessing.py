import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image, ImageChops
from requests import Response

from clients.http.entities import VisualisationResponseFormat
from clients.http.utils.post_processing import (
    adjust_segmentation_polygon_to_client_scaling_factor,
    adjust_bbox_coordinates_to_client_scaling_factor,
    adjust_instance_segmentation_predictions_to_client_scaling_factor,
    adjust_object_detection_predictions_to_client_scaling_factor,
    response_contains_jpeg_image,
    transform_base64_visualisation,
)


def test_adjust_segmentation_polygon_to_client_scaling_factor() -> None:
    # given
    points = [{"x": 50, "y": 100}, {"x": 80, "y": 120}]

    # when
    result = adjust_segmentation_polygon_to_client_scaling_factor(
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


def test_adjust_instance_segmentation_predictions_to_client_scaling_factor() -> None:
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
    result = adjust_instance_segmentation_predictions_to_client_scaling_factor(
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
