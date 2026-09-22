import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from inference_server.legacy.bridge import Route
from inference_server.legacy.common import ImagePayload
from inference_server.legacy.entities import (
    ClassificationInferenceRequest,
    ClassificationInferenceResponse,
    InferenceResponseImage,
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceRequest,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.visualization import render_visualization
from tests.unit_tests.legacy.conftest import FakeGateway

JPEG_MAGIC = b"\xff\xd8"


def _jpeg(width=32, height=32):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _payload(width=32, height=32):
    return ImagePayload(_jpeg(width, height), width, height)


def _image_field():
    return {"type": "base64", "value": base64.b64encode(_jpeg()).decode()}


def _route(task_type, class_names=None, class_colors=None):
    return Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type=task_type,
        action="infer",
        class_names=class_names if class_names is not None else ["cat", "dog"],
        class_colors=class_colors,
    )


def _od_response():
    return ObjectDetectionInferenceResponse(
        image=InferenceResponseImage(width=32, height=32),
        predictions=[
            {
                "x": 16.0,
                "y": 16.0,
                "width": 10.0,
                "height": 12.0,
                "confidence": 0.9,
                "class": "cat",
                "class_id": 0,
            }
        ],
    )


def _od_request(**kwargs):
    return ObjectDetectionInferenceRequest(
        model_id="ds/1", image=_image_field(), **kwargs
    )


def test_render_visualization_object_detection_returns_jpeg():
    rendered = render_visualization(
        _route("object-detection"),
        _od_request(visualize_predictions=True, visualization_stroke_width=2),
        _od_response(),
        _payload(),
    )
    assert isinstance(rendered, bytes)
    assert rendered.startswith(JPEG_MAGIC)


@pytest.mark.parametrize("labels", [True, False])
def test_render_visualization_object_detection_labels_toggle(labels):
    rendered = render_visualization(
        _route("object-detection"),
        _od_request(visualize_predictions=True, visualization_labels=labels),
        _od_response(),
        _payload(),
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_accepts_ndarray_payload():
    array = np.full((32, 32, 3), 40, dtype=np.uint8)
    rendered = render_visualization(
        _route("object-detection"),
        _od_request(visualize_predictions=True),
        _od_response(),
        ImagePayload(array, 32, 32),
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_instance_segmentation_draws_points():
    response = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=32, height=32),
        predictions=[
            {
                "x": 16.0,
                "y": 16.0,
                "width": 10.0,
                "height": 12.0,
                "confidence": 0.9,
                "class": "cat",
                "class_id": 0,
                "points": [
                    {"x": 11.0, "y": 10.0},
                    {"x": 21.0, "y": 10.0},
                    {"x": 21.0, "y": 22.0},
                    {"x": 11.0, "y": 22.0},
                ],
            }
        ],
    )
    request = InstanceSegmentationInferenceRequest(
        model_id="ds/1",
        image=_image_field(),
        visualize_predictions=True,
        visualization_labels=True,
    )
    rendered = render_visualization(
        _route("instance-segmentation"), request, response, _payload()
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_keypoints_draws_circles():
    response = KeypointsDetectionInferenceResponse(
        image=InferenceResponseImage(width=32, height=32),
        predictions=[
            {
                "x": 16.0,
                "y": 16.0,
                "width": 10.0,
                "height": 12.0,
                "confidence": 0.9,
                "class": "cat",
                "class_id": 0,
                "keypoints": [
                    {
                        "x": 13.0,
                        "y": 13.0,
                        "confidence": 0.8,
                        "class_id": 0,
                        "class": "nose",
                    }
                ],
            }
        ],
    )
    request = KeypointsDetectionInferenceRequest(
        model_id="ds/1", image=_image_field(), visualize_predictions=True
    )
    rendered = render_visualization(
        _route("keypoint-detection"), request, response, _payload()
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_classification_returns_jpeg():
    response = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=32, height=32),
        predictions=[{"class": "cat", "class_id": 0, "confidence": 0.9}],
        top="cat",
        confidence=0.9,
    )
    request = ClassificationInferenceRequest(
        model_id="ds/1", image=_image_field(), visualize_predictions=True
    )
    rendered = render_visualization(
        _route("classification"), request, response, _payload()
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_multi_label_classification_returns_jpeg():
    response = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=32, height=32),
        predictions={
            "cat": {"confidence": 0.9, "class_id": 0},
            "dog": {"confidence": 0.7, "class_id": 1},
        },
        predicted_classes=["cat"],
    )
    request = ClassificationInferenceRequest(
        model_id="ds/1", image=_image_field(), visualize_predictions=True
    )
    rendered = render_visualization(
        _route("multi-label-classification"), request, response, _payload()
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_uses_route_class_colors():
    route = _route("object-detection", class_colors={"cat": "#FF0000"})
    rendered = render_visualization(
        route,
        _od_request(visualize_predictions=True, visualization_stroke_width=3),
        _od_response(),
        _payload(),
    )
    assert rendered.startswith(JPEG_MAGIC)


def test_render_visualization_semantic_segmentation_is_501():
    with pytest.raises(LegacyHTTPError) as error:
        render_visualization(
            _route("semantic-segmentation"),
            _od_request(),
            _od_response(),
            _payload(),
        )
    assert error.value.status_code == 501
    assert "semantic-segmentation" in error.value.message


def _detection_gateway():
    detections = SimpleNamespace(
        xyxy=np.array([[2, 2, 20, 24]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    return FakeGateway(
        predictions={("ds/1", "infer"): detections},
        model_info={"ds/1": {"class_names": ["cat"], "tasks": {"infer": {}}}},
    )


def test_catch_all_format_image_returns_jpeg(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    response = legacy_client(_detection_gateway()).post(
        "/ds/1?api_key=k&format=image&labels=true",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(JPEG_MAGIC)


def test_catch_all_format_image_and_json_returns_base64_visualization(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    response = legacy_client(_detection_gateway()).post(
        "/ds/1?api_key=k&format=image_and_json",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"][0]["class"] == "cat"
    assert base64.b64decode(body["visualization"]).startswith(JPEG_MAGIC)


def test_catch_all_format_json_has_no_visualization(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    response = legacy_client(_detection_gateway()).post(
        "/ds/1?api_key=k",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert "visualization" not in response.json()


@pytest.mark.parametrize("image_format", ["image", "image_and_json"])
def test_catch_all_semantic_segmentation_visualization_is_501(
    legacy_client, fake_stat, image_format
):
    fake_stat["ds/1"] = ("semantic-segmentation", "infer")
    segmentation = SimpleNamespace(
        segmentation_map=np.zeros((32, 32), dtype=np.uint8),
        confidence=np.zeros((32, 32), dtype=float),
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): segmentation},
        model_info={"ds/1": {"class_names": ["cat"], "tasks": {"infer": {}}}},
    )
    response = legacy_client(gateway).post(
        f"/ds/1?api_key=k&format={image_format}",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 501
    assert "semantic-segmentation" in response.json()["message"]


def test_typed_object_detection_visualize_predictions(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    response = legacy_client(_detection_gateway()).post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": _image_field(),
            "visualize_predictions": True,
            "visualization_labels": True,
        },
    )
    assert response.status_code == 200
    assert base64.b64decode(response.json()["visualization"]).startswith(JPEG_MAGIC)
