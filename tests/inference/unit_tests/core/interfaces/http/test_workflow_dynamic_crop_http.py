"""Exercise crop lineage through the real HTTP workflow execution endpoint.

Run in separate processes with ENABLE_TENSOR_DATA_REPRESENTATION=False and True
(WORKFLOWS_IMAGE_TENSOR_DEVICE=cpu). OFFLINE_MODE=True and
DISABLE_VERSION_CHECK=True keep server bootstrap offline. No model weights,
credentials, GPU, or listening HTTP server are needed. TestClient exercises the
real ASGI app, request parsing, engine, blocks, image codec, and JSON response;
only the model-manager boundary returns synthetic inference results.
"""

import base64
import socket
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
import requests
import torch
from fastapi.testclient import TestClient
from prometheus_client.core import REGISTRY

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference_models.models.base.object_detection import Detections


def _image() -> np.ndarray:
    y, x = np.indices((80, 120), dtype=np.uint8)
    image = np.stack((x, y, np.full_like(x, 173)), axis=-1)

    return image


def _model_result(model_id: str, image: np.ndarray) -> tuple:
    # Check actual decoded/cropped pixels at the model boundary. These expected
    # slices do not use any production crop or coordinate helper.
    if model_id == "synthetic-root/1":
        np.testing.assert_array_equal(image, _image())
        box = (20, 10, 100, 70)
    else:
        assert model_id == "synthetic-detail/1"
        np.testing.assert_array_equal(image, _image()[10:70, 20:100])
        box = (5, 7, 35, 27)

    return box


def _infer_numpy(model_id, request, **kwargs):
    responses = []
    images = request.image if isinstance(request.image, list) else [request.image]
    for image in images:
        assert image.type == "numpy_object"
        x1, y1, x2, y2 = _model_result(model_id, image.value)
        responses.append(
            ObjectDetectionInferenceResponse(
                image=InferenceResponseImage(
                    width=image.value.shape[1], height=image.value.shape[0]
                ),
                predictions=[
                    ObjectDetectionPrediction(
                        x=(x1 + x2) / 2,
                        y=(y1 + y2) / 2,
                        width=x2 - x1,
                        height=y2 - y1,
                        confidence=0.9,
                        class_id=0,
                        detection_id=f"{model_id}-detection",
                        **{"class": "synthetic"},
                    )
                ],
            )
        )

    return responses


def _infer_tensor(model_id, images, input_color_format, **kwargs):
    responses = []
    for image in images:
        if isinstance(image, torch.Tensor):
            assert image.device.type == "cpu"
            image = image.permute(1, 2, 0).numpy()
        if input_color_format == "rgb":
            image = image[..., ::-1]
        else:
            assert input_color_format == "bgr"
        box = _model_result(model_id, image)
        responses.append(
            Detections(
                xyxy=torch.tensor([box], dtype=torch.float32),
                class_id=torch.tensor([0], dtype=torch.long),
                confidence=torch.tensor([0.9], dtype=torch.float32),
                bboxes_metadata=[{"detection_id": f"{model_id}-detection"}],
            )
        )

    return responses


@pytest.fixture
def _offline_client(monkeypatch):
    import inference.core.interfaces.http.http_api as http_api

    # Use self-hosted, offline routes without unrelated external services.
    for flag in (
        "GCP_SERVERLESS",
        "LAMBDA",
        "ENABLE_STREAM_API",
        "ENABLE_BUILDER",
        "ENABLE_DASHBOARD",
        "ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG",
        "HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED",
        "OTEL_TRACING_ENABLED",
    ):
        monkeypatch.setattr(http_api, flag, False)
    monkeypatch.setattr(http_api, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "DISABLE_WORKFLOW_ENDPOINTS", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )

    network_attempts = []

    def _deny_network(*args, **kwargs):
        network_attempts.append("outbound connection attempted")
        raise AssertionError("The synthetic HTTP workflow must not contact a service")

    # TestClient uses its in-process ASGI transport, not requests.
    monkeypatch.setattr(requests.sessions.Session, "request", _deny_network)
    for method_name in ("connect", "connect_ex"):
        original = getattr(socket.socket, method_name)

        def _guard_connect(sock, address, *, _original=original):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                network_attempts.append(address)
                _deny_network()

            result = _original(sock, address)

            return result

        monkeypatch.setattr(socket.socket, method_name, _guard_connect)

    manager = MagicMock()
    manager.pingback = None
    manager.num_errors = 0
    manager.infer_from_request_sync.side_effect = _infer_numpy
    manager.run_tensor_native_inference.side_effect = _infer_tensor
    manager.get_class_names.return_value = ["synthetic"]
    interface = http_api.HttpInterface(model_manager=manager)
    try:
        with TestClient(interface.app) as client:
            yield client, manager
    finally:
        REGISTRY.unregister(interface._instrumentator.collector)
        assert not network_attempts


def _workflow(*, inner_workflow: bool) -> dict:
    steps = [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "root_detection",
            "images": "$inputs.image",
            "model_id": "synthetic-root/1",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "first_crop",
            "images": "$inputs.image",
            "predictions": "$steps.root_detection.predictions",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detail_detection",
            "images": "$steps.first_crop.crops",
            "model_id": "synthetic-detail/1",
        },
    ]
    second_crop = {
        "type": "roboflow_core/dynamic_crop@v1",
        "name": "second_crop",
        "images": "$steps.first_crop.crops",
        "predictions": "$steps.detail_detection.predictions",
    }
    if inner_workflow:
        second_crop = {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "second_crop",
            "workflow_definition": {
                "version": "1.0",
                "inputs": [
                    {"type": "WorkflowImage", "name": "image"},
                    {
                        "type": "WorkflowBatchInput",
                        "name": "predictions",
                        "kind": ["object_detection_prediction"],
                    },
                ],
                "steps": [
                    {
                        "type": "roboflow_core/dynamic_crop@v1",
                        "name": "crop",
                        "images": "$inputs.image",
                        "predictions": "$inputs.predictions",
                    }
                ],
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": name,
                        "selector": f"$steps.crop.{name}",
                    }
                    for name in ("crops", "predictions")
                ],
            },
            "parameter_bindings": {
                "image": "$steps.first_crop.crops",
                "predictions": "$steps.detail_detection.predictions",
            },
        }
    steps.extend(
        [
            second_crop,
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "offset",
                "predictions": "$steps.second_crop.predictions",
                "offset_width": 20,
                "offset_height": 12,
                "units": "Pixels",
            },
        ]
    )
    outputs = [
        {
            "type": "JsonField",
            "name": f"{step}_{coordinates}",
            "selector": f"$steps.{step}.predictions",
            "coordinates_system": coordinates,
        }
        for step in ("first_crop", "second_crop", "offset")
        for coordinates in ("own", "parent")
    ]
    outputs.append(
        {
            "type": "JsonField",
            "name": "crop_image",
            "selector": "$steps.second_crop.crops",
        }
    )
    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": outputs,
    }

    return specification


def _only_item(value):
    while isinstance(value, list):
        assert len(value) == 1
        value = value[0]

    return value


def _assert_box(payload, *, box, image_size):
    payload = _only_item(payload)
    assert payload["image"] == {"width": image_size[0], "height": image_size[1]}
    assert len(payload["predictions"]) == 1
    prediction = payload["predictions"][0]
    assert [prediction[key] for key in ("x", "y", "width", "height")] == box
    assert prediction["class"] == "synthetic"
    assert prediction["confidence"] == pytest.approx(0.9)


@pytest.mark.parametrize("inner_workflow", [False, True], ids=["inline", "inner"])
@pytest.mark.parametrize("batch_size", [1, 2], ids=["single", "batch"])
def test_http_dynamic_crop_preserves_root_coordinates_and_offset_clipping(
    _offline_client, inner_workflow: bool, batch_size: int
) -> None:
    """Preserve crop-local and root coordinates over the complete server path.

    Args:
        _offline_client: Real HTTP application and synthetic model manager.
        inner_workflow: Put the second crop inside a real inner workflow.
        batch_size: Number of synthetic image inputs submitted over HTTP.
    """
    client, manager = _offline_client
    success, png = cv2.imencode(".png", _image())
    assert success
    encoded = base64.b64encode(png.tobytes()).decode("ascii")
    images = {"type": "base64", "value": encoded}
    if batch_size > 1:
        images = [images] * batch_size

    response = client.post(
        "/workflows/run",
        json={
            "specification": _workflow(inner_workflow=inner_workflow),
            "inputs": {"image": images},
            "disable_sinks": True,
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    results = response.json()["outputs"]
    assert len(results) == batch_size
    for result in results:
        # Literal expectations: root ROI is [20, 10, 100, 70]. Its detail box
        # [5, 7, 35, 27] therefore occupies [25, 17, 55, 37] in the root image.
        # Offset expansion clips to the second crop's 30 x 20 bounds, not the
        # first crop's 80 x 60 bounds or the root image's 120 x 80 bounds.
        _assert_box(
            result["first_crop_parent"], box=[60, 40, 80, 60], image_size=(120, 80)
        )
        _assert_box(result["first_crop_own"], box=[40, 30, 80, 60], image_size=(80, 60))
        for step in ("second_crop", "offset"):
            _assert_box(
                result[f"{step}_own"], box=[15, 10, 30, 20], image_size=(30, 20)
            )
            _assert_box(
                result[f"{step}_parent"], box=[40, 27, 30, 20], image_size=(120, 80)
            )
        crop = _only_item(result["crop_image"])
        assert crop["type"] == "base64"
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(crop["value"]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded.shape == (20, 30, 3)

    if ENABLE_TENSOR_DATA_REPRESENTATION:
        assert manager.run_tensor_native_inference.call_count == 2
        manager.infer_from_request_sync.assert_not_called()
    else:
        assert manager.infer_from_request_sync.call_count == 2
        manager.run_tensor_native_inference.assert_not_called()
    assert manager.add_model.call_count == 2
