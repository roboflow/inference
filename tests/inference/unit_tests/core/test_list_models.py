"""Tests for the GET /list_models endpoint and its helper logic.

Tests are split into:
1. Unit tests for the pure helper functions (_strip_resolution, _lookup_tasks,
   _pick_task, _find_infer_endpoint) — exercised indirectly through the endpoint.
2. Integration-style tests using FastAPI TestClient against a minimal
   HttpInterface with a mocked model registry and mocked Roboflow API.
"""

from typing import Dict, List, Optional, Set
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from requests_mock import Mocker
from starlette.testclient import TestClient

from inference.core.env import API_BASE_URL
from inference.core.exceptions import WorkspaceLoadError
from inference.core.registries.base import ModelRegistry
from inference.core.utils.url_utils import wrap_url


# ---------------------------------------------------------------------------
# Minimal fixtures that construct an HttpInterface with a controlled registry
# ---------------------------------------------------------------------------

def _make_model_manager(registry_dict: dict):
    """Build a minimal ModelManager-like object matching the decorator chain
    used in production (WithFixedSizeCache wrapping ModelManager).
    HttpInterface accesses the registry via model_manager.model_manager.model_registry."""
    inner = MagicMock()
    inner.model_registry = ModelRegistry(registry_dict)
    outer = MagicMock()
    outer.model_manager = inner
    outer.pingback = None
    outer._models = {}
    outer.describe_models.return_value = []
    return outer


def _build_app(registry_dict: dict) -> FastAPI:
    """
    Instantiate HttpInterface with a controlled registry and return the
    FastAPI app. We patch env flags so that the routes we care about are
    registered without needing real model dependencies.
    """
    manager = _make_model_manager(registry_dict)
    with (
        patch("inference.core.interfaces.http.http_api.LAMBDA", False),
        patch("inference.core.interfaces.http.http_api.GCP_SERVERLESS", False),
        patch("inference.core.interfaces.http.http_api.CORE_MODELS_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.LEGACY_ROUTE_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.ENABLE_STREAM_API", False),
        patch("inference.core.interfaces.http.http_api.DISABLE_WORKFLOW_ENDPOINTS", True),
        patch("inference.core.interfaces.http.http_api.ENABLE_BUILDER", False),
        patch("inference.core.interfaces.http.http_api.ENABLE_DASHBOARD", False),
        patch("inference.core.interfaces.http.http_api.METRICS_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.NOTEBOOK_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.WEBRTC_WORKER_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.LMM_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.MOONDREAM2_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.DEPTH_ESTIMATION_ENABLED", False),
        patch("inference.core.interfaces.http.http_api.GET_MODEL_REGISTRY_ENABLED", False),
    ):
        from inference.core.interfaces.http.http_api import HttpInterface

        interface = HttpInterface(model_manager=manager)
        return interface.app


# A minimal registry that covers the key cases: object-detection,
# instance-segmentation, classification, keypoint-detection, plus some
# model_types that appear under multiple tasks (like yolov8n).
SAMPLE_REGISTRY = {
    ("object-detection", "rfdetr-nano"): MagicMock,
    ("object-detection", "rfdetr-small"): MagicMock,
    ("object-detection", "yolov8n"): MagicMock,
    ("object-detection", "yolov8s"): MagicMock,
    ("instance-segmentation", "yolov8n"): MagicMock,
    ("instance-segmentation", "yolov8n-seg"): MagicMock,
    ("instance-segmentation", "yolov8s-seg"): MagicMock,
    ("instance-segmentation", "rfdetr-seg-nano"): MagicMock,
    ("classification", "yolov8n"): MagicMock,
    ("classification", "resnet50"): MagicMock,
    ("keypoint-detection", "yolov8n-pose"): MagicMock,
    # Naming variant tests: yolov11 in registry, yolo11 aliases exist
    ("object-detection", "yolov11n"): MagicMock,
    ("instance-segmentation", "yolov11n-seg"): MagicMock,
    # Naming variant tests: yolo26 in registry (without 'v'), yolov26 aliases exist
    ("object-detection", "yolo26n"): MagicMock,
    ("instance-segmentation", "yolo26n-seg"): MagicMock,
    # Naming variant tests: yolo_nas_s in registry, yolo-nas-s aliases exist
    ("object-detection", "yolo_nas_s"): MagicMock,
}


@pytest.fixture(scope="module")
def app():
    return _build_app(SAMPLE_REGISTRY)


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests for GET /list_models — public aliased models
# ---------------------------------------------------------------------------


class TestListModelsPublicAliases:
    """Test that public models from REGISTERED_ALIASES appear correctly."""

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_rfdetr_nano_appears_as_object_detection(
        self, _mock_ws, client: TestClient
    ):
        """rfdetr-nano should resolve to /infer/object_detection."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=rfdetr-nano" in e]
        assert len(matching) == 1
        assert "/infer/object_detection?model_id=rfdetr-nano" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolov8s_seg_640_appears_as_instance_segmentation(
        self, _mock_ws, client: TestClient
    ):
        """yolov8s-seg-640 should strip -640 and resolve to instance_segmentation."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolov8s-seg-640" in e]
        assert len(matching) == 1
        assert "/infer/instance_segmentation?model_id=yolov8s-seg-640" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolov8n_640_appears_as_object_detection_not_classification(
        self, _mock_ws, client: TestClient
    ):
        """yolov8n is in multiple tasks; alias yolov8n-640 (no -seg/-pose) should
        resolve to object-detection, not classification."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolov8n-640" in e]
        assert len(matching) == 1
        assert "/infer/object_detection?model_id=yolov8n-640" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolov8x_pose_640_appears_as_keypoints_detection(
        self, _mock_ws, client: TestClient
    ):
        """yolov8x-pose-640 should resolve to keypoints_detection."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolov8n-pose-640" in e]
        assert len(matching) == 1
        assert "/infer/keypoints_detection?model_id=yolov8n-pose-640" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_resnet50_appears_as_classification(
        self, _mock_ws, client: TestClient
    ):
        """resnet50 has no resolution suffix and maps to classification."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=resnet50" in e]
        assert len(matching) == 1
        assert "/infer/classification?model_id=resnet50" in matching[0]


# ---------------------------------------------------------------------------
# Tests for naming variant resolution
# ---------------------------------------------------------------------------


class TestListModelsNamingVariants:
    """Test that YOLO naming variants (yolo11/yolov11, yolov26/yolo26,
    yolo-nas/yolo_nas) are resolved correctly."""

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolo11n_640_resolves_via_yolov11n(
        self, _mock_ws, client: TestClient
    ):
        """Alias yolo11n-640 strips to yolo11n, which should fallback-match
        yolov11n in the registry."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolo11n-640" in e]
        assert len(matching) == 1
        assert "/infer/object_detection" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolov26n_640_resolves_via_yolo26n(
        self, _mock_ws, client: TestClient
    ):
        """Alias yolov26n-640 strips to yolov26n, which should fallback-match
        yolo26n in the registry."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolov26n-640" in e]
        assert len(matching) == 1
        assert "/infer/object_detection" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolo26n_seg_640_resolves_to_instance_segmentation(
        self, _mock_ws, client: TestClient
    ):
        """yolo26n-seg-640 → yolo26n-seg → instance-segmentation."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolo26n-seg-640" in e]
        assert len(matching) == 1
        assert "/infer/instance_segmentation" in matching[0]

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_yolo_nas_s_640_resolves_via_underscore_fallback(
        self, _mock_ws, client: TestClient
    ):
        """yolo-nas-s-640 → yolo-nas-s → yolo_nas_s via hyphen→underscore."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolo-nas-s-640" in e]
        assert len(matching) == 1
        assert "/infer/object_detection" in matching[0]


# ---------------------------------------------------------------------------
# Tests for GET /list_models — private workspace models
# ---------------------------------------------------------------------------


class TestListModelsPrivateModels:
    """Test that private models from the user's workspace are included."""

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        return_value=(
            {
                "id": "my-dataset",
                "type": "object-detection",
                "versions": [1, 2],
            },
            {
                "id": "seg-project",
                "type": "instance-segmentation",
                "versions": [3],
            },
        ),
    )
    def test_private_models_included_in_response(
        self, _mock_projects, client: TestClient
    ):
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]

        # Check private object-detection models
        assert any(
            "/infer/object_detection?model_id=my-dataset/1" in e
            for e in endpoints
        )
        assert any(
            "/infer/object_detection?model_id=my-dataset/2" in e
            for e in endpoints
        )
        # Check private instance-segmentation model
        assert any(
            "/infer/instance_segmentation?model_id=seg-project/3" in e
            for e in endpoints
        )

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        return_value=(
            {
                "id": "my-dataset",
                "type": "object-detection",
                "versions": [{"id": 1}, {"id": 2}],
            },
        ),
    )
    def test_private_models_with_dict_versions(
        self, _mock_projects, client: TestClient
    ):
        """Versions can be dicts with an 'id' key rather than plain ints."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]

        assert any(
            "/infer/object_detection?model_id=my-dataset/1" in e
            for e in endpoints
        )
        assert any(
            "/infer/object_detection?model_id=my-dataset/2" in e
            for e in endpoints
        )

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("API unreachable"),
    )
    def test_workspace_failure_does_not_break_endpoint(
        self, _mock_projects, client: TestClient
    ):
        """If the workspace API fails, public models should still be returned."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        # Public aliases should still be present
        assert any("rfdetr-nano" in e for e in endpoints)


# ---------------------------------------------------------------------------
# Tests for URL construction
# ---------------------------------------------------------------------------


class TestListModelsURLConstruction:
    """Test that URLs are constructed using request.base_url."""

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_urls_use_testclient_base_url(
        self, _mock_ws, client: TestClient
    ):
        """TestClient uses http://testserver as base_url."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        # All endpoints should start with the test server base URL
        for endpoint in endpoints:
            assert endpoint.startswith("http://testserver/")

    @patch(
        "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
        side_effect=WorkspaceLoadError("no workspace"),
    )
    def test_response_structure(self, _mock_ws, client: TestClient):
        """Response should be a JSON object with an 'endpoints' list of strings."""
        response = client.get("/list_models", params={"api_key": "test_key"})
        assert response.status_code == 200
        body = response.json()
        assert "endpoints" in body
        assert isinstance(body["endpoints"], list)
        assert all(isinstance(e, str) for e in body["endpoints"])


# ---------------------------------------------------------------------------
# Tests for the api_key requirement
# ---------------------------------------------------------------------------


class TestListModelsAuth:
    """Test that the endpoint requires an api_key."""

    def test_missing_api_key_returns_422(self, client: TestClient):
        """FastAPI should return 422 when required query param is missing."""
        response = client.get("/list_models")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests for get_roboflow_workspace_models in roboflow_api.py
# ---------------------------------------------------------------------------


class TestGetRoboflowWorkspaceModels:
    """Test the new get_roboflow_workspace_models function."""

    def test_returns_projects_from_workspace_detail(
        self, requests_mock: Mocker
    ):
        from inference.core.roboflow_api import get_roboflow_workspace_models

        # Root endpoint returns workspace slug
        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/"),
            json={"workspace": "my_workspace"},
        )
        # Workspace detail endpoint returns projects
        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/my_workspace"),
            json={
                "workspace": {
                    "name": "my_workspace",
                    "projects": [
                        {
                            "id": "cars",
                            "type": "object-detection",
                            "versions": [1, 2],
                            "extra_field": "ignored",
                        },
                        {
                            "id": "people-seg",
                            "type": "instance-segmentation",
                            "versions": [3],
                        },
                    ],
                },
            },
        )

        result = get_roboflow_workspace_models(api_key="test_key_1")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0]["id"] == "cars"
        assert result[0]["type"] == "object-detection"
        assert result[0]["versions"] == [1, 2]
        assert result[1]["id"] == "people-seg"
        assert result[1]["type"] == "instance-segmentation"

    def test_returns_empty_tuple_when_no_projects(
        self, requests_mock: Mocker
    ):
        from inference.core.roboflow_api import get_roboflow_workspace_models

        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/"),
            json={"workspace": "empty_ws"},
        )
        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/empty_ws"),
            json={"workspace": {"projects": []}},
        )

        result = get_roboflow_workspace_models(api_key="test_key_3")

        assert result == ()

    def test_defaults_type_when_missing(
        self, requests_mock: Mocker
    ):
        from inference.core.roboflow_api import get_roboflow_workspace_models

        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/"),
            json={"workspace": "ws_no_type"},
        )
        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/ws_no_type"),
            json={
                "workspace": {
                    "projects": [
                        {"id": "no-type-project", "versions": [1]},
                    ],
                },
            },
        )

        result = get_roboflow_workspace_models(api_key="test_key_4")

        assert result[0]["type"] == "object-detection"

    def test_raises_on_unauthorized(self, requests_mock: Mocker):
        from inference.core.exceptions import RoboflowAPINotAuthorizedError
        from inference.core.roboflow_api import get_roboflow_workspace_models

        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/"),
            status_code=401,
        )

        with pytest.raises(RoboflowAPINotAuthorizedError):
            get_roboflow_workspace_models(api_key="bad_key")
