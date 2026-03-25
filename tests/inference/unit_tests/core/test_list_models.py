"""Tests for the GET /list_models endpoint.

Tests cover:
1. Public aliased models — task types resolved via get_model_type
   (same path the inference server uses), cached per dataset_id.
2. Private workspace models — fetched via get_roboflow_workspace_models.
3. Core model routes — discovered via GENERIC_MODELS prefix matching.
4. URL construction and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from inference.core.env import API_BASE_URL
from inference.core.exceptions import WorkspaceLoadError
from inference.core.registries.base import ModelRegistry
from inference.core.utils.url_utils import wrap_url


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_manager(registry_dict: dict):
    inner = MagicMock()
    inner.model_registry = ModelRegistry(registry_dict)
    outer = MagicMock()
    outer.model_manager = inner
    outer.pingback = None
    outer._models = {}
    outer.describe_models.return_value = []
    return outer


def _build_app(registry_dict: dict):
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


SAMPLE_REGISTRY = {
    ("object-detection", "stub"): MagicMock,
}


@pytest.fixture(scope="module")
def app():
    return _build_app(SAMPLE_REGISTRY)


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


# Dataset → task type mapping used by the mocked get_model_type
DATASET_TASK_MAP = {
    "coco": "object-detection",
    "coco-dataset-vdnr1": "instance-segmentation",
    "coco-pose-detection": "keypoint-detection",
    "classifiers": "classification",
    "paligemma-pretrains": "object-detection",
    "florence-pretrains": "object-detection",
    "qwen-pretrains": "object-detection",
}


def _fake_get_model_type(model_id, api_key=None, **kwargs):
    """Mock get_model_type: resolve dataset_id and return (task, model_type)."""
    dataset_id = model_id.split("/")[0]
    task = DATASET_TASK_MAP.get(dataset_id, "object-detection")
    return (task, "stub")


def _mock_roboflow_api():
    """Return a dict of patches that mock the Roboflow API calls."""
    return {
        "model_type": patch(
            "inference.core.interfaces.http.http_api.get_model_type",
            side_effect=_fake_get_model_type,
        ),
        "ws_models": patch(
            "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
            side_effect=WorkspaceLoadError("no workspace"),
        ),
    }


# ---------------------------------------------------------------------------
# Public aliased models — task type from get_model_type
# ---------------------------------------------------------------------------


class TestListModelsPublicAliases:
    """Task types are resolved via get_model_type, cached per dataset_id."""

    def test_rfdetr_nano_appears_as_object_detection(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=rfdetr-nano" in e]
        assert len(matching) == 1
        assert "/infer/object_detection?model_id=rfdetr-nano" in matching[0]

    def test_yolov8s_seg_640_appears_as_instance_segmentation(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolov8s-seg-640" in e]
        assert len(matching) == 1
        assert "/infer/instance_segmentation?model_id=yolov8s-seg-640" in matching[0]

    def test_yolov8n_pose_640_appears_as_keypoints_detection(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=yolov8n-pose-640" in e]
        assert len(matching) == 1
        assert "/infer/keypoints_detection?model_id=yolov8n-pose-640" in matching[0]

    def test_resnet50_appears_as_classification(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        endpoints = response.json()["endpoints"]
        matching = [e for e in endpoints if "model_id=resnet50" in e]
        assert len(matching) == 1
        assert "/infer/classification?model_id=resnet50" in matching[0]

    def test_get_model_type_called_once_per_dataset(self, client: TestClient):
        """Multiple aliases sharing a dataset should trigger only one API call."""
        mocks = _mock_roboflow_api()
        with mocks["model_type"] as mock_mt, mocks["ws_models"]:
            client.get("/list_models", params={"api_key": "test_key"})

        # Count calls where model_id starts with "coco/" — should be exactly 1
        coco_calls = [
            c for c in mock_mt.call_args_list
            if c.kwargs.get("model_id", "").startswith("coco/")
            or (c.args and c.args[0].startswith("coco/"))
        ]
        assert len(coco_calls) == 1

    def test_failed_dataset_lookup_skips_alias(self, client: TestClient):
        """If get_model_type raises for a dataset, those aliases are skipped."""
        with (
            patch(
                "inference.core.interfaces.http.http_api.get_model_type",
                side_effect=Exception("API error"),
            ),
            patch(
                "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
                side_effect=WorkspaceLoadError("no workspace"),
            ),
        ):
            response = client.get("/list_models", params={"api_key": "test_key"})

        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        # No alias-based endpoints should appear, but GENERIC_MODELS
        # entries (like classifiers, qwen3_5) still show up
        alias_endpoints = [
            e for e in endpoints
            if "model_id=" in e and "/" in e.split("model_id=")[1]
        ]
        assert len(alias_endpoints) == 0


# ---------------------------------------------------------------------------
# Private workspace models
# ---------------------------------------------------------------------------


class TestListModelsPrivateModels:

    def test_private_models_included_in_response(self, client: TestClient):
        with (
            patch(
                "inference.core.interfaces.http.http_api.get_model_type",
                side_effect=_fake_get_model_type,
            ),
            patch(
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
            ),
        ):
            response = client.get("/list_models", params={"api_key": "test_key"})

        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        assert any("/infer/object_detection?model_id=my-dataset/1" in e for e in endpoints)
        assert any("/infer/object_detection?model_id=my-dataset/2" in e for e in endpoints)
        assert any("/infer/instance_segmentation?model_id=seg-project/3" in e for e in endpoints)

    def test_private_models_with_dict_versions(self, client: TestClient):
        with (
            patch(
                "inference.core.interfaces.http.http_api.get_model_type",
                side_effect=_fake_get_model_type,
            ),
            patch(
                "inference.core.interfaces.http.http_api.get_roboflow_workspace_models",
                return_value=(
                    {
                        "id": "my-dataset",
                        "type": "object-detection",
                        "versions": [{"id": 1}, {"id": 2}],
                    },
                ),
            ),
        ):
            response = client.get("/list_models", params={"api_key": "test_key"})

        endpoints = response.json()["endpoints"]
        assert any("/infer/object_detection?model_id=my-dataset/1" in e for e in endpoints)
        assert any("/infer/object_detection?model_id=my-dataset/2" in e for e in endpoints)

    def test_workspace_failure_does_not_break_endpoint(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        assert response.status_code == 200
        endpoints = response.json()["endpoints"]
        assert any("rfdetr-nano" in e for e in endpoints)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestListModelsURLConstruction:

    def test_urls_use_testclient_base_url(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        for endpoint in response.json()["endpoints"]:
            assert endpoint.startswith("http://testserver/")

    def test_response_structure(self, client: TestClient):
        mocks = _mock_roboflow_api()
        with mocks["model_type"], mocks["ws_models"]:
            response = client.get("/list_models", params={"api_key": "test_key"})

        assert response.status_code == 200
        body = response.json()
        assert "endpoints" in body
        assert isinstance(body["endpoints"], list)
        assert all(isinstance(e, str) for e in body["endpoints"])


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestListModelsAuth:

    def test_missing_api_key_returns_422(self, client: TestClient):
        response = client.get("/list_models")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# get_roboflow_workspace_models unit tests
# ---------------------------------------------------------------------------


class TestGetRoboflowWorkspaceModels:

    def test_returns_projects_from_workspace_detail(self, requests_mock):
        from inference.core.roboflow_api import get_roboflow_workspace_models

        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/"),
            json={"workspace": "my_workspace"},
        )
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

    def test_returns_empty_tuple_when_no_projects(self, requests_mock):
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

    def test_defaults_type_when_missing(self, requests_mock):
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

    def test_raises_on_unauthorized(self, requests_mock):
        from inference.core.exceptions import RoboflowAPINotAuthorizedError
        from inference.core.roboflow_api import get_roboflow_workspace_models

        requests_mock.get(
            url=wrap_url(f"{API_BASE_URL}/"),
            status_code=401,
        )

        with pytest.raises(RoboflowAPINotAuthorizedError):
            get_roboflow_workspace_models(api_key="bad_key")
