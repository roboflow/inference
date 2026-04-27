"""Regression test for the legacy ``POST /{dataset_id}/{version_id}`` route.

The route's ``confidence`` query param backs ``Confidence = Union[float,
Literal["best", "default"]]`` request entities. Earlier the param was typed as
``float`` and FastAPI rejected ``confidence=best`` with a 422 — only manifesting
in remote workflow execution mode where the v3 model blocks select the v0 API
(``WORKFLOWS_REMOTE_API_TARGET=hosted``). This test spins up the FastAPI app
with a mocked model manager and confirms each task type accepts the same
confidence-mode values the v3 blocks send.
"""
from typing import Optional, Union
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from starlette.testclient import TestClient


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


class _DummyResponse(BaseModel):
    visualization: Optional[bytes] = None


def _build_interface(monkeypatch, task_type: str):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.get_task_type.return_value = task_type
    model_manager.infer_from_request_sync.return_value = _DummyResponse()

    interface = http_api.HttpInterface(model_manager=model_manager)
    return interface, model_manager


# task_type -> set of confidence values the corresponding request entity is
# expected to accept. "best"/"default" are the two non-float modes v3 blocks
# can serialize; semseg and keypoint-detection reject "best" via field
# validators because model eval doesn't yet produce thresholds for those
# tasks.
_ACCEPTED_CONFIDENCE_MODES = {
    "object-detection": ["best", "default", 0.5],
    "instance-segmentation": ["best", "default", 0.5],
    "keypoint-detection": ["default", 0.5],
    "classification": ["best", "default", 0.5],
    "semantic-segmentation": ["default", 0.5],
}


@pytest.mark.parametrize(
    "task_type, confidence",
    [
        (task, conf)
        for task, modes in _ACCEPTED_CONFIDENCE_MODES.items()
        for conf in modes
    ],
)
def test_legacy_route_accepts_confidence_mode(
    monkeypatch,
    task_type: str,
    confidence: Union[str, float],
) -> None:
    interface, model_manager = _build_interface(monkeypatch, task_type=task_type)

    with TestClient(interface.app) as client:
        response = client.post(
            "/dummy-dataset/1",
            params={
                "api_key": "query-api-key",
                "confidence": confidence,
                "image": "https://example.com/test.jpg",
            },
        )

    assert (
        response.status_code == 200
    ), f"task={task_type} confidence={confidence!r} body={response.text}"
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.confidence == confidence


@pytest.mark.parametrize(
    "task_type",
    ["semantic-segmentation", "keypoint-detection"],
)
def test_legacy_route_rejects_best_confidence_for_unsupported_tasks(
    monkeypatch,
    task_type: str,
) -> None:
    """Tasks whose request entities reject ``confidence="best"`` should
    surface a 4xx, not silently fall back."""
    interface, _ = _build_interface(monkeypatch, task_type=task_type)

    with TestClient(interface.app) as client:
        response = client.post(
            "/dummy-dataset/1",
            params={
                "api_key": "query-api-key",
                "confidence": "best",
                "image": "https://example.com/test.jpg",
            },
        )

    assert response.status_code >= 400, response.text
