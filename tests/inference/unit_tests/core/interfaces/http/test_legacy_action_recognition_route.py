"""Regression test for the legacy ``POST /{dataset_id}/{version_id}`` route
when the task is action recognition.

Under Lambda the route reads ``request_model_id`` from the authorizer, and it
differs from the ``model_id`` in the path. ``add_model`` registers the model
under the alias, which is ``model_id``, so an inference call naming
``request_model_id`` finds nothing. Off Lambda the two are equal by
assignment, so only a Lambda-shaped request shows the difference.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from starlette.testclient import TestClient

AUTHORIZER_ENDPOINT = "rf-other-workspace--other-model"
RESOLVED_REQUEST_MODEL_ID = "other-workspace/other-model"
PATH_MODEL_ID = "dummy-dataset/1"


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


class _DummyResponse(BaseModel):
    visualization: Optional[bytes] = None


class _LambdaScope:
    """Puts the Lambda authorizer event on the ASGI scope, as API Gateway does."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope["aws.event"] = {
                "requestContext": {
                    "authorizer": {
                        "lambda": {
                            "model": {"endpoint": AUTHORIZER_ENDPOINT},
                            "actor": "actor-id",
                        }
                    }
                }
            }
        await self.app(scope, receive, send)


def _build_interface(monkeypatch, lambda_mode: bool):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "LAMBDA", lambda_mode)
    if lambda_mode:
        # trackUsage is imported at module scope only when LAMBDA is set, so
        # patching LAMBDA after import leaves the name unbound.
        monkeypatch.setattr(http_api, "trackUsage", MagicMock(), raising=False)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.get_task_type.return_value = "action-recognition"
    model_manager.infer_from_request_sync.return_value = _DummyResponse()
    interface = http_api.HttpInterface(model_manager=model_manager)
    return interface, model_manager


@pytest.mark.parametrize("lambda_mode", [False, True])
def test_legacy_action_recognition_infers_with_the_registered_identifier(
    monkeypatch, lambda_mode: bool
) -> None:
    interface, model_manager = _build_interface(monkeypatch, lambda_mode=lambda_mode)
    app = _LambdaScope(interface.app) if lambda_mode else interface.app

    with TestClient(app) as client:
        response = client.post(
            f"/{PATH_MODEL_ID}",
            params={
                "api_key": "query-api-key",
                "image": "https://example.com/clip.mp4",
            },
        )

    assert response.status_code == 200, response.text
    # add_model registers under model_id_alias when one is given.
    add_model_call = model_manager.add_model.call_args
    registered_under = add_model_call.kwargs["model_id_alias"]
    assert registered_under == PATH_MODEL_ID
    # The inference call has to name that same identifier.
    inferred_with = model_manager.infer_from_request_sync.call_args.args[0]
    assert inferred_with == registered_under


def test_lambda_request_model_id_really_does_differ(monkeypatch) -> None:
    """Guards the premise: without a difference the test above proves nothing."""
    interface, model_manager = _build_interface(monkeypatch, lambda_mode=True)

    with TestClient(_LambdaScope(interface.app)) as client:
        client.post(
            f"/{PATH_MODEL_ID}",
            params={
                "api_key": "query-api-key",
                "image": "https://example.com/clip.mp4",
            },
        )

    assert model_manager.add_model.call_args.args[0] == RESOLVED_REQUEST_MODEL_ID
    assert RESOLVED_REQUEST_MODEL_ID != PATH_MODEL_ID


@pytest.mark.parametrize(
    "confidence, expected",
    [
        (None, None),
        ("default", None),
        ("best", "best"),
        (0.5, 0.5),
        (50, 0.5),
        (1, 0.01),
        (0, 0),
    ],
)
@pytest.mark.parametrize("include_candidates", [False, True])
def test_legacy_action_recognition_normalizes_confidence(
    monkeypatch, confidence, expected, include_candidates: bool
) -> None:
    interface, model_manager = _build_interface(monkeypatch, lambda_mode=False)
    params = {
        "api_key": "query-api-key",
        "image": "https://example.com/clip.mp4",
        "include_candidates": str(include_candidates).lower(),
    }
    if confidence is not None:
        params["confidence"] = confidence

    with TestClient(interface.app) as client:
        response = client.post(f"/{PATH_MODEL_ID}", params=params)

    assert response.status_code == 200, response.text
    action_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert action_request.confidence == expected
    assert action_request.include_candidates is include_candidates


def test_legacy_cosmos_ignores_explicit_confidence(monkeypatch) -> None:
    from inference.core.models import inference_models_adapters as adapters
    from inference_models.models.base.action_recognition import (
        ActionRecognitionPrediction,
        VideoSampling,
    )

    interface, manager = _build_interface(monkeypatch, lambda_mode=False)
    adapter = adapters.InferenceModelsActionRecognitionAdapter.__new__(
        adapters.InferenceModelsActionRecognitionAdapter
    )
    adapter._model = SimpleNamespace(
        supports_confidence=False,
        supports_observed_duration=False,
        video_sampling=VideoSampling(window_seconds=2, sample_fps=4, min_frames=1),
        class_names=["walk"],
        resolved_model=None,
        infer=MagicMock(return_value=[ActionRecognitionPrediction(0, 1, "walk")]),
    )
    monkeypatch.setattr(
        adapters, "video_source_path", lambda **kwargs: nullcontext("clip")
    )
    monkeypatch.setattr(adapters, "probe_video", lambda **kwargs: (4, 8))
    monkeypatch.setattr(
        adapters, "read_frame_windows", lambda **kwargs: ([None] * 8 for _ in range(1))
    )
    manager.infer_from_request_sync.side_effect = (
        lambda model_id, request, **kwargs: adapter.infer_from_request(
            request, **kwargs
        )
    )

    with TestClient(interface.app) as client:
        response = client.post(
            f"/{PATH_MODEL_ID}",
            params={"image": "https://example.com/clip.mp4", "confidence": 40},
        )

    assert response.status_code == 200, response.text
    assert response.json()["timeline"][0]["class"] == "walk"
    assert "confidence" not in adapter._model.infer.call_args.kwargs


def test_action_recognition_input_error_returns_400(monkeypatch) -> None:
    from inference_models.errors import ModelInputError

    interface, manager = _build_interface(monkeypatch, lambda_mode=False)
    manager.infer_from_request_sync.side_effect = ModelInputError(
        "Unknown V-JEPA class filter"
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/action_recognition",
            json={
                "model_id": PATH_MODEL_ID,
                "video": {"type": "url", "value": "https://example.com/clip.mp4"},
                "class_filter": ["not-a-class"],
            },
        )

    assert response.status_code == 400, response.text
    assert "Unknown V-JEPA class filter" in response.json()["message"]


@pytest.mark.parametrize("typed_endpoint", [False, True])
@pytest.mark.parametrize("disconnected", [False, True])
def test_action_routes_propagate_request_control(
    monkeypatch, typed_endpoint, disconnected
):
    from inference.core.interfaces.http import video_processing

    interface, manager = _build_interface(monkeypatch, lambda_mode=False)
    monkeypatch.setattr(video_processing.from_thread, "run", lambda *args: disconnected)

    def expire(model_id, request, *, processing_control):
        processing_control.deadline = 0
        processing_control.check()

    manager.infer_from_request_sync.side_effect = expire
    with TestClient(interface.app) as client:
        if typed_endpoint:
            response = client.post(
                "/infer/action_recognition",
                json={
                    "model_id": PATH_MODEL_ID,
                    "video": {"type": "url", "value": "https://example.com/clip.mp4"},
                },
            )
        else:
            response = client.post(
                f"/{PATH_MODEL_ID}", params={"image": "https://example.com/clip.mp4"}
            )

    assert response.status_code == (499 if disconnected else 504)
    assert manager.infer_from_request_sync.call_count == (0 if disconnected else 1)
