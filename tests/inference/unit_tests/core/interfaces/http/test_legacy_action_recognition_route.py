"""Regression test for the legacy ``POST /{dataset_id}/{version_id}`` route
when the task is action recognition.

Under Lambda the route reads ``request_model_id`` from the authorizer, and it
differs from the ``model_id`` in the path. ``add_model`` registers the model
under the alias, which is ``model_id``, so an inference call naming
``request_model_id`` finds nothing. Off Lambda the two are equal by
assignment, so only a Lambda-shaped request shows the difference.
"""

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
