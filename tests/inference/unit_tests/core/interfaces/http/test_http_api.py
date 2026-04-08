from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel
from starlette.testclient import TestClient

from inference.core.constants import WORKSPACE_ID_HEADER
from inference.core.roboflow_api import ServerlessUsageCheckResponse


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


class _DummyResponse(BaseModel):
    ok: bool = True


def _make_inference_request() -> dict:
    return {
        "model_id": "florence-2-base",
        "image": {
            "type": "url",
            "value": "https://example.com/test.jpg",
        },
        "prompt": "caption",
    }


def _build_serverless_interface(monkeypatch, usage_check_result):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", True)
    usage_check_mock = AsyncMock(return_value=usage_check_result)
    monkeypatch.setattr(
        http_api,
        "get_serverless_usage_check_async",
        usage_check_mock,
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.infer_from_request_sync.return_value = _DummyResponse()
    interface = http_api.HttpInterface(model_manager=model_manager)
    return interface, model_manager, usage_check_mock


def test_infer_lmm_with_model_id_uses_alias_registry_key(monkeypatch) -> None:
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
    model_manager.infer_from_request_sync.return_value = _DummyResponse()

    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json={
                "model_id": "florence-2-base",
                "image": {
                    "type": "url",
                    "value": "https://example.com/test.jpg",
                },
                "prompt": "caption",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    model_manager.add_model.assert_called_once_with(
        "florence-pretrains/3",
        "query-api-key",
        model_id_alias="florence-2-base",
        countinference=None,
        service_secret=None,
    )
    model_manager.infer_from_request_sync.assert_called_once()
    assert (
        model_manager.infer_from_request_sync.call_args.args[0] == "florence-2-base"
    )
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.model_id == "florence-2-base"
    assert inference_request.api_key == "query-api-key"


def test_serverless_auth_middleware_allows_authorized_key_and_caches(
    monkeypatch,
) -> None:
    interface, _, usage_check_mock = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="workspace-1",
            under_cap=True,
        ),
    )

    with TestClient(interface.app) as client:
        first_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )
        second_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.headers[WORKSPACE_ID_HEADER] == "workspace-1"
    assert second_response.headers[WORKSPACE_ID_HEADER] == "workspace-1"
    assert usage_check_mock.await_count == 1


def test_serverless_auth_middleware_caches_unauthorized_response(monkeypatch) -> None:
    interface, model_manager, usage_check_mock = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(status_code=401),
    )

    with TestClient(interface.app) as client:
        first_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )
        second_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )

    assert first_response.status_code == 401
    assert second_response.status_code == 401
    assert (
        first_response.json()["message"]
        == "Unauthorized api_key. This key is not authorized for serverless inference."
    )
    assert second_response.json() == first_response.json()
    assert usage_check_mock.await_count == 1
    model_manager.add_model.assert_not_called()
    model_manager.infer_from_request_sync.assert_not_called()


def test_serverless_auth_middleware_caches_payment_required_response(
    monkeypatch,
) -> None:
    interface, model_manager, usage_check_mock = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=402,
            workspace_id="workspace-1",
            under_cap=False,
            error="Workspace is billing-restricted.",
        ),
    )

    with TestClient(interface.app) as client:
        first_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )
        second_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )

    assert first_response.status_code == 402
    assert second_response.status_code == 402
    assert (
        "cannot currently spend credits for serverless inference"
        in first_response.json()["message"]
    )
    assert "Workspace is billing-restricted." in first_response.json()["message"]
    assert second_response.json() == first_response.json()
    assert WORKSPACE_ID_HEADER not in first_response.headers
    assert usage_check_mock.await_count == 1
    model_manager.add_model.assert_not_called()
    model_manager.infer_from_request_sync.assert_not_called()
