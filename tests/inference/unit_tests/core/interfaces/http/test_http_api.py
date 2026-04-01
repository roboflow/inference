from unittest.mock import AsyncMock, MagicMock

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
    ok: bool = True


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
