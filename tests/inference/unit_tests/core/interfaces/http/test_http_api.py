from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel
from starlette.testclient import TestClient

from inference.core.constants import (
    PROCESSING_TIME_HEADER,
    TRACE_ID_HEADER,
    WORKSPACE_ID_HEADER,
)
from inference.core.env import CORRELATION_ID_HEADER
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


def _build_serverless_interface(
    monkeypatch,
    usage_check_result,
    workspace_lookup_result="rf-inference-benchmark",
):
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
    workspace_lookup_mock = AsyncMock(return_value=workspace_lookup_result)
    monkeypatch.setattr(
        http_api,
        "get_roboflow_workspace_async",
        workspace_lookup_mock,
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.infer_from_request_sync.return_value = _DummyResponse()
    interface = http_api.HttpInterface(model_manager=model_manager)
    return interface, model_manager, usage_check_mock, workspace_lookup_mock


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


def test_serverless_auth_middleware_logs_request_received_with_execution_id(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(http_api, "EXECUTION_ID_HEADER", "X-Execution-Id")
    log_mock = MagicMock()
    monkeypatch.setattr(http_api.logger, "info", log_mock)
    interface, _, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
            under_cap=True,
        ),
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            headers={
                CORRELATION_ID_HEADER: "request-123",
                "X-Execution-Id": "execution-123",
            },
            json=_make_inference_request(),
        )

    assert response.status_code == 200
    log_mock.assert_any_call(
        http_api.REQUEST_RECEIVED_LOG_MESSAGE,
        method="POST",
        path="/infer/lmm/florence-2-base",
        request_id="request-123",
        execution_id="execution-123",
    )


def test_serverless_auth_middleware_skips_request_received_log_when_api_logging_disabled(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "API_LOGGING_ENABLED", False)
    log_mock = MagicMock()
    monkeypatch.setattr(http_api.logger, "info", log_mock)
    interface, _, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
            under_cap=True,
        ),
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )

    assert response.status_code == 200
    request_received_calls = (
        call
        for call in log_mock.call_args_list
        if call.args and call.args[0] == http_api.REQUEST_RECEIVED_LOG_MESSAGE
    )
    assert list(request_received_calls) == []


def test_serverless_auth_middleware_allows_authorized_key_and_caches(
    monkeypatch,
) -> None:
    interface, _, usage_check_mock, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
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
    assert first_response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert second_response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert usage_check_mock.await_count == 1


def test_serverless_auth_middleware_caches_unauthorized_response(monkeypatch) -> None:
    interface, model_manager, usage_check_mock, _ = _build_serverless_interface(
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
    interface, model_manager, usage_check_mock, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=402,
            workspace_id="rf-inference-benchmark",
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
    assert first_response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert second_response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert usage_check_mock.await_count == 1
    model_manager.add_model.assert_not_called()
    model_manager.infer_from_request_sync.assert_not_called()


def test_serverless_auth_middleware_adds_observability_headers_and_logs_on_denial(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "EXECUTION_ID_HEADER", "X-Execution-Id")
    monkeypatch.setattr(http_api, "get_trace_id", lambda: "trace-123")
    denied_log_mock = MagicMock()
    monkeypatch.setattr(http_api.logger, "info", denied_log_mock)
    interface, _, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=402,
            workspace_id="rf-inference-benchmark",
            under_cap=False,
            error="Workspace is billing-restricted.",
        ),
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            headers={
                CORRELATION_ID_HEADER: "request-123",
                "X-Execution-Id": "execution-123",
            },
            json=_make_inference_request(),
        )

    assert response.status_code == 402
    assert response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert response.headers[CORRELATION_ID_HEADER] == "request-123"
    assert response.headers["X-Execution-Id"] == "execution-123"
    assert response.headers[TRACE_ID_HEADER] == "trace-123"
    assert PROCESSING_TIME_HEADER in response.headers
    denied_log_mock.assert_any_call(
        "Serverless authorization denied",
        method="POST",
        path="/infer/lmm/florence-2-base",
        status_code=402,
        denial_message=(
            "This workspace cannot currently spend credits for serverless inference. "
            "Verify billing or credit cap settings. Workspace is billing-restricted."
        ),
        request_id="request-123",
        execution_id="execution-123",
        workspace_id="rf-inference-benchmark",
        cache_hit=False,
    )


def test_serverless_auth_middleware_uses_auth_only_path_for_internal_non_billable_requests(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "ROBOFLOW_SERVICE_SECRET", "shared-secret")
    interface, _, usage_check_mock, workspace_lookup_mock = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=402,
            workspace_id="rf-inference-benchmark",
            under_cap=False,
            error="Workspace is billing-restricted.",
        ),
    )

    with TestClient(interface.app) as client:
        first_response = client.post(
            "/infer/lmm/florence-2-base",
            params={
                "api_key": "query-api-key",
                "countinference": "false",
                "service_secret": "shared-secret",
            },
            json=_make_inference_request(),
        )
        second_response = client.post(
            "/infer/lmm/florence-2-base",
            params={
                "api_key": "query-api-key",
                "countinference": "false",
                "service_secret": "shared-secret",
            },
            json=_make_inference_request(),
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert second_response.headers[WORKSPACE_ID_HEADER] == "rf-inference-benchmark"
    assert usage_check_mock.await_count == 0
    assert workspace_lookup_mock.await_count == 1


def test_serverless_auth_middleware_keeps_non_billable_and_billable_cache_entries_separate(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "ROBOFLOW_SERVICE_SECRET", "shared-secret")
    interface, model_manager, usage_check_mock, workspace_lookup_mock = (
        _build_serverless_interface(
            monkeypatch=monkeypatch,
            usage_check_result=ServerlessUsageCheckResponse(
                status_code=402,
                workspace_id="rf-inference-benchmark",
                under_cap=False,
                error="Workspace is billing-restricted.",
            ),
        )
    )

    with TestClient(interface.app) as client:
        non_billable_response = client.post(
            "/infer/lmm/florence-2-base",
            params={
                "api_key": "query-api-key",
                "countinference": "false",
                "service_secret": "shared-secret",
            },
            json=_make_inference_request(),
        )
        billable_response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "query-api-key"},
            json=_make_inference_request(),
        )

    assert non_billable_response.status_code == 200
    assert billable_response.status_code == 402
    assert usage_check_mock.await_count == 1
    assert workspace_lookup_mock.await_count == 1
    assert model_manager.infer_from_request_sync.call_count == 1
