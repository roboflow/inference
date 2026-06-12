import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from starlette.testclient import TestClient

from inference.core.constants import (
    PROCESSING_TIME_HEADER,
    TRACE_ID_HEADER,
    WORKSPACE_ID_HEADER,
)
from inference.core.env import CORRELATION_ID_HEADER
from inference.core.exceptions import RoboflowAPINotAuthorizedError
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


class _DummyArray:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class _DummyImage:
    base64_image = "depth-image"


class _DummyDepthResponse:
    response = {
        "normalized_depth": _DummyArray([[0.0]]),
        "image": _DummyImage(),
    }


def _make_inference_request() -> dict:
    return {
        "model_id": "florence-2-base",
        "image": {
            "type": "url",
            "value": "https://example.com/test.jpg",
        },
        "prompt": "caption",
    }


def _route_paths(interface):
    return {route.path for route in interface.app.routes}


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


def _build_dedicated_deployment_interface(
    monkeypatch,
    workspace_lookup_result="dedicated-workspace",
    dedicated_workspace_url="dedicated-workspace",
    workspace_lookup_side_effect=None,
    local_whitelist=None,
):
    """Build an HttpInterface with the workspace-allowlist middleware enabled.

    `local_whitelist` patches `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`.
    Default `None` preserves the historical behaviour for existing callers
    (env var unset). Pass a list to simulate the env var being set; pass
    `[]` to assert the empty-list-doesn't-enable-middleware contract.
    """
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(
        http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", dedicated_workspace_url
    )
    monkeypatch.setattr(
        http_api,
        "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
        list(local_whitelist) if local_whitelist is not None else None,
    )
    if workspace_lookup_side_effect is not None:
        workspace_lookup_mock = AsyncMock(side_effect=workspace_lookup_side_effect)
    else:
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
    return interface, model_manager, workspace_lookup_mock


def test_serverless_registers_sam3_routes_when_model_flags_are_enabled(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "CORE_MODEL_SAM3_ENABLED", True)
    monkeypatch.setattr(http_api, "SAM3_3D_OBJECTS_ENABLED", True)
    interface, _, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
            under_cap=True,
        ),
    )

    paths = _route_paths(interface)
    assert "/sam3/embed_image" in paths
    assert "/sam3_3d/infer" in paths


def test_serverless_sam3_3d_route_only_depends_on_sam3_3d_flag(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "CORE_MODEL_SAM3_ENABLED", False)
    monkeypatch.setattr(http_api, "SAM3_3D_OBJECTS_ENABLED", True)
    interface, _, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
            under_cap=True,
        ),
    )

    paths = _route_paths(interface)
    assert "/sam3/embed_image" not in paths
    assert "/sam3_3d/infer" in paths


def test_serverless_does_not_register_sam3_3d_route_when_sam3_3d_flag_is_disabled(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "CORE_MODEL_SAM3_ENABLED", True)
    monkeypatch.setattr(http_api, "SAM3_3D_OBJECTS_ENABLED", False)
    interface, _, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
            under_cap=True,
        ),
    )

    paths = _route_paths(interface)
    assert "/sam3/embed_image" in paths
    assert "/sam3_3d/infer" not in paths


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
    assert model_manager.infer_from_request_sync.call_args.args[0] == "florence-2-base"
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.model_id == "florence-2-base"
    assert inference_request.api_key == "query-api-key"


def test_depth_estimation_uses_query_api_key_for_model_loading(monkeypatch) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "DEPTH_ESTIMATION_ENABLED", True)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.infer_from_request_sync.return_value = _DummyDepthResponse()

    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/depth-estimation",
            params={"api_key": "query-api-key"},
            json={
                "model_id": "depth-anything-v3/small",
                "image": {
                    "type": "url",
                    "value": "https://example.com/test.jpg",
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {"normalized_depth": [[0.0]], "image": "depth-image"}
    model_manager.add_model.assert_called_once_with(
        "depth-anything-v3/small",
        "query-api-key",
        countinference=None,
        service_secret=None,
    )
    model_manager.infer_from_request_sync.assert_called_once()
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.model_id == "depth-anything-v3/small"
    assert inference_request.api_key == "query-api-key"

def test_depth_estimation_with_model_id_path_sets_request_model_id(monkeypatch) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "DEPTH_ESTIMATION_ENABLED", True)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.infer_from_request_sync.return_value = _DummyDepthResponse()

    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/depth-estimation/depth-anything-v3/small",
            params={"api_key": "query-api-key"},
            json={
                "image": {
                    "type": "url",
                    "value": "https://example.com/test.jpg",
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {"normalized_depth": [[0.0]], "image": "depth-image"}
    model_manager.add_model.assert_called_once_with(
        "depth-anything-v3/small",
        "query-api-key",
        countinference=None,
        service_secret=None,
    )
    model_manager.infer_from_request_sync.assert_called_once()
    inference_request = model_manager.infer_from_request_sync.call_args.args[1]
    assert inference_request.model_id == "depth-anything-v3/small"
    assert inference_request.api_key == "query-api-key"


def test_depth_estimation_with_model_id_path_rejects_body_mismatch(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "DEPTH_ESTIMATION_ENABLED", True)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0

    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/depth-estimation/depth-anything-v3/small",
            params={"api_key": "query-api-key"},
            json={
                "model_id": "depth-anything-v3/base",
                "image": {
                    "type": "url",
                    "value": "https://example.com/test.jpg",
                },
            },
        )

    assert response.status_code == 400
    assert response.json()["message"] == (
        "Model ID mismatch: path specifies 'depth-anything-v3/small' "
        "but request body specifies 'depth-anything-v3/base'"
    )
    model_manager.add_model.assert_not_called()
    model_manager.infer_from_request_sync.assert_not_called()


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


def test_startup_preload_uses_same_alias_registry_key_as_live_requests(
    monkeypatch,
) -> None:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "PRELOAD_MODELS", ["florence-2-base"])
    monkeypatch.setattr(http_api, "PINNED_MODELS", ["florence-2-base"])
    monkeypatch.setattr(http_api, "PRELOAD_API_KEY", "preload-key")
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0

    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as client:
        deadline = time.monotonic() + 1
        response = client.get("/readiness")
        while response.status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.01)
            response = client.get("/readiness")

    assert response.status_code == 200
    model_manager.add_model.assert_called_once_with(
        "florence-pretrains/3",
        "preload-key",
        model_id_alias="florence-2-base",
    )
    model_manager.pin_model.assert_called_once_with("florence-2-base")


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


def test_serverless_auth_middleware_rejects_host_header_path_injection(
    monkeypatch,
) -> None:
    # CVE-2026-48710 (BadHost): vulnerable Starlette derived request.url.path
    # from the Host header. A Host value containing `/`, `?`, or `#` could make
    # request.url.path appear to be an allowlisted route (e.g. "/docs") while
    # ASGI routed the request to an authenticated handler. Guard against both
    # the dependency regressing and the middleware drifting back to
    # request.url.path by asserting auth is still enforced when malicious Host
    # headers are sent at a protected endpoint.
    interface, model_manager, _, _ = _build_serverless_interface(
        monkeypatch=monkeypatch,
        usage_check_result=ServerlessUsageCheckResponse(
            status_code=200,
            workspace_id="rf-inference-benchmark",
            under_cap=True,
        ),
    )

    injection_hosts = [
        "testserver/docs?",
        "testserver?/docs",
        "testserver/healthz?",
        "testserver/_next/x",
        "testserver/static/x",
        "testserver#/docs",
    ]

    with TestClient(interface.app) as client:
        for host in injection_hosts:
            response = client.post(
                "/infer/lmm/florence-2-base",
                headers={"Host": host},
                json=_make_inference_request(),
            )
            assert response.status_code == 401, (
                f"Host-injection bypass for header {host!r}: expected 401, "
                f"got {response.status_code}"
            )

    model_manager.infer_from_request_sync.assert_not_called()


def test_dedicated_deployment_auth_middleware_rejects_host_header_path_injection(
    monkeypatch,
) -> None:
    # CVE-2026-48710 (BadHost) — sibling coverage for the dedicated-deployment
    # auth middleware (check_authorization). Same shape as the serverless test
    # so a future refactor that drops scope_path in one middleware but not the
    # other still trips a regression.
    interface, model_manager, workspace_lookup_mock = (
        _build_dedicated_deployment_interface(monkeypatch=monkeypatch)
    )

    injection_hosts = [
        "testserver/docs?",
        "testserver?/docs",
        "testserver/healthz?",
        "testserver/redoc?",
        "testserver/_next/x",
        "testserver/static/x",
        "testserver#/docs",
    ]

    with TestClient(interface.app) as client:
        for host in injection_hosts:
            response = client.post(
                "/infer/lmm/florence-2-base",
                headers={"Host": host},
                json=_make_inference_request(),
            )
            assert response.status_code == 401, (
                f"Host-injection bypass for header {host!r}: expected 401, "
                f"got {response.status_code}"
            )

    model_manager.infer_from_request_sync.assert_not_called()
    workspace_lookup_mock.assert_not_called()


# A representative spread of paths the workspace-allowlist middleware must
# guard. Mix of: real POST inference handlers, a non-exempt GET, a registered
# admin POST, and an unregistered route (the middleware runs before FastAPI's
# 404, so even unregistered paths must produce a 401 instead of leaking 404).
# Used to parametrise the rejection tests below so a future refactor that
# skips the middleware on any path family trips a regression.
_SECURED_GATE_TARGETS = [
    ("POST", "/infer/lmm/florence-2-base"),
    ("POST", "/infer/object_detection"),
    ("POST", "/infer/instance_segmentation"),
    ("POST", "/infer/classification"),
    ("POST", "/model/add"),
    ("GET", "/device/stats"),
    ("POST", "/some/unregistered/secured/path"),
]


def _send_secured_request(client, method, path, *, api_key=None):
    """Drive a request through the middleware for parametrised tests.

    The body is irrelevant for the middleware (it runs before route
    validation), so we send a generic JSON payload on POST and nothing on
    GET. `api_key`, when provided, is passed as a query parameter — which is
    where the middleware looks first.
    """
    kwargs = {}
    if api_key is not None:
        kwargs["params"] = {"api_key": api_key}
    if method == "POST":
        kwargs["json"] = _make_inference_request()
    return client.request(method, path, **kwargs)


def test_local_whitelist_alone_enables_middleware_and_allows_matching_workspace(
    monkeypatch,
) -> None:
    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        workspace_lookup_result="local-allowed-ws",
        dedicated_workspace_url=None,
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "key-for-allowed-ws"},
            json=_make_inference_request(),
        )

    assert response.status_code == 200
    workspace_lookup_mock.assert_awaited_once_with(api_key="key-for-allowed-ws")


@pytest.mark.parametrize("method,path", _SECURED_GATE_TARGETS)
def test_local_whitelist_alone_rejects_non_matching_workspace(
    monkeypatch, method, path
) -> None:
    interface, _, _ = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        workspace_lookup_result="some-other-ws",
        dedicated_workspace_url=None,
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = _send_secured_request(
            client, method, path, api_key="key-for-other-ws"
        )

    assert (
        response.status_code == 401
    ), f"{method} {path}: expected 401, got {response.status_code}"
    assert response.json() == {"status": 401, "message": "Unauthorized api_key"}


def test_dedicated_workspace_remains_accepted_when_local_whitelist_is_also_set(
    monkeypatch,
) -> None:
    """Union semantics — the dedicated path must keep working after the local
    whitelist is added alongside it."""
    interface, _, _ = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        workspace_lookup_result="dedicated-workspace",
        dedicated_workspace_url="dedicated-workspace",
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "key-for-dedicated"},
            json=_make_inference_request(),
        )

    assert response.status_code == 200


def test_locally_whitelisted_workspace_is_accepted_in_union_with_dedicated(
    monkeypatch,
) -> None:
    """A workspace that is in the local whitelist but does NOT match the
    dedicated URL must still be accepted when both env vars are set."""
    interface, _, _ = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        workspace_lookup_result="local-allowed-ws",
        dedicated_workspace_url="dedicated-workspace",
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "key-for-local"},
            json=_make_inference_request(),
        )

    assert response.status_code == 200


@pytest.mark.parametrize("method,path", _SECURED_GATE_TARGETS)
def test_workspace_outside_both_allowlists_is_rejected(
    monkeypatch, method, path
) -> None:
    interface, _, _ = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        workspace_lookup_result="some-other-ws",
        dedicated_workspace_url="dedicated-workspace",
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = _send_secured_request(client, method, path, api_key="key-for-other")

    assert (
        response.status_code == 401
    ), f"{method} {path}: expected 401, got {response.status_code}"


def test_local_whitelist_with_multiple_entries_accepts_each_member(monkeypatch) -> None:
    import inference.core.interfaces.http.http_api as http_api

    workspace_for_key = {
        "key-a": "ws-a",
        "key-b": "ws-b",
        "key-c": "ws-c",
    }

    async def lookup(api_key: str) -> str:
        return workspace_for_key[api_key]

    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        dedicated_workspace_url=None,
        local_whitelist=["ws-a", "ws-b", "ws-c"],
    )
    # Override the lookup with a per-key one (the helper installs a single
    # AsyncMock(return_value=...) which is fine for most tests but not for
    # this one).
    monkeypatch.setattr(
        http_api, "get_roboflow_workspace_async", AsyncMock(side_effect=lookup)
    )

    with TestClient(interface.app) as client:
        for api_key in ("key-a", "key-b", "key-c"):
            response = client.post(
                "/infer/lmm/florence-2-base",
                params={"api_key": api_key},
                json=_make_inference_request(),
            )
            assert (
                response.status_code == 200
            ), f"{api_key} should map to a whitelisted workspace"


@pytest.mark.parametrize("method,path", _SECURED_GATE_TARGETS)
def test_local_whitelist_middleware_rejects_request_without_api_key(
    monkeypatch, method, path
) -> None:
    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        dedicated_workspace_url=None,
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = _send_secured_request(client, method, path)

    assert (
        response.status_code == 401
    ), f"{method} {path}: expected 401, got {response.status_code}"
    assert response.json() == {"status": 401, "message": "Unauthorized api_key"}
    workspace_lookup_mock.assert_not_awaited()


@pytest.mark.parametrize("method,path", _SECURED_GATE_TARGETS)
def test_local_whitelist_middleware_returns_401_when_roboflow_api_rejects_key(
    monkeypatch, method, path
) -> None:
    interface, _, _ = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        dedicated_workspace_url=None,
        local_whitelist=["local-allowed-ws"],
        workspace_lookup_side_effect=RoboflowAPINotAuthorizedError("key revoked"),
    )

    with TestClient(interface.app) as client:
        response = _send_secured_request(client, method, path, api_key="revoked-key")

    assert (
        response.status_code == 401
    ), f"{method} {path}: expected 401, got {response.status_code}"


def test_local_whitelist_middleware_caches_successful_workspace_lookup(
    monkeypatch,
) -> None:
    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        workspace_lookup_result="local-allowed-ws",
        dedicated_workspace_url=None,
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        for _ in range(3):
            response = client.post(
                "/infer/lmm/florence-2-base",
                params={"api_key": "same-key"},
                json=_make_inference_request(),
            )
            assert response.status_code == 200

    assert workspace_lookup_mock.await_count == 1, (
        "After the first lookup, the api_key/workspace_id pair is cached "
        "and the upstream Roboflow API must not be hit again"
    )


@pytest.mark.parametrize(
    "exempt_path",
    [
        "/healthz",
        "/readiness",
        "/info",
        "/openapi.json",
    ],
)
def test_local_whitelist_middleware_skips_check_for_exempt_paths(
    monkeypatch, exempt_path
) -> None:
    """Liveness and metadata endpoints must remain reachable without an
    api_key even when the local-deployment allowlist is enforcing auth."""
    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        dedicated_workspace_url=None,
        local_whitelist=["local-allowed-ws"],
    )

    with TestClient(interface.app) as client:
        response = client.get(exempt_path)

    assert response.status_code == 200, f"{exempt_path} must bypass the auth middleware"
    workspace_lookup_mock.assert_not_awaited()


def test_middleware_is_not_installed_when_neither_env_var_is_set(monkeypatch) -> None:
    """Pins the gate at http_api.py:985 — when neither
    `DEDICATED_DEPLOYMENT_WORKSPACE_URL` nor
    `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` is set, the middleware is
    never installed and unauthenticated requests proceed (community default).
    """
    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        dedicated_workspace_url=None,
        local_whitelist=None,
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "anything"},
            json=_make_inference_request(),
        )

    assert response.status_code == 200
    workspace_lookup_mock.assert_not_awaited()


def test_empty_local_whitelist_alone_does_not_enable_middleware(monkeypatch) -> None:
    """An empty list is falsy, so the gate `if DEDICATED_... or WORKSPACES_...`
    must NOT install the middleware when only an empty list is provided.
    Otherwise an operator who clears the env var to `""` (which `safe_split_value`
    can collapse to an empty list depending on implementation) would silently
    enable an allowlist that rejects every api_key."""
    interface, _, workspace_lookup_mock = _build_dedicated_deployment_interface(
        monkeypatch=monkeypatch,
        dedicated_workspace_url=None,
        local_whitelist=[],
    )

    with TestClient(interface.app) as client:
        response = client.post(
            "/infer/lmm/florence-2-base",
            params={"api_key": "anything"},
            json=_make_inference_request(),
        )

    assert response.status_code == 200
    workspace_lookup_mock.assert_not_awaited()
