import asyncio
import logging
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from starlette.requests import Request
from starlette.testclient import TestClient

import inference.core.interfaces.http.http_api as http_api
from inference.core.constants import WORKSPACE_ID_HEADER
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
    RoboflowAPIUsagePausedError,
    WorkspaceLoadError,
)
from inference.core.interfaces.http.appliance_authorization import (
    MISSING_API_KEY_MESSAGE,
    ApplianceKeyAuthorizer,
    read_request_api_key,
    request_skips_api_key_check,
)
from inference.core.roboflow_api import ServerlessUsageCheckResponse

API_KEY = "appliance-key-7f3a"


def _under_cap(workspace_id: str = "workspace-a") -> ServerlessUsageCheckResponse:
    return ServerlessUsageCheckResponse(
        status_code=200,
        workspace_id=workspace_id,
        workspace_db_id="db-id-a",
        under_cap=True,
    )


def _over_cap(reason: str = "credit_cap_exceeded") -> ServerlessUsageCheckResponse:
    return ServerlessUsageCheckResponse(
        status_code=402,
        workspace_id="workspace-a",
        workspace_db_id="db-id-a",
        under_cap=False,
        error={
            "type": "AccessException",
            "reason": reason,
            "message": f"api_key={API_KEY} see https://app/settings/plan",
            "planCycleUsed": 1500.5,
            "cycleCreditCap": 1500,
        },
    )


INFER_PATH = "/infer/lmm/florence-2-base"


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app

    def set_stream_manager_client(self, stream_manager_client) -> None:
        pass


class _DummyResponse(BaseModel):
    ok: bool = True


def _inference_body(**extra) -> dict:
    body = {
        "model_id": "florence-2-base",
        "image": {"type": "url", "value": "https://example.com/test.jpg"},
        "prompt": "caption",
    }
    body.update(extra)

    return body


def _patch_modes(monkeypatch, *, appliance: bool, serverless: bool = False) -> None:
    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "ON_PREMISE_APPLIANCE", appliance)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", serverless)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)


def _model_manager() -> MagicMock:
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    model_manager.infer_from_request_sync.return_value = _DummyResponse()

    return model_manager


def _usage_check_mock(outcome) -> AsyncMock:
    if isinstance(outcome, Exception):
        return AsyncMock(side_effect=outcome)

    return AsyncMock(return_value=outcome)


def _build_appliance_interface(monkeypatch, lookup: AsyncMock):
    _patch_modes(monkeypatch, appliance=True)
    monkeypatch.setattr(http_api, "get_serverless_usage_check_async", lookup)

    return http_api.HttpInterface(model_manager=_model_manager())


def test_query_key_is_checked_and_workspace_header_is_set(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.post(
            INFER_PATH, params={"api_key": API_KEY}, json=_inference_body()
        )

    assert response.status_code == 200
    assert response.headers[WORKSPACE_ID_HEADER] == "workspace-a"
    lookup.assert_awaited_once_with(api_key=API_KEY)


def test_bearer_header_key_is_accepted(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.post(
            INFER_PATH,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=_inference_body(),
        )

    assert response.status_code == 200
    lookup.assert_awaited_once_with(api_key=API_KEY)


def test_header_key_is_checked_before_the_json_body_key(monkeypatch) -> None:
    # The routes prefer the header key over the body key (api_key_fallback,
    # api_key_override), so the middleware must check the key the route uses.
    lookup = AsyncMock(return_value=_under_cap())
    model_manager = _model_manager()
    _patch_modes(monkeypatch, appliance=True)
    monkeypatch.setattr(http_api, "get_serverless_usage_check_async", lookup)
    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as client:
        response = client.post(
            "/model/add",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model_id": "project-a/1", "api_key": "body-key"},
        )

    assert response.status_code == 200
    lookup.assert_awaited_once_with(api_key=API_KEY)
    assert model_manager.add_model.call_args.args[1] == API_KEY


def test_json_body_key_is_checked_without_a_header(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.post(INFER_PATH, json=_inference_body(api_key=API_KEY))

    assert response.status_code == 200
    lookup.assert_awaited_once_with(api_key=API_KEY)


def test_empty_query_key_wins_and_counts_as_no_key(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.post(
            INFER_PATH,
            params={"api_key": ""},
            json=_inference_body(api_key=API_KEY),
        )

    assert response.status_code == 401
    lookup.assert_not_awaited()


def test_request_without_key_is_refused(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.post(INFER_PATH, json=_inference_body())

    assert response.status_code == 401
    assert response.json() == {"status": 401, "message": MISSING_API_KEY_MESSAGE}
    lookup.assert_not_awaited()


def test_form_body_key_is_not_read(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.post(INFER_PATH, data={"api_key": API_KEY})

    assert response.status_code == 401
    lookup.assert_not_awaited()


@pytest.mark.parametrize(
    "outcome, expected_status, expected_message",
    [
        (RoboflowAPINotAuthorizedError("revoked"), 401, "Unauthorized api_key."),
        (
            ServerlessUsageCheckResponse(status_code=401),
            401,
            "Unauthorized api_key.",
        ),
        (
            _over_cap(),
            403,
            "Workspace workspace-a has used its inference quota (1500.5 of 1500 "
            "credits). Ask your organization administrator to raise the workspace "
            "quota.",
        ),
        (
            _over_cap(reason="workspace_restricted"),
            403,
            "Workspace workspace-a is restricted. Ask your organization "
            "administrator to restore access.",
        ),
        (
            RoboflowAPIForbiddenError("scope"),
            403,
            "The api_key is not allowed to use this inference server.",
        ),
        (
            RoboflowAPIUsagePausedError("paused"),
            403,
            "Usage of this api_key is paused. Ask your workspace administrator to "
            "re-enable it.",
        ),
    ],
)
def test_refused_key_answer_is_cached(
    monkeypatch, outcome, expected_status, expected_message
) -> None:
    lookup = _usage_check_mock(outcome)
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        responses = [
            client.post(INFER_PATH, params={"api_key": API_KEY}, json=_inference_body())
            for _ in range(2)
        ]

    for response in responses:
        assert response.status_code == expected_status
        assert response.json() == {
            "status": expected_status,
            "message": expected_message,
        }
    assert lookup.await_count == 1


@pytest.mark.parametrize(
    "outcome, error_type",
    [
        (RoboflowAPIConnectionError("down"), "RoboflowAPIConnectionError"),
        (RoboflowAPITimeoutError("slow"), "RoboflowAPITimeoutError"),
        (
            RoboflowAPIUnsuccessfulRequestError("500"),
            "RoboflowAPIUnsuccessfulRequestError",
        ),
        (RoboflowAPINotNotFoundError("404"), "RoboflowAPINotNotFoundError"),
        (
            MalformedRoboflowAPIResponseError("not json"),
            "MalformedRoboflowAPIResponseError",
        ),
        (WorkspaceLoadError("incomplete usage check"), "WorkspaceLoadError"),
        (
            ServerlessUsageCheckResponse(status_code=200, workspace_id="w"),
            "UsageCheckStatus200",
        ),
    ],
)
def test_unreachable_appliance_api_fails_closed_and_is_not_cached(
    monkeypatch, outcome, error_type
) -> None:
    lookup = _usage_check_mock(outcome)
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        responses = [
            client.post(INFER_PATH, params={"api_key": API_KEY}, json=_inference_body())
            for _ in range(2)
        ]

    for response in responses:
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert response.json()["message"] == (
            "The api_key could not be verified because the appliance API did not "
            f"answer ({error_type}). The request was refused."
        )
    assert lookup.await_count == 2


def test_accepted_key_is_cached(monkeypatch) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        for _ in range(3):
            response = client.post(
                INFER_PATH, params={"api_key": API_KEY}, json=_inference_body()
            )
            assert response.status_code == 200

    assert lookup.await_count == 1


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/"),
        ("GET", "/info"),
        ("GET", "/healthz"),
        ("GET", "/readiness"),
        ("GET", "/openapi.json"),
        ("GET", "/model/registry"),
        ("OPTIONS", INFER_PATH),
    ],
)
def test_probe_and_listing_paths_need_no_key(monkeypatch, method, path) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.request(method, path)

    assert response.status_code != 401
    lookup.assert_not_awaited()


@pytest.mark.parametrize(
    "method, path",
    [
        ("DELETE", "/model/registry"),
        ("POST", "/model/registry"),
        ("POST", "/model/clear"),
        ("GET", "/device/stats"),
        ("HEAD", "/device/stats"),
        ("POST", "/workflows/run"),
    ],
)
def test_other_methods_and_management_paths_need_a_key(
    monkeypatch, method, path
) -> None:
    lookup = AsyncMock(return_value=_under_cap())
    interface = _build_appliance_interface(monkeypatch, lookup)

    with TestClient(interface.app) as client:
        response = client.request(method, path)

    assert response.status_code == 401
    lookup.assert_not_awaited()


def test_no_answer_or_log_line_contains_the_key(monkeypatch, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    statuses = []
    for outcome in (
        RoboflowAPINotAuthorizedError("revoked"),
        RoboflowAPIUsagePausedError("paused"),
        _over_cap(),
        RoboflowAPIConnectionError("down"),
    ):
        lookup = _usage_check_mock(outcome)
        interface = _build_appliance_interface(monkeypatch, lookup)
        with TestClient(interface.app) as client:
            response = client.post(
                INFER_PATH, params={"api_key": API_KEY}, json=_inference_body()
            )
        statuses.append(response.status_code)
        assert API_KEY not in response.text

    assert statuses == [401, 403, 403, 503]
    # The test client logs its own request URL; only server records count.
    server_records = [
        record.getMessage()
        for record in caplog.records
        if not record.name.startswith("httpx")
    ]
    assert server_records
    assert not [message for message in server_records if API_KEY in message]


@pytest.mark.parametrize(
    "setting, value",
    [
        ("GCP_SERVERLESS", True),
        ("LAMBDA", True),
        ("OFFLINE_MODE", True),
        ("DEDICATED_DEPLOYMENT_WORKSPACE_URL", "workspace-a"),
        ("WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", ["workspace-a"]),
    ],
)
def test_appliance_mode_refuses_a_second_auth_policy(
    monkeypatch, setting, value
) -> None:
    _patch_modes(monkeypatch, appliance=True)
    monkeypatch.setattr(http_api, setting, value)

    with pytest.raises(RuntimeError, match="ON_PREMISE_APPLIANCE cannot be combined"):
        http_api.HttpInterface(model_manager=_model_manager())


def _route_table(interface) -> List[Tuple[str, Tuple[str, ...]]]:
    return sorted(
        (route.path, tuple(sorted(getattr(route, "methods", None) or ())))
        for route in interface.app.routes
    )


def _middleware_stack(interface) -> List[str]:
    stack = []
    for middleware in interface.app.user_middleware:
        dispatch = middleware.kwargs.get("dispatch")
        stack.append(dispatch.__name__ if dispatch else middleware.cls.__name__)

    return stack


@pytest.mark.parametrize("serverless", [False, True])
def test_mode_off_adds_no_middleware_and_no_route(monkeypatch, serverless) -> None:
    _patch_modes(monkeypatch, appliance=False, serverless=serverless)
    interface_off = http_api.HttpInterface(model_manager=_model_manager())

    assert "check_authorization_on_premise_appliance" not in _middleware_stack(
        interface_off
    )
    if serverless:
        return

    _patch_modes(monkeypatch, appliance=True)
    interface_on = http_api.HttpInterface(model_manager=_model_manager())

    assert _route_table(interface_on) == _route_table(interface_off)
    stack_on = _middleware_stack(interface_on)
    stack_on.remove("check_authorization_on_premise_appliance")
    assert stack_on == _middleware_stack(interface_off)


def _request(
    method: str, path: str, body: bytes = b"", content_type: str = ""
) -> Request:
    headers = [(b"content-length", str(len(body)).encode())]
    if content_type:
        headers.append((b"content-type", content_type.encode()))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


@pytest.mark.parametrize(
    "method, body, content_type, expected",
    [
        ("GET", b"", "", True),
        ("POST", b'{"blocks": []}', "application/json", True),
        (
            "POST",
            b'{"dynamic_blocks_definitions": [{"type": "x"}]}',
            "application/json",
            False,
        ),
        ("POST", b"", "", False),
        ("POST", b"not json", "application/json", False),
    ],
)
def test_block_description_needs_a_key_only_with_dynamic_blocks(
    method, body, content_type, expected
) -> None:
    request = _request(
        method, "/workflows/blocks/describe", body=body, content_type=content_type
    )

    assert asyncio.run(request_skips_api_key_check(request)) is expected


def test_host_header_cannot_change_the_checked_path() -> None:
    request = _request("POST", INFER_PATH)
    request.scope["headers"].append((b"host", b"x?/docs"))

    assert asyncio.run(request_skips_api_key_check(request)) is False
    assert asyncio.run(read_request_api_key(request)) is None


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_authorizer_expires_accepted_and_refused_keys() -> None:
    clock = _ManualClock()
    lookup = AsyncMock(
        side_effect=[_under_cap(), ServerlessUsageCheckResponse(status_code=401)]
    )
    authorizer = ApplianceKeyAuthorizer(
        lookup,
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=10,
        lookup_timeout_seconds=5,
        max_concurrent_lookups=8,
        timer=clock,
    )

    async def scenario() -> list:
        decisions = [await authorizer.authorize(API_KEY)]
        clock.now = 299
        decisions.append(await authorizer.authorize(API_KEY))
        clock.now = 301
        decisions.append(await authorizer.authorize(API_KEY))
        clock.now = 330
        decisions.append(await authorizer.authorize(API_KEY))
        return decisions

    decisions = asyncio.run(scenario())

    assert [d.status_code for d in decisions] == [200, 200, 401, 401]
    assert lookup.await_count == 2


def test_authorizer_shares_one_lookup_between_concurrent_requests() -> None:
    calls = []

    async def slow_lookup(api_key: str) -> ServerlessUsageCheckResponse:
        calls.append(api_key)
        await asyncio.sleep(0.05)
        return _under_cap()

    authorizer = ApplianceKeyAuthorizer(
        slow_lookup,
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=10,
        lookup_timeout_seconds=5,
        max_concurrent_lookups=8,
    )

    async def scenario() -> list:
        return await asyncio.gather(*(authorizer.authorize(API_KEY) for _ in range(20)))

    decisions = asyncio.run(scenario())

    assert len(calls) == 1
    assert {(d.workspace_id, d.workspace_db_id) for d in decisions} == {
        ("workspace-a", "db-id-a")
    }


def test_authorizer_lookup_survives_a_cancelled_waiter() -> None:
    calls = []

    async def slow_lookup(api_key: str) -> ServerlessUsageCheckResponse:
        calls.append(api_key)
        await asyncio.sleep(0.05)
        return _under_cap()

    authorizer = ApplianceKeyAuthorizer(
        slow_lookup,
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=10,
        lookup_timeout_seconds=5,
        max_concurrent_lookups=8,
    )

    async def scenario():
        first = asyncio.ensure_future(authorizer.authorize(API_KEY))
        second = asyncio.ensure_future(authorizer.authorize(API_KEY))
        await asyncio.sleep(0.01)
        first.cancel()
        return await second

    decision = asyncio.run(scenario())

    assert decision.status_code == 200
    assert len(calls) == 1


def test_authorizer_cache_size_is_bounded() -> None:
    lookup = AsyncMock(return_value=_under_cap())
    authorizer = ApplianceKeyAuthorizer(
        lookup,
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=2,
        lookup_timeout_seconds=5,
        max_concurrent_lookups=8,
    )

    async def scenario() -> None:
        for api_key in ("key-1", "key-2", "key-3", "key-1"):
            await authorizer.authorize(api_key)

    asyncio.run(scenario())

    assert [call.args[0] for call in lookup.await_args_list] == [
        "key-1",
        "key-2",
        "key-3",
        "key-1",
    ]


def test_quota_message_leaves_out_amounts_the_api_did_not_send() -> None:
    over_cap = ServerlessUsageCheckResponse(
        status_code=402,
        workspace_id=None,
        under_cap=False,
        error={"reason": "credit_cap_exceeded", "cycleCreditCap": "none"},
    )
    authorizer = ApplianceKeyAuthorizer(
        AsyncMock(return_value=over_cap),
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=10,
        lookup_timeout_seconds=5,
        max_concurrent_lookups=8,
    )

    decision = asyncio.run(authorizer.authorize(API_KEY))

    assert decision.status_code == 403
    assert decision.message == (
        "The workspace has used its inference quota. Ask your organization "
        "administrator to raise the workspace quota."
    )


def test_authorizer_answers_503_when_the_usage_check_is_slow() -> None:
    async def hung_usage_check(api_key: str) -> ServerlessUsageCheckResponse:
        await asyncio.sleep(10)
        return _under_cap()

    authorizer = ApplianceKeyAuthorizer(
        hung_usage_check,
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=10,
        lookup_timeout_seconds=0.05,
        max_concurrent_lookups=8,
    )

    async def scenario() -> list:
        return await asyncio.wait_for(
            asyncio.gather(*(authorizer.authorize(API_KEY) for _ in range(5))),
            timeout=2,
        )

    decisions = asyncio.run(scenario())

    assert {d.status_code for d in decisions} == {503}
    assert "TimeoutError" in decisions[0].message
    assert decisions[0].retry_after_seconds == 5


def test_authorizer_bounds_concurrent_usage_checks() -> None:
    in_flight = 0
    peak = 0

    async def slow_usage_check(api_key: str) -> ServerlessUsageCheckResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return ServerlessUsageCheckResponse(status_code=401)

    authorizer = ApplianceKeyAuthorizer(
        slow_usage_check,
        ttl_seconds=300,
        negative_ttl_seconds=30,
        max_size=100,
        lookup_timeout_seconds=5,
        max_concurrent_lookups=3,
    )

    async def scenario() -> list:
        return await asyncio.gather(
            *(authorizer.authorize(f"key-{index}") for index in range(20))
        )

    decisions = asyncio.run(scenario())

    assert peak == 3
    assert {d.status_code for d in decisions} == {401}
