import asyncio
import logging
import urllib.parse

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from inference_models.weights_providers import roboflow as roboflow_provider

import inference_server.auth as auth_mod
from inference_server import configuration
from inference_server.errors import AuthBackendUnavailable
from inference_server.hosted import dedicated_auth, serverless_auth
from tests.unit_tests.test_error_classification import (
    _FakeResp,
    _http_scope,
    _run_middleware,
)

GATEWAY = "https://gw.example"
API_BASE = "https://api.example.com"


class RecordingSession:
    def __init__(self, response=None, error=None):
        self.requests = []
        self.closed = False
        self._response = response or _FakeResp(200, {"workspace": "ws-1"})
        self._error = error

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture(autouse=True)
def fresh_auth_state(monkeypatch):
    monkeypatch.setattr(auth_mod, "_cache", {})
    monkeypatch.setattr(auth_mod, "_inflight", {})
    monkeypatch.setattr(auth_mod, "API_BASE_URL", API_BASE)


@pytest.fixture
def session(monkeypatch):
    recording = RecordingSession()
    monkeypatch.setattr(auth_mod, "_get_session", lambda: recording)
    return recording


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(roboflow_provider, "SECURE_GATEWAY", GATEWAY)


@pytest.fixture
def no_gateway(monkeypatch):
    monkeypatch.setattr(roboflow_provider, "SECURE_GATEWAY", None)


def test_default_lookup_is_the_direct_call_even_with_a_gateway(session, gateway):
    asyncio.run(auth_mod.validate_api_key("k1"))

    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]


def test_wrapped_lookup_goes_through_the_gateway(session, gateway):
    asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))

    ((url, kwargs),) = session.requests
    assert "params" not in kwargs
    requested = str(url)
    assert requested.startswith(f"{GATEWAY}/proxy?url=")
    proxied = urllib.parse.unquote(requested.split("?url=", 1)[1])
    assert proxied == f"{API_BASE}/?api_key=k1&nocache=true"


def test_wrapped_lookup_is_identical_to_direct_without_a_gateway(session, no_gateway):
    asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))

    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]


def test_cache_is_not_shared_between_wrapped_and_direct_lookups_with_a_gateway(
    session, gateway
):
    wrapped = asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))
    direct = asyncio.run(auth_mod.validate_api_key("k1"))
    wrapped_again = asyncio.run(
        auth_mod.validate_api_key("k1", through_secure_gateway=True)
    )

    assert wrapped == direct == wrapped_again == (True, "ws-1")
    assert len(session.requests) == 2


def test_cache_is_shared_between_wrapped_and_direct_lookups_without_a_gateway(
    session, no_gateway
):
    wrapped = asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))
    direct = asyncio.run(auth_mod.validate_api_key("k1"))

    assert wrapped == direct == (True, "ws-1")
    assert len(session.requests) == 1


class _PendingThenTimeout:
    def __init__(self, release):
        self._release = release

    async def __aenter__(self):
        await self._release.wait()
        raise asyncio.TimeoutError()

    async def __aexit__(self, *args):
        return False


class SplitSession:
    def __init__(self, wrapped_response):
        self.requests = []
        self.wrapped_response = wrapped_response

    def get(self, url, **kwargs):
        self.requests.append(str(url))
        if str(url).startswith(GATEWAY):
            return self.wrapped_response
        return _FakeResp(200, {"workspace": "ws-1"})


def test_direct_lookup_does_not_receive_a_pending_wrapped_failure(monkeypatch, gateway):
    async def scenario():
        release = asyncio.Event()
        split = SplitSession(_PendingThenTimeout(release))
        monkeypatch.setattr(auth_mod, "_get_session", lambda: split)
        wrapped = asyncio.ensure_future(
            auth_mod.validate_api_key("k1", through_secure_gateway=True)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        direct = await asyncio.wait_for(auth_mod.validate_api_key("k1"), timeout=2)
        release.set()
        with pytest.raises(AuthBackendUnavailable):
            await wrapped

        return direct

    assert asyncio.run(scenario()) == (True, "ws-1")


def test_direct_lookup_does_its_own_request_after_a_wrapped_rejection(
    monkeypatch, gateway
):
    split = SplitSession(_FakeResp(403))
    monkeypatch.setattr(auth_mod, "_get_session", lambda: split)

    wrapped = asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))
    direct = asyncio.run(auth_mod.validate_api_key("k1"))

    assert wrapped == (False, None)
    assert direct == (True, "ws-1")
    assert len(split.requests) == 2


def test_wrapped_lookup_does_its_own_request_after_a_direct_rejection(
    monkeypatch, gateway
):
    class DirectRejected(SplitSession):
        def get(self, url, **kwargs):
            self.requests.append(str(url))
            if str(url).startswith(GATEWAY):
                return _FakeResp(200, {"workspace": "ws-1"})
            return _FakeResp(403)

    split = DirectRejected(None)
    monkeypatch.setattr(auth_mod, "_get_session", lambda: split)

    direct = asyncio.run(auth_mod.validate_api_key("k1"))
    wrapped = asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))

    assert direct == (False, None)
    assert wrapped == (True, "ws-1")
    assert len(split.requests) == 2


def test_cache_limit_holds_over_wrapped_and_direct_entries(
    monkeypatch, session, gateway
):
    monkeypatch.setattr(auth_mod, "_MAX_CACHE_SIZE", 3)

    async def scenario():
        for index in range(4):
            await auth_mod.validate_api_key(f"k{index}")
            await auth_mod.validate_api_key(f"k{index}", through_secure_gateway=True)

    asyncio.run(scenario())

    assert len(auth_mod._cache) <= 4


def test_v2_bearer_auth_performs_the_direct_lookup_with_a_gateway(session, gateway):
    sent, downstream = _run_middleware(
        _http_scope("/v2/models/infer", [(b"authorization", b"Bearer k1")]),
        validate=auth_mod.validate_api_key,
    )

    assert downstream is True
    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]


def _dedicated_client():
    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe():
        return JSONResponse({})

    inner.add_middleware(dedicated_auth.DedicatedAuthMiddleware)
    return TestClient(inner)


@pytest.fixture
def allow_list(monkeypatch):
    monkeypatch.setattr(configuration, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", "ws-url")
    monkeypatch.setattr(
        configuration, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", ["ws-1"]
    )


def test_dedicated_lookup_goes_through_the_gateway(session, gateway, allow_list):
    response = _dedicated_client().get("/some/route?api_key=k1")

    assert response.status_code == 200
    ((url, _),) = session.requests
    assert str(url).startswith(f"{GATEWAY}/proxy?url=")


def test_dedicated_lookup_is_direct_without_a_gateway(session, no_gateway, allow_list):
    response = _dedicated_client().get("/some/route?api_key=k1")

    assert response.status_code == 200
    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]


def test_serverless_lookup_goes_through_the_gateway(session, gateway):
    denial, entry = asyncio.run(
        serverless_auth._authorize_without_credits(
            "k1", ("k1", False), through_secure_gateway=True
        )
    )

    assert denial is None and entry.workspace_id == "ws-1"
    ((url, _),) = session.requests
    assert str(url).startswith(f"{GATEWAY}/proxy?url=")


def test_serverless_lookup_is_direct_without_a_gateway(session, no_gateway):
    asyncio.run(
        serverless_auth._authorize_without_credits(
            "k1", ("k1", False), through_secure_gateway=True
        )
    )

    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]


def test_wrapped_failure_does_not_log_or_raise_the_url_with_the_api_key(
    monkeypatch, gateway
):
    leaking = RecordingSession(
        error=OSError(f"Connection timeout to host {GATEWAY}/proxy?url=secret-k1")
    )
    monkeypatch.setattr(auth_mod, "_get_session", lambda: leaking)
    logged = []

    class _Handler(logging.Handler):
        def emit(self, record):
            logged.append(self.format(record))

    handler = _Handler()
    handler.setFormatter(logging.Formatter("%(message)s %(exc_text)s"))
    logger = logging.getLogger("inference_server.auth")
    logger.addHandler(handler)
    try:
        with pytest.raises(AuthBackendUnavailable) as raised:
            asyncio.run(auth_mod.validate_api_key("k1", through_secure_gateway=True))
    finally:
        logger.removeHandler(handler)

    assert "secret-k1" not in str(raised.value)
    assert not [line for line in logged if "secret-k1" in line]


class ClosableSession:
    def __init__(self, *args, **kwargs):
        self.closed = False

    async def close(self):
        self.closed = True


def test_close_session_closes_the_session_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(auth_mod, "_session", None)
    monkeypatch.setattr(auth_mod.aiohttp, "ClientSession", ClosableSession)

    async def scenario():
        opened = auth_mod._get_session()
        await auth_mod.close_session()
        closed_after_first = opened.closed
        await auth_mod.close_session()
        reopened = auth_mod._get_session()

        return opened, closed_after_first, reopened

    opened, closed_after_first, reopened = asyncio.run(scenario())

    assert closed_after_first is True
    assert reopened is not opened and reopened.closed is False
    assert auth_mod._session is reopened


def test_close_session_without_a_session_is_a_no_op(monkeypatch):
    monkeypatch.setattr(auth_mod, "_session", None)

    asyncio.run(auth_mod.close_session())

    assert auth_mod._session is None


@pytest.mark.asyncio
async def test_lifespan_shutdown_closes_the_auth_session(monkeypatch):
    import inference_server.app as app_mod

    class _StubProxy:
        async def start(self):
            pass

        async def shutdown(self):
            pass

    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: _StubProxy()
    )
    monkeypatch.setattr(auth_mod, "_session", None)
    monkeypatch.setattr(auth_mod.aiohttp, "ClientSession", ClosableSession)
    opened = auth_mod._get_session()

    async with app_mod._lifespan(app_mod.app):
        assert opened.closed is False

    assert opened.closed is True


def test_a_new_session_is_used_after_close_session(monkeypatch):
    monkeypatch.setattr(auth_mod, "_session", None)
    sessions = []

    class _OkSession(ClosableSession):
        def __init__(self, *args, **kwargs):
            super().__init__()
            sessions.append(self)

        def get(self, url, **kwargs):
            return _FakeResp(200, {"workspace": "ws-1"})

    monkeypatch.setattr(auth_mod.aiohttp, "ClientSession", _OkSession)

    async def scenario():
        first = await auth_mod.validate_api_key("k1")
        await auth_mod.close_session()
        await auth_mod.close_session()
        second = await auth_mod.validate_api_key("k2")

        return first, second

    first, second = asyncio.run(scenario())

    assert first == second == (True, "ws-1")
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert sessions[0].closed is True and sessions[1].closed is False


def _v2_composed_app(middleware):
    from inference_server.routers import v2_server

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(v2_server.router)

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe():
        return JSONResponse({})

    app.add_middleware(middleware)
    return app


@pytest.mark.parametrize(
    "root_path,v2_path,legacy_path",
    [
        ("", "/v2/server/health", "/some/route"),
        ("/service", "/service/v2/server/health", "/service/some/route"),
    ],
)
def test_dedicated_v2_lookup_is_direct_and_legacy_lookup_is_wrapped(
    session, gateway, allow_list, root_path, v2_path, legacy_path
):
    client = TestClient(
        _v2_composed_app(dedicated_auth.DedicatedAuthMiddleware), root_path=root_path
    )

    assert client.get(f"{v2_path}?api_key=k1").status_code == 200
    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]

    session.requests.clear()
    auth_mod._cache.clear()
    assert client.get(f"{legacy_path}?api_key=k1").status_code == 200
    ((url, _),) = session.requests
    assert str(url).startswith(f"{GATEWAY}/proxy?url=")


@pytest.mark.parametrize(
    "root_path,v2_path,legacy_path",
    [
        ("", "/v2/server/health", "/some/route"),
        ("/service", "/service/v2/server/health", "/service/some/route"),
    ],
)
def test_serverless_v2_lookup_is_direct_and_legacy_lookup_is_wrapped(
    monkeypatch, session, gateway, root_path, v2_path, legacy_path
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", "s3cret")
    serverless_auth._cache.clear()
    client = TestClient(
        _v2_composed_app(serverless_auth.ServerlessAuthMiddleware),
        root_path=root_path,
    )
    non_billable = "countinference=false&service_secret=s3cret"

    assert client.get(f"{v2_path}?api_key=k1&{non_billable}").status_code == 200
    assert session.requests == [
        (f"{API_BASE}/", {"params": {"api_key": "k1", "nocache": "true"}})
    ]

    session.requests.clear()
    auth_mod._cache.clear()
    serverless_auth._cache.clear()
    assert client.get(f"{legacy_path}?api_key=k1&{non_billable}").status_code == 200
    ((url, _),) = session.requests
    assert str(url).startswith(f"{GATEWAY}/proxy?url=")
