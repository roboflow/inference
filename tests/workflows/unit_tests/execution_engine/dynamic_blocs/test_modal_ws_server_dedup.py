"""Server-side protocol v2 guarantees in modal/modal_app.py.

Covers the at-most-once execution machinery: the executed-request registry
that survives response-cache eviction, the bounded response cache, and the
session registry that answers the client's hello.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import msgpack
import pytest


class _FakeModalImage:
    @classmethod
    def debian_slim(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def entrypoint(self, *args, **kwargs):
        return self


class _FakeModalApp:
    def __init__(self, name: str):
        self.name = name

    def cls(self, *args, **kwargs):
        return lambda cls: cls


def _identity_decorator(*args, **kwargs):
    return lambda obj: obj


@pytest.fixture()
def modal_app(monkeypatch):
    fake_modal = ModuleType("modal")
    fake_modal.App = _FakeModalApp
    fake_modal.Image = _FakeModalImage
    fake_modal.parameter = lambda *args, **kwargs: None
    fake_modal.enter = _identity_decorator
    fake_modal.fastapi_endpoint = _identity_decorator
    fake_modal.asgi_app = _identity_decorator
    fake_modal.concurrent = _identity_decorator
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    modal_app_path = Path(__file__).resolve().parents[5] / "modal" / "modal_app.py"
    spec = importlib.util.spec_from_file_location(
        "modal_app_ws_dedup_test", modal_app_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTtlKeySet:
    def test_membership_add_and_refresh(self, modal_app) -> None:
        keys = modal_app._TtlKeySet(ttl_seconds=3600, max_entries=8)

        assert "a" not in keys
        keys.add("a")
        assert "a" in keys
        # refresh never adds, so an unknown key stays unknown
        assert keys.refresh("b") is False
        assert "b" not in keys
        assert keys.refresh("a") is True

    def test_empty_key_is_never_stored(self, modal_app) -> None:
        keys = modal_app._TtlKeySet(ttl_seconds=3600, max_entries=8)

        keys.add("")

        assert "" not in keys

    def test_expired_entries_drop_out(self, modal_app) -> None:
        keys = modal_app._TtlKeySet(ttl_seconds=-1, max_entries=8)

        keys.add("a")

        assert "a" not in keys

    def test_size_cap_evicts_oldest_first(self, modal_app) -> None:
        keys = modal_app._TtlKeySet(ttl_seconds=3600, max_entries=2)

        keys.add("a")
        keys.add("b")
        keys.add("c")

        assert "a" not in keys
        assert "b" in keys and "c" in keys

    def test_refresh_protects_an_entry_from_eviction(self, modal_app) -> None:
        keys = modal_app._TtlKeySet(ttl_seconds=3600, max_entries=2)
        keys.add("a")
        keys.add("b")

        keys.refresh("a")
        keys.add("c")

        assert "a" in keys
        assert "b" not in keys


class TestResponseCache:
    def test_round_trip_and_entry_cap(self, modal_app) -> None:
        cache = modal_app._WsResponseCache(
            max_entries=2, max_bytes=10**9, ttl_seconds=60
        )

        cache.put("1", b"a")
        cache.put("2", b"b")
        cache.put("3", b"c")

        assert cache.get("1") is None
        assert cache.get("2") == b"b"
        assert cache.get("3") == b"c"

    def test_byte_cap_evicts_least_recently_used(self, modal_app) -> None:
        cache = modal_app._WsResponseCache(
            max_entries=16, max_bytes=100, ttl_seconds=60
        )
        cache.put("a", b"x" * 40)
        cache.put("b", b"y" * 40)

        cache.get("a")  # promote a, so b is the eviction candidate
        cache.put("c", b"z" * 40)

        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_expired_entry_is_not_served(self, modal_app) -> None:
        cache = modal_app._WsResponseCache(
            max_entries=8, max_bytes=10**9, ttl_seconds=-1
        )

        cache.put("a", b"payload")

        assert cache.get("a") is None

    def test_oversized_payload_does_not_corrupt_accounting(self, modal_app) -> None:
        # A response larger than the whole budget cannot be retained; what
        # matters is that it evicts itself cleanly and the cache stays
        # usable (the executed-request registry, not this cache, is what
        # keeps execution at-most-once).
        cache = modal_app._WsResponseCache(max_entries=8, max_bytes=100, ttl_seconds=60)

        cache.put("huge", b"x" * 500)

        assert cache.get("huge") is None
        cache.put("small", b"ok")
        assert cache.get("small") == b"ok"


def _ws_app(modal_app, run_user_code):
    """Build the websocket app with user-code execution stubbed out."""
    cls = modal_app.Executor
    user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls
    executor = user_cls.__new__(user_cls)
    executor.workspace_id = "test-ws"
    user_cls.identify(executor)
    monkeyed = staticmethod(run_user_code)
    user_cls._run_user_code_ws = monkeyed.__func__
    return executor, user_cls.wsapp(executor)


class TestExecutionIsAtMostOnce:
    def test_resend_of_completed_request_is_answered_from_cache(
        self, modal_app, monkeypatch
    ) -> None:
        from fastapi.testclient import TestClient

        calls = []

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            return {"success": True, "result": {"n": len(calls)}}

        executor, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            first = msgpack.unpackb(ws.receive_bytes(), raw=False)
            ws.send_bytes(frame)
            second = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert len(calls) == 1, "resend must not run the user code again"
        assert first == second

    def test_resend_after_cache_eviction_fails_loudly_instead_of_rerunning(
        self, modal_app
    ) -> None:
        # The response cache is best effort; the executed-request registry
        # is what makes execution at-most-once. With the payload gone, a
        # resend must get an error, never a second execution.
        from fastapi.testclient import TestClient

        calls = []

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            return {"success": True, "result": {}}

        executor, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            msgpack.unpackb(ws.receive_bytes(), raw=False)
            # Simulate eviction of the payload while the request id is still
            # known to have executed.
            executor._ws_response_cache = modal_app._WsResponseCache()
            ws.send_bytes(frame)
            resent = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert len(calls) == 1
        assert resent["success"] is False
        assert resent["error_type"] == "ResponseNoLongerAvailable"
        assert resent["request_id"] == "req-1"

    def test_unserializable_result_returns_error_and_is_not_rerun(
        self, modal_app
    ) -> None:
        # A result the server cannot pack must not escape as an exception:
        # that would kill the connection with nothing cached and invite a
        # resend of a request whose side effects already happened.
        from fastapi.testclient import TestClient

        calls = []

        class _Unpackable:
            def __repr__(self):
                raise RuntimeError("cannot repr")

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            return {"success": True, "result": {"bad": _Unpackable()}}

        executor, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            first = msgpack.unpackb(ws.receive_bytes(), raw=False)
            ws.send_bytes(frame)
            second = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert len(calls) == 1, "the failed request must not be re-executed"
        assert first["success"] is False
        assert first["request_id"] == "req-1"
        # The resend is answered from the cached error, identically.
        assert second == first

    def test_session_is_registered_only_after_a_successful_execution(
        self, modal_app
    ) -> None:
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            return {"success": False, "error": "boom", "error_type": "ValueError"}

        executor, app = _ws_app(modal_app, run_user_code)
        hello = msgpack.packb(
            {"_kind": "hello", "proto": 2, "session_id": "s1"}, use_bin_type=True
        )

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(hello)
            first_hello = msgpack.unpackb(ws.receive_bytes(), raw=False)
            ws.send_bytes(
                msgpack.packb({"request_id": "r1", "inputs": {}}, use_bin_type=True)
            )
            msgpack.unpackb(ws.receive_bytes(), raw=False)
            ws.send_bytes(hello)
            second_hello = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert first_hello["session_known"] is False
        # A failed execution builds no runtime state, and asking twice must
        # not itself register the session.
        assert second_hello["session_known"] is False
        assert first_hello["container_id"] == executor._container_id


class TestSessionRegistry:
    def _executor(self, modal_app):
        cls = modal_app.Executor
        user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls

        class _Standin:
            _WS_SESSION_TTL_SECONDS = user_cls._WS_SESSION_TTL_SECONDS
            _WS_SESSION_MAX_ENTRIES = user_cls._WS_SESSION_MAX_ENTRIES
            _ws_session_seen = user_cls._ws_session_seen
            _ws_register_session = user_cls._ws_register_session

        standin = _Standin()
        standin._ws_sessions = modal_app._TtlKeySet(
            ttl_seconds=_Standin._WS_SESSION_TTL_SECONDS,
            max_entries=_Standin._WS_SESSION_MAX_ENTRIES,
        )
        return standin

    def test_unknown_session_is_not_registered_by_asking(self, modal_app) -> None:
        # Asking must not create the entry: a hello that reports "unknown"
        # and then registers would let the very next reconnect pass the
        # check silently.
        executor = self._executor(modal_app)

        assert executor._ws_session_seen("s1") is False
        assert executor._ws_session_seen("s1") is False

    def test_registered_session_is_known(self, modal_app) -> None:
        executor = self._executor(modal_app)

        executor._ws_register_session("s1")

        assert executor._ws_session_seen("s1") is True
        assert executor._ws_session_seen("s2") is False

    def test_seen_refreshes_so_an_active_session_cannot_age_out(
        self, modal_app
    ) -> None:
        # The namespaces this answers for are never evicted, so a session
        # that keeps reconnecting must not expire out of the registry.
        executor = self._executor(modal_app)
        executor._ws_sessions = modal_app._TtlKeySet(ttl_seconds=3600, max_entries=2)
        executor._ws_register_session("s1")
        executor._ws_register_session("s2")

        executor._ws_session_seen("s1")
        executor._ws_register_session("s3")

        assert executor._ws_session_seen("s1") is True
        assert executor._ws_session_seen("s2") is False
