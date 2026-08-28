"""Server-side protocol v2 guarantees in modal/modal_app.py.

Covers the at-most-once execution machinery: the executed-request registry
that survives response-cache eviction, the bounded response cache, and the
session registry that answers the client's hello.
"""

import time

import msgpack
import pytest

from .conftest import build_ws_app as _ws_app


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

    def test_byte_cap_evicts_oldest_first(self, modal_app) -> None:
        cache = modal_app._WsResponseCache(
            max_entries=16, max_bytes=100, ttl_seconds=60
        )
        cache.put("a", b"x" * 40)
        cache.put("b", b"y" * 40)

        # A hit must NOT promote: entries stay in age order so the TTL is a
        # real TTL and the early-exit expiry scan stays correct.
        cache.get("a")
        cache.put("c", b"z" * 40)

        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_hit_does_not_extend_the_ttl(self, modal_app) -> None:
        cache = modal_app._WsResponseCache(
            max_entries=8, max_bytes=10**9, ttl_seconds=0.05
        )
        cache.put("a", b"payload")
        assert cache.get("a") == b"payload"

        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            cache.get("a")

        assert cache.get("a") is None

    def test_expired_entry_is_not_served(self, modal_app) -> None:
        cache = modal_app._WsResponseCache(
            max_entries=8, max_bytes=10**9, ttl_seconds=-1
        )

        cache.put("a", b"payload")

        assert cache.get("a") is None

    def test_oversized_payload_is_refused_without_draining_the_cache(
        self, modal_app
    ) -> None:
        # A response too large to coexist with anything else must not be
        # inserted at all: inserting it would evict every other entry and
        # then itself, wiping the dedup cache for every concurrent
        # connection on this container.
        cache = modal_app._WsResponseCache(max_entries=8, max_bytes=100, ttl_seconds=60)
        cache.put("keep", b"ok")

        cache.put("huge", b"x" * 500)

        assert cache.get("huge") is None
        assert cache.get("keep") == b"ok"
        assert cache._total_bytes == 2


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

    def test_undecodable_inputs_are_reported_as_not_run_and_stay_retryable(
        self, modal_app, monkeypatch
    ) -> None:
        # Nothing executed, so the error must say so — and the request must
        # not be marked executed, or a legitimate retry would be refused.
        from fastapi.testclient import TestClient

        calls = []

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            return {"success": True, "result": {}}

        executor, app = _ws_app(modal_app, run_user_code)

        def boom(_inputs):
            raise ValueError("bad input payload")

        monkeypatch.setattr(
            (
                modal_app.Executor._get_user_cls()
                if hasattr(modal_app.Executor, "_get_user_cls")
                else modal_app.Executor
            ),
            "_deserialize_msgpack_inputs",
            staticmethod(boom),
        )
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert calls == []
        assert resp["success"] is False
        assert "was not run" in resp["error"]
        assert "req-1" not in executor._ws_executed

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


class TestBaseExceptionFromUserCode:
    def test_system_exit_returns_an_error_frame_instead_of_killing_the_conn(
        self, modal_app
    ) -> None:
        # Untrusted user code can call sys.exit(). concurrent.futures
        # re-raises SystemExit in the coroutine, so `except Exception` missed
        # it: the exception escaped the shielded task, tore the connection
        # down with nothing cached, and poisoned the request id for the whole
        # executed-marker TTL.
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            raise SystemExit(2)

        executor, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)
            # The connection is still usable afterwards.
            ws.send_bytes(frame)
            resent = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is False
        assert resp["error_type"] == "SystemExit"
        assert resp["server_error"] is True
        # The payload was cached, so the resend is answered, not re-run.
        assert resent == resp

    def test_a_request_that_never_ran_stays_resendable(self, modal_app) -> None:
        # If the worker thread never entered the user code, the executed
        # marker must be rolled back — otherwise every resend of that id
        # gets ResponseNoLongerAvailable for the whole TTL.
        registry = modal_app._TtlKeySet(ttl_seconds=60, max_entries=8)
        registry.add("req-1")
        assert "req-1" in registry

        registry.discard("req-1")

        assert "req-1" not in registry


class TestMalformedFrames:
    def test_non_map_execution_frame_is_reported_not_fatal(self, modal_app) -> None:
        # request.get(...) on a list used to raise AttributeError outside
        # every handler and kill the connection.
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            raise AssertionError("must not run")

        _, app = _ws_app(modal_app, run_user_code)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(msgpack.packb([1, 2, 3], use_bin_type=True))
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is False
        assert resp["error_type"] == "InvalidRequest"

    @pytest.mark.parametrize("chunk_count", [0, -1, 10**9, "many"])
    def test_bogus_chunk_header_closes_with_a_server_error(
        self, modal_app, chunk_count
    ) -> None:
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            raise AssertionError("must not run")

        _, app = _ws_app(modal_app, run_user_code)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(msgpack.packb({"_chunked": chunk_count}, use_bin_type=True))
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is False
        assert "chunk count" in resp["error"]


class TestWireBoundaryTypes:
    def test_non_string_request_id_is_ignored_rather_than_half_registered(
        self, modal_app
    ) -> None:
        # _TtlKeySet.add() used to store any hashable while __contains__
        # gated on isinstance(str), so an int request_id was recorded as
        # executed but never found again — the at-most-once backstop
        # silently did not hold.
        from fastapi.testclient import TestClient

        calls = []

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            return {"success": True, "result": {}}

        executor, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": 12345, "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is True
        # Treated as "no request id at all": nothing registered anywhere.
        assert 12345 not in executor._ws_executed
        assert len(executor._ws_executed._seen) == 0
        assert executor._ws_inflight == {}

    def test_registries_agree_on_what_a_key_is(self, modal_app) -> None:
        registry = modal_app._TtlKeySet(ttl_seconds=60, max_entries=8)

        for bogus in (12345, None, "", (1, 2)):
            registry.add(bogus)
            assert bogus not in registry
            assert registry.refresh(bogus) is False
        assert len(registry._seen) == 0

    def test_unhashable_session_id_does_not_kill_the_connection(
        self, modal_app
    ) -> None:
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            return {"success": True, "result": {}}

        _, app = _ws_app(modal_app, run_user_code)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(
                msgpack.packb(
                    {"_kind": "hello", "session_id": {"a": 1}}, use_bin_type=True
                )
            )
            reply = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert reply["_kind"] == "hello"
        assert reply["session_known"] is False


class TestInflightDedup:
    def test_concurrent_resend_awaits_the_running_execution(self, modal_app) -> None:
        # The one genuinely new concurrency guarantee: a second connection
        # resending the same request_id WHILE the first execution is still
        # running must await the in-flight task, not start a second one.
        import asyncio
        import threading

        from fastapi.testclient import TestClient

        release = threading.Event()
        calls = []

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            release.wait(timeout=5)
            return {"success": True, "result": {"n": len(calls)}}

        _, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)
        # One TestClient context => one portal => one event loop, matching a
        # real container where every connection shares the ASGI loop.
        client = TestClient(app)
        client.__enter__()

        results = {}

        def second_connection():
            with client.websocket_connect("/ws") as ws2:
                # Wait until the first execution is actually running.
                while not calls:
                    time.sleep(0.01)
                ws2.send_bytes(frame)
                release.set()
                results["second"] = msgpack.unpackb(ws2.receive_bytes(), raw=False)

        with client.websocket_connect("/ws") as ws1:
            ws1.send_bytes(frame)
            worker = threading.Thread(target=second_connection)
            worker.start()
            results["first"] = msgpack.unpackb(ws1.receive_bytes(), raw=False)
            worker.join(timeout=10)
        client.__exit__(None, None, None)

        assert len(calls) == 1, "the resend must not start a second execution"
        assert results["first"] == results["second"]


class TestContainerIdentity:
    def test_two_containers_get_distinct_ids(self, modal_app, monkeypatch) -> None:
        # The client trusts container_id to decide whether a resend may be
        # answered from this container's dedup registry. Two containers
        # sharing an id (e.g. if identify() were ever marked snap=True and
        # ran pre-snapshot) would let a resend re-execute user code silently.
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        cls = modal_app.Executor
        user_cls = cls._get_user_cls() if hasattr(cls, "_get_user_cls") else cls

        ids = set()
        for _ in range(2):
            executor = user_cls.__new__(user_cls)
            executor.workspace_id = "test-ws"
            user_cls.identify(executor)
            ids.add(executor._container_id)

        assert len(ids) == 2
