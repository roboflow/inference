"""Client <-> server wire contract for the webexec websocket protocol.

The client ships inside the ``inference`` package and the server is deployed
separately with ``modal deploy``, so the two halves ride independent release
trains — exactly the situation where a silently drifted constant or wire key
bites. Every value that crosses the boundary is pinned here against BOTH
implementations, loaded as real modules.

The repo already established this pattern in
``test_modal_code_hash.py::test_client_and_server_code_hashes_stay_in_sync``.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgpack
import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    modal_executor as client,
)

from .conftest import build_ws_app as _ws_app


class TestSharedConstants:
    def test_max_frame_bytes_match(self, modal_app) -> None:
        # Both sides chunk independently; a mismatch means one side splits at
        # a size the other never expects.
        assert client._WS_MAX_FRAME_BYTES == modal_app.WEBEXEC_WS_MAX_FRAME_BYTES

    def test_max_chunk_count_matches(self, modal_app) -> None:
        assert client._WS_MAX_CHUNKS == modal_app.WEBEXEC_WS_MAX_CHUNKS

    def test_connection_cap_stays_under_the_modal_input_timeout(
        self, modal_app
    ) -> None:
        # A websocket connection is one Modal input. If the cap is not below
        # the input timeout, Modal kills the connection instead of the server
        # closing it cleanly: no close frame, and in-flight executions are
        # cancelled mid-run.
        assert (
            modal_app.WEBEXEC_WS_MAX_CONNECTION_SECONDS
            < modal_app._executor_decorator_kwargs["timeout"]
        )

    def test_connection_cap_is_advertised_to_containers(self, modal_app) -> None:
        assert modal_app._executor_decorator_kwargs["env"][
            "WEBEXEC_WS_MAX_CONNECTION_SECONDS"
        ] == str(modal_app.WEBEXEC_WS_MAX_CONNECTION_SECONDS)


class TestHandshakeContract:
    def test_client_handshake_against_the_real_server_reply(self, modal_app) -> None:
        from fastapi.testclient import TestClient

        _, app = _ws_app(
            modal_app, lambda self, *a, **kw: {"success": True, "result": {}}
        )

        executor = client.WebSocketModalExecutor(workspace_id="test-ws")

        with TestClient(app).websocket_connect("/ws") as ws:
            # Drive the REAL client handshake over the REAL server route.
            executor._ws = _BridgeWS(ws)
            executor._handshake()

        assert executor._server.proto == 2
        # The one constant that cannot drift: learned, not duplicated.
        assert executor._server.idle_timeout == float(
            modal_app.WEBEXEC_WS_IDLE_TIMEOUT_SECONDS
        )
        assert executor._server.container_id is not None

    def test_reconnect_to_the_same_container_reports_session_known(
        self, modal_app
    ) -> None:
        from fastapi.testclient import TestClient

        _, app = _ws_app(
            modal_app, lambda self, *a, **kw: {"success": True, "result": {}}
        )
        executor = client.WebSocketModalExecutor(workspace_id="test-ws")
        test_client = TestClient(app)

        with test_client.websocket_connect("/ws") as ws:
            executor._ws = _BridgeWS(ws)
            executor._handshake()
            first_container = executor._server.container_id
            ws.send_bytes(
                msgpack.packb(
                    {"request_id": "r1", "inputs": {}, "code_hash": "h"},
                    use_bin_type=True,
                )
            )
            assert msgpack.unpackb(ws.receive_bytes(), raw=False)["success"] is True

        executor._had_success = True
        with test_client.websocket_connect("/ws") as ws:
            executor._ws = _BridgeWS(ws)
            # Same container object -> the session is known, no loud failure.
            executor._handshake()

        assert executor._server.container_id == first_container


class TestRequestFrameContract:
    def test_every_key_the_client_sends_is_a_key_the_server_reads(
        self, modal_app
    ) -> None:
        from fastapi.testclient import TestClient

        seen = {}

        def run_user_code(
            self,
            code_str,
            imports,
            run_function_name,
            inputs,
            client_code_hash,
            workflow_context,
        ):
            seen.update(
                code_str=code_str,
                imports=imports,
                run_function_name=run_function_name,
                inputs=inputs,
                code_hash=client_code_hash,
                workflow_context=workflow_context,
            )
            return {"success": True, "result": {}}

        _, app = _ws_app(modal_app, run_user_code)

        frame = client.WebSocketModalExecutor._build_ws_frame(
            python_code=SimpleNamespace(
                run_function_code="def run(x):\n    return x\n",
                run_function_name="run",
                imports=["import os"],
            ),
            packed_inputs={"x": 1},
            code_hash="hash-1",
            send_full_code=True,
            msgpack=msgpack,
            workflow_context={"workflow_id": "wf-1"},
            request_id="req-1",
        )

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is True
        assert resp["request_id"] == "req-1"
        assert seen == {
            "code_str": "def run(x):\n    return x\n",
            "imports": ["import os"],
            "run_function_name": "run",
            "inputs": {"x": 1},
            "code_hash": "hash-1",
            "workflow_context": {"workflow_id": "wf-1"},
        }


class TestChunkingContract:
    def test_server_reassembles_a_client_split_frame(self, modal_app) -> None:
        from fastapi.testclient import TestClient

        seen = {}

        def run_user_code(self, code_str, imports, name, inputs, code_hash, ctx):
            seen["inputs"] = inputs
            return {"success": True, "result": {}}

        _, app = _ws_app(modal_app, run_user_code)
        big = b"x" * (client._WS_MAX_FRAME_BYTES * 2 + 17)
        frame = msgpack.packb(
            {"request_id": "req-1", "inputs": {"blob": big}}, use_bin_type=True
        )
        frames = client._split_ws_frames(frame, msgpack)
        assert len(frames) == 4  # control frame + 3 chunks

        with TestClient(app).websocket_connect("/ws") as ws:
            for part in frames:
                ws.send_bytes(part)
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is True
        assert seen["inputs"] == {"blob": big}

    def test_client_reassembles_a_server_split_payload(self, modal_app) -> None:
        from fastapi.testclient import TestClient

        big = b"y" * (modal_app.WEBEXEC_WS_MAX_FRAME_BYTES * 2 + 5)

        def run_user_code(self, *args, **kwargs):
            return {"success": True, "result": {"blob": big}}

        _, app = _ws_app(modal_app, run_user_code)
        executor = client.WebSocketModalExecutor(workspace_id="test-ws")

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(
                msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)
            )
            executor._ws = _BridgeWS(ws)
            payload = executor._recv_reassembled(msgpack)

        assert msgpack.unpackb(payload, raw=False)["result"]["blob"] == big

    def test_a_cached_response_is_re_chunked_on_resend(self, modal_app) -> None:
        from fastapi.testclient import TestClient

        big = b"z" * (modal_app.WEBEXEC_WS_MAX_FRAME_BYTES * 2 + 5)
        calls = []

        def run_user_code(self, *args, **kwargs):
            calls.append(1)
            return {"success": True, "result": {"blob": big}}

        _, app = _ws_app(modal_app, run_user_code)
        executor = client.WebSocketModalExecutor(workspace_id="test-ws")
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            executor._ws = _BridgeWS(ws)
            ws.send_bytes(frame)
            first = executor._recv_reassembled(msgpack)
            ws.send_bytes(frame)
            second = executor._recv_reassembled(msgpack)

        assert len(calls) == 1
        assert first == second


class TestErrorTypeContract:
    def test_unknown_code_hash_string_matches(self, modal_app) -> None:
        from fastapi.testclient import TestClient

        def run_user_code(self, code_str, imports, name, inputs, code_hash, ctx):
            # Mirrors the real hash-only path: no code_str and no cached
            # namespace for this hash.
            return {
                "success": False,
                "error": "unknown hash",
                "error_type": "UnknownCodeHash",
            }

        _, app = _ws_app(modal_app, run_user_code)
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(
                msgpack.packb(
                    {"request_id": "req-1", "inputs": {}, "code_hash": "h"},
                    use_bin_type=True,
                )
            )
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        # The literal the client keys its resend-with-full-code retry on.
        assert resp["error_type"] == "UnknownCodeHash"
        assert "UnknownCodeHash" in Path(client.__file__).read_text()

    def test_response_no_longer_available_is_recognised_by_the_client(
        self, modal_app
    ) -> None:
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            return {"success": True, "result": {}}

        executor_obj, app = _ws_app(modal_app, run_user_code)
        frame = msgpack.packb({"request_id": "req-1", "inputs": {}}, use_bin_type=True)

        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            msgpack.unpackb(ws.receive_bytes(), raw=False)
            executor_obj._ws_response_cache = modal_app._WsResponseCache()
            ws.send_bytes(frame)
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        # The client must classify this as a TRANSPORT failure, not as the
        # user's block raising.
        from inference.core.workflows.errors import (
            DynamicBlockCodeError,
            DynamicBlockError,
        )

        with pytest.raises(DynamicBlockError) as excinfo:
            client.WebSocketModalExecutor._raise_server_error_if_infrastructure(
                resp, "MyBlock"
            )
        assert not isinstance(excinfo.value, DynamicBlockCodeError)

    def test_server_stamps_infrastructure_errors(self, modal_app) -> None:
        from fastapi.testclient import TestClient

        def run_user_code(self, *args, **kwargs):
            raise AssertionError("must not run")

        _, app = _ws_app(modal_app, run_user_code)
        with TestClient(app).websocket_connect("/ws") as ws:
            # Undecodable inputs: the block is never run.
            ws.send_bytes(
                msgpack.packb(
                    {"request_id": "req-1", "inputs": {"i": {"type": "nope"}}},
                    use_bin_type=True,
                )
            )
            resp = msgpack.unpackb(ws.receive_bytes(), raw=False)

        assert resp["success"] is False
        assert resp.get("server_error") is True


class _BridgeWS:
    """Adapts a starlette TestClient websocket to websocket-client's surface."""

    def __init__(self, ws: Any):
        self._ws = ws

    def send_binary(self, frame: bytes) -> None:
        self._ws.send_bytes(frame)

    def recv(self) -> Any:
        message = self._ws.receive()
        if "bytes" in message and message["bytes"] is not None:
            return message["bytes"]
        return message.get("text", "")

    def settimeout(self, value: float) -> None:
        pass

    def close(self) -> None:
        pass

    def ping(self) -> None:
        pass
