"""Protocol v2 behavior of WebSocketModalExecutor.

Covers the failure modes behind the idle-timeout incident: text frames on
dead connections, safe resend with request ids, and loud failure on lost
custom-Python sessions.
"""

from typing import Any, List, Optional

import msgpack
import pytest

from inference.core.env import WEBEXEC_WS_READ_TIMEOUT_SECONDS
from inference.core.workflows.errors import DynamicBlockError
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    WebexecSessionLostError,
    WebSocketModalExecutor,
)


def _pack(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


class _FakeWS:
    """Scripted socket: each recv pops the next canned reply.

    A reply that is an Exception instance is raised instead of returned.
    """

    def __init__(self, replies: Optional[List[Any]] = None):
        self.replies = list(replies or [])
        self.sent: List[bytes] = []
        self.closed = False
        self.timeouts: List[float] = []

    def send_binary(self, frame: bytes) -> None:
        self.sent.append(frame)

    def recv(self) -> Any:
        if not self.replies:
            raise AssertionError("recv called with no scripted reply")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self) -> None:
        self.closed = True

    def ping(self) -> None:
        pass

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


def _executor_with_ws(ws: _FakeWS, proto: int = 2) -> WebSocketModalExecutor:
    executor = WebSocketModalExecutor(workspace_id="test-ws")
    executor._ws = ws
    executor._server_proto = proto
    executor._server_idle_timeout = 10.0 if proto == 2 else None
    return executor


def _hello_reply(
    session_known: bool = False,
    idle_timeout_s: int = 10,
    container_id: Optional[str] = "container-1",
) -> bytes:
    reply = {
        "_kind": "hello",
        "proto": 2,
        "idle_timeout_s": idle_timeout_s,
        "session_known": session_known,
    }
    if container_id is not None:
        reply["container_id"] = container_id
    return _pack(reply)


class TestHandshake:
    def test_v2_server_sets_proto_and_idle_timeout(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS([_hello_reply(idle_timeout_s=12)])

        executor._handshake()

        assert executor._server_proto == 2
        assert executor._server_idle_timeout == 12.0
        sent = msgpack.unpackb(executor._ws.sent[0], raw=False)
        assert sent["_kind"] == "hello"
        assert sent["session_id"] == executor._session_id

    def test_v1_server_reply_falls_back_to_legacy(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        # A v1 server executes the hello as an empty request and answers
        # with a plain response dict.
        executor._ws = _FakeWS([_pack({"success": False, "error": "no code"})])

        executor._handshake()

        assert executor._server_proto == 1
        assert executor._server_idle_timeout is None

    def test_v2_server_container_id_is_recorded(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS([_hello_reply(container_id="container-7")])

        executor._handshake()

        assert executor._server_container_id == "container-7"

    def test_session_lost_after_prior_success_raises(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = True
        executor._ws = _FakeWS([_hello_reply(session_known=False)])

        with pytest.raises(WebexecSessionLostError):
            executor._handshake()

    def test_session_lost_rotates_session_so_it_fails_only_once(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = True
        executor._hashes_sent_on_ws = {"hash-1"}
        old_session = executor._session_id
        executor._ws = _FakeWS([_hello_reply(session_known=False)])

        with pytest.raises(WebexecSessionLostError):
            executor._handshake()

        # The failure is loud exactly once: the executor starts an honest
        # fresh session instead of finding the old id registered and
        # silently continuing with reset runtime state.
        assert executor._session_id != old_session
        assert executor._had_success is False
        assert executor._hashes_sent_on_ws == set()

        executor._ws = _FakeWS([_hello_reply(session_known=False)])
        executor._handshake()
        assert executor._server_proto == 2

    def test_unknown_session_without_prior_state_is_fine(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = False
        executor._ws = _FakeWS([_hello_reply(session_known=False)])

        executor._handshake()

        assert executor._server_proto == 2

    def test_text_frame_during_handshake_is_connection_error(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS(["connection closed"])

        with pytest.raises(ConnectionError):
            executor._handshake()


class TestRecvTyping:
    def test_text_frame_raises_instead_of_reaching_msgpack(self) -> None:
        executor = _executor_with_ws(_FakeWS(["server says bye"]))

        with pytest.raises(ConnectionError, match="non-binary"):
            executor._recv_reassembled(msgpack)

    def test_text_chunk_raises_instead_of_reaching_msgpack(self) -> None:
        executor = _executor_with_ws(
            _FakeWS([_pack({"_chunked": 2}), b"chunk-1", "bye"])
        )

        with pytest.raises(ConnectionError, match="non-binary"):
            executor._recv_reassembled(msgpack)


class TestSendRecvRetry:
    def test_v2_resends_after_recv_failure(self) -> None:
        first_ws = _FakeWS([ConnectionError("idle close")])
        second_ws = _FakeWS([_pack({"success": True, "request_id": "req-1"})])
        executor = _executor_with_ws(first_ws)
        sockets = [second_ws]

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)

        executor._ensure_connection = fake_ensure

        resp = executor._send_recv_with_retry(
            _pack({"inputs": {}}), "test-ws", request_id="req-1"
        )

        assert msgpack.unpackb(resp, raw=False)["success"] is True
        assert first_ws.closed
        # The frame was resent on the fresh socket.
        assert len(second_ws.sent) == 1

    def test_v2_resends_after_recv_failure_on_same_container(self) -> None:
        first_ws = _FakeWS([ConnectionError("idle close")])
        second_ws = _FakeWS([_pack({"success": True, "request_id": "req-1"})])
        executor = _executor_with_ws(first_ws)
        executor._server_container_id = "container-1"
        sockets = [second_ws]

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)
                executor._server_container_id = "container-1"

        executor._ensure_connection = fake_ensure

        resp = executor._send_recv_with_retry(
            _pack({"inputs": {}}), "test-ws", request_id="req-1"
        )

        assert msgpack.unpackb(resp, raw=False)["success"] is True
        assert len(second_ws.sent) == 1

    def test_v2_does_not_resend_when_reconnect_lands_on_other_container(
        self,
    ) -> None:
        # The dedup cache is per-container: a resend that reaches a
        # different container would run the user code a second time, so the
        # ambiguous outcome must fail loudly instead.
        first_ws = _FakeWS([ConnectionError("idle close")])
        second_ws = _FakeWS()
        executor = _executor_with_ws(first_ws)
        executor._server_container_id = "container-1"
        sockets = [second_ws]

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)
                executor._server_container_id = "container-2"

        executor._ensure_connection = fake_ensure

        with pytest.raises(DynamicBlockError, match="different container"):
            executor._send_recv_with_retry(
                _pack({"inputs": {}}), "test-ws", request_id="req-1"
            )

        # Nothing was sent on the new connection.
        assert second_ws.sent == []

    def test_v1_does_not_resend_after_recv_failure(self) -> None:
        ws = _FakeWS([ConnectionError("idle close")])
        executor = _executor_with_ws(ws, proto=1)
        executor._ensure_connection = lambda _workspace: None

        with pytest.raises(DynamicBlockError, match="not retried"):
            executor._send_recv_with_retry(_pack({"inputs": {}}), "test-ws")

    def test_session_lost_from_reconnect_propagates(self) -> None:
        executor = _executor_with_ws(_FakeWS([ConnectionError("idle close")]))

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                raise WebexecSessionLostError("state gone")

        executor._ensure_connection = fake_ensure

        with pytest.raises(WebexecSessionLostError):
            executor._send_recv_with_retry(
                _pack({"inputs": {}}), "test-ws", request_id="req-1"
            )


class TestResponseIdGuard:
    def test_mismatched_request_id_drops_connection(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        with pytest.raises(DynamicBlockError, match="stale"):
            executor._check_response_id({"request_id": "other"}, "req-1")

        assert executor._ws is None

    def test_matching_or_absent_id_passes(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        executor._check_response_id({"request_id": "req-1"}, "req-1")
        executor._check_response_id({"success": True}, "req-1")


class TestHeartbeat:
    def test_interval_derives_from_server_idle_timeout(self) -> None:
        executor = _executor_with_ws(_FakeWS())
        executor._server_idle_timeout = 9.0

        assert executor._heartbeat_interval() == 3.0

    def test_interval_falls_back_on_v1(self) -> None:
        executor = _executor_with_ws(_FakeWS(), proto=1)

        assert (
            executor._heartbeat_interval()
            == WebSocketModalExecutor._KEEPALIVE_IDLE_SECONDS
        )

    def test_v2_heartbeat_is_application_frame_with_ack(self) -> None:
        ws = _FakeWS([_pack({"_kind": "heartbeat_ack"})])
        executor = _executor_with_ws(ws)

        executor._send_heartbeat(ws)

        sent = msgpack.unpackb(ws.sent[0], raw=False)
        assert sent == {"_kind": "heartbeat"}

    def test_v2_heartbeat_text_reply_raises(self) -> None:
        ws = _FakeWS(["closing"])
        executor = _executor_with_ws(ws)

        with pytest.raises(ConnectionError):
            executor._send_heartbeat(ws)

    def test_v2_heartbeat_uses_short_ack_timeout_and_restores_it(self) -> None:
        # The ack must not be awaited under the execution-sized read
        # timeout: a half-open connection would pin _io_lock for minutes.
        ws = _FakeWS([_pack({"_kind": "heartbeat_ack"})])
        executor = _executor_with_ws(ws)

        executor._send_heartbeat(ws)

        assert ws.timeouts[0] == WebSocketModalExecutor._HEARTBEAT_ACK_TIMEOUT_SECONDS
        assert ws.timeouts[-1] == WEBEXEC_WS_READ_TIMEOUT_SECONDS

    def test_v2_heartbeat_restores_read_timeout_on_failure(self) -> None:
        ws = _FakeWS([ConnectionError("half-open")])
        executor = _executor_with_ws(ws)

        with pytest.raises(ConnectionError):
            executor._send_heartbeat(ws)

        assert ws.timeouts[-1] == WEBEXEC_WS_READ_TIMEOUT_SECONDS
