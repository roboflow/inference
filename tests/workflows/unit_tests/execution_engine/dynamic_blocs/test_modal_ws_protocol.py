"""Protocol v2 behavior of WebSocketModalExecutor.

Covers the failure modes behind the idle-timeout incident: text frames on
dead connections, safe resend with request ids, and loud failure on lost
custom-Python sessions.
"""

import sys
import threading
import time as _time
from types import SimpleNamespace
from typing import Any, List, Optional

import msgpack
import pytest

from inference.core.env import (
    WEBEXEC_WS_IDLE_RELEASE_SECONDS,
    WEBEXEC_WS_READ_TIMEOUT_SECONDS,
)
from inference.core.workflows.errors import DynamicBlockCodeError, DynamicBlockError
from inference.core.workflows.execution_engine.v1.dynamic_blocks import modal_executor
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    WebexecSessionLostError,
    WebSocketModalExecutor,
    _ServerInfo,
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
    executor._server = _ServerInfo(
        proto=proto, idle_timeout=10.0 if proto == 2 else None
    )
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

        assert executor._server.proto == 2
        assert executor._server.idle_timeout == 12.0
        sent = msgpack.unpackb(executor._ws.sent[0], raw=False)
        assert sent["_kind"] == "hello"
        assert sent["session_id"] == executor._session_id

    def test_v1_server_reply_falls_back_to_legacy(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        # A v1 server executes the hello as an empty request and answers
        # with a plain response dict.
        executor._ws = _FakeWS([_pack({"success": False, "error": "no code"})])

        executor._handshake()

        assert executor._server.proto == 1
        assert executor._server.idle_timeout is None

    def test_v1_fallback_after_prior_success_does_not_fail_the_request(self) -> None:
        # A v1 server cannot confirm the session, but it also idle-closes
        # every 10s with no way for v1 keepalives to reset that timer, so
        # reconnects are its normal state. Failing each one after a prior
        # success would fail roughly every other request — worse than the
        # rare silent reset it would prevent. The guarantee needs v2.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = True
        session = executor._session_id
        executor._ws = _FakeWS([_pack({"success": False, "error": "no code"})])

        executor._handshake()

        assert executor._server.proto == 1
        assert executor._had_success is True
        assert executor._session_id == session

    def test_handshake_never_exposes_v2_proto_without_its_idle_timeout(self) -> None:
        # The keepalive reads server info unlocked, so it must never observe
        # proto 2 alongside a stale idle timeout: _heartbeat_interval would
        # fall back to 25s, which a v2 server closes under. Publishing one
        # immutable value makes the torn read unrepresentable — this test
        # samples the state after EVERY bytecode line of the handshake.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._server = _ServerInfo(proto=1, idle_timeout=None)
        executor._ws = _FakeWS([_hello_reply(idle_timeout_s=9)])
        seen = []

        handshake_code = WebSocketModalExecutor._handshake.__code__

        def trace(frame, event, _arg):
            if frame.f_code is not handshake_code:
                return None
            if event == "line":
                server = executor._server
                seen.append((server.proto, server.idle_timeout))
            return trace

        # Restore whatever tracer was active (coverage.py installs one), not
        # None: clearing it would silently disable coverage for every test
        # that runs after this one in the same worker process.
        previous_tracer = sys.gettrace()
        sys.settrace(trace)
        try:
            executor._handshake()
        finally:
            sys.settrace(previous_tracer)

        assert seen, "tracing did not observe the handshake"
        inconsistent = [pair for pair in seen if pair[0] == 2 and pair[1] is None]
        assert not inconsistent, f"torn server state observable: {inconsistent}"
        assert executor._heartbeat_interval() == 3.0

    def test_v2_server_container_id_is_recorded(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS([_hello_reply(container_id="container-7")])

        executor._handshake()

        assert executor._server.container_id == "container-7"

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
        assert executor._server.proto == 2

    def test_unknown_session_without_prior_state_is_fine(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = False
        executor._ws = _FakeWS([_hello_reply(session_known=False)])

        executor._handshake()

        assert executor._server.proto == 2

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
    def test_v2_does_not_resend_when_container_is_unidentified(self) -> None:
        # A server that does not identify its container (pre-container_id
        # build) cannot promise the reconnect reaches the same dedup
        # registry, so the guard must fail closed rather than assume it did.
        first_ws = _FakeWS([ConnectionError("idle close")])
        second_ws = _FakeWS()
        executor = _executor_with_ws(first_ws)
        executor._server = executor._server._replace(container_id=None)
        sockets = [second_ws]

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)

        executor._ensure_connection = fake_ensure

        with pytest.raises(DynamicBlockError, match="same container"):
            executor._send_recv_with_retry(
                _pack({"inputs": {}}), "test-ws", request_id="req-1"
            )

        assert first_ws.closed
        assert second_ws.sent == []

    def test_v2_resends_after_recv_failure_on_same_container(self) -> None:
        first_ws = _FakeWS([ConnectionError("idle close")])
        second_ws = _FakeWS([_pack({"success": True, "request_id": "req-1"})])
        executor = _executor_with_ws(first_ws)
        executor._server = executor._server._replace(container_id="container-1")
        sockets = [second_ws]

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)
                executor._server = executor._server._replace(container_id="container-1")

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
        executor._server = executor._server._replace(container_id="container-1")
        sockets = [second_ws]

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)
                executor._server = executor._server._replace(container_id="container-2")

        executor._ensure_connection = fake_ensure

        with pytest.raises(DynamicBlockError, match="same container"):
            executor._send_recv_with_retry(
                _pack({"inputs": {}}), "test-ws", request_id="req-1"
            )

        # Nothing was sent on the new connection.
        assert second_ws.sent == []

    def test_container_id_is_cleared_when_connection_is_dropped(self) -> None:
        # A stale id left behind would let the resend guard match a
        # container this executor is no longer talking to.
        executor = _executor_with_ws(_FakeWS())
        executor._server = executor._server._replace(container_id="container-1")

        executor._drop_ws_connection()

        assert executor._server.container_id is None

    def test_v1_does_not_resend_after_recv_failure(self) -> None:
        ws = _FakeWS([ConnectionError("idle close")])
        executor = _executor_with_ws(ws, proto=1)
        executor._ensure_connection = lambda _workspace: None

        with pytest.raises(DynamicBlockError, match="not retried"):
            executor._send_recv_with_retry(_pack({"inputs": {}}), "test-ws")

    def test_session_lost_on_resend_reports_ambiguous_outcome(self) -> None:
        # The frame was already accepted somewhere. Telling the user to
        # replay from a checkpoint would re-run side effects that may
        # already have happened, so the ambiguous-outcome error wins over
        # the session-lost one.
        executor = _executor_with_ws(_FakeWS([ConnectionError("idle close")]))
        executor._server = executor._server._replace(container_id="container-1")

        def fake_ensure(_workspace: str) -> None:
            if executor._ws is None:
                raise WebexecSessionLostError("state gone")

        executor._ensure_connection = fake_ensure

        with pytest.raises(DynamicBlockError, match="may have already executed"):
            executor._send_recv_with_retry(
                _pack({"inputs": {}}), "test-ws", request_id="req-1"
            )

    def test_session_lost_before_send_still_propagates(self) -> None:
        # Nothing was sent yet, so there is no ambiguity to report: the job
        # genuinely has to be replayed from its checkpoint.
        executor = _executor_with_ws(_FakeWS())
        executor._ws = None

        def fake_ensure(_workspace: str) -> None:
            raise WebexecSessionLostError("state gone")

        executor._ensure_connection = fake_ensure

        with pytest.raises(WebexecSessionLostError):
            executor._send_recv_with_retry(
                _pack({"inputs": {}}), "test-ws", request_id="req-1"
            )


def _single_pass_stop_event() -> Any:
    """Stop event letting the keepalive body run exactly once."""
    passes = {"n": 0}

    def wait(_timeout: float) -> bool:
        passes["n"] += 1
        return passes["n"] > 1

    return SimpleNamespace(wait=wait, is_set=lambda: passes["n"] > 1)


class _LockAcquiredHook:
    """Lock that runs a callback once acquired — models a thread winning the
    lock only after another has already done its work."""

    def __init__(self, on_acquire) -> None:
        self._lock = threading.Lock()
        self._on_acquire = on_acquire

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._on_acquire()
        return acquired

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_LockAcquiredHook":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


class TestKeepaliveIdleRelease:
    def test_releases_connection_and_rotates_session_when_idle(self) -> None:
        ws = _FakeWS()
        executor = _executor_with_ws(ws)
        executor._had_success = True
        old_session = executor._session_id
        executor._last_activity = _time.monotonic() - (
            WEBEXEC_WS_IDLE_RELEASE_SECONDS + 5
        )

        executor._keepalive_loop(_single_pass_stop_event())

        assert executor._ws is None
        assert ws.closed
        assert executor._session_id != old_session
        assert executor._had_success is False

    def test_activity_just_before_the_lock_cancels_the_release(self) -> None:
        # The idle value read before acquiring the lock can be stale: a frame
        # may complete in between. Releasing on it would discard a live
        # session silently, since rotation clears the fail-loudly latch.
        ws = _FakeWS()
        executor = _executor_with_ws(ws)
        executor._had_success = True
        old_session = executor._session_id
        executor._last_activity = _time.monotonic() - (
            WEBEXEC_WS_IDLE_RELEASE_SECONDS + 5
        )
        executor._io_lock = _LockAcquiredHook(
            lambda: setattr(executor, "_last_activity", _time.monotonic())
        )

        executor._keepalive_loop(_single_pass_stop_event())

        assert executor._ws is ws
        assert not ws.closed
        assert executor._session_id == old_session
        assert executor._had_success is True

    def test_keepalive_backs_off_while_a_request_holds_the_lock(self) -> None:
        # The whole design rests on the request path and the heartbeat never
        # touching the socket at the same time: they share one connection and
        # both read frames, so an interleave would hand a response to the
        # keepalive (or an ack to the request). The keepalive must therefore
        # take _io_lock non-blockingly and give up when a real frame holds it.
        ws = _FakeWS()
        executor = _executor_with_ws(ws)
        executor._had_success = True
        old_session = executor._session_id
        executor._last_activity = _time.monotonic() - (
            WEBEXEC_WS_IDLE_RELEASE_SECONDS + 5
        )

        # Held for the whole pass by "another thread", never released.
        assert executor._io_lock.acquire(blocking=False)
        try:
            executor._keepalive_loop(_single_pass_stop_event())
        finally:
            executor._io_lock.release()

        # Nothing was written, nothing was read, and the idle release did not
        # fire — the in-flight frame owns the socket.
        assert ws.sent == []
        assert not ws.closed
        assert executor._ws is ws
        assert executor._session_id == old_session
        assert executor._had_success is True


class TestGracefulServerClose:
    """A close the server announced is proof the frame was never read.

    The server decides to close at the TOP of its receive loop and never
    reads again, so a frame written concurrently is unprocessed. Reporting
    that as "may have already executed" fails work that never ran — and at
    the default connection cap it happens on a schedule.
    """

    def test_closing_frame_in_the_execution_slot_is_not_a_response(self) -> None:
        executor = _executor_with_ws(_FakeWS([_pack({"_kind": "closing"})]))

        with pytest.raises(modal_executor._ServerClosingError):
            executor._recv_reassembled(msgpack)

    def test_closing_frame_during_a_heartbeat_is_not_an_ack(self) -> None:
        ws = _FakeWS([_pack({"_kind": "closing"})])
        executor = _executor_with_ws(ws)

        with pytest.raises(modal_executor._ServerClosingError):
            executor._send_heartbeat(ws)

    def test_graceful_close_retries_on_any_container(self, monkeypatch: Any) -> None:
        # Attempt 1: the frame goes out, then the server's announced close
        # arrives instead of a response. Attempt 2 lands on a DIFFERENT
        # container. Without proof of non-delivery the same-container guard
        # would refuse the resend and report "may have already executed";
        # with it, the frame is simply retried and succeeds.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        containers = iter(["container-1", "container-2"])
        sockets = [
            _FakeWS([_pack({"_kind": "closing"})]),
            _FakeWS([_pack({"success": True, "request_id": "req-1", "result": {}})]),
        ]

        def fake_ensure_connection(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)
                executor._server = _ServerInfo(
                    proto=2, idle_timeout=10.0, container_id=next(containers)
                )

        monkeypatch.setattr(executor, "_ensure_connection", fake_ensure_connection)

        resp = executor._send_recv_with_retry(
            _pack({"request_id": "req-1"}), "test-ws", request_id="req-1"
        )

        assert msgpack.unpackb(resp, raw=False)["success"] is True

    def test_graceful_close_does_not_disarm_an_earlier_delivery(
        self, monkeypatch: Any
    ) -> None:
        # A closing frame proves only that THIS attempt's write went unread.
        # Attempt 1 hands the frame to container-1, which may have executed it
        # and lost only the response. Attempt 2 reaches container-1 again and
        # is closed before being read. If that close cleared the delivery
        # state, attempt 3 would re-send to container-2 -- which has no dedup
        # record of the request -- and run the user's block a SECOND time.
        # The ambiguity, once incurred, has to survive.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        containers = iter(["container-1", "container-1", "container-2"])
        sockets = [
            # Attempt 1: frame delivered, response lost.
            _FakeWS([ConnectionError("response lost after send")]),
            # Attempt 2: same container, server closes before reading.
            _FakeWS([_pack({"_kind": "closing"})]),
            # Attempt 3 must never get here on a different container.
            _FakeWS([_pack({"success": True, "request_id": "req-1", "result": {}})]),
        ]

        def fake_ensure_connection(_workspace: str) -> None:
            if executor._ws is None:
                executor._ws = sockets.pop(0)
                executor._server = _ServerInfo(
                    proto=2, idle_timeout=10.0, container_id=next(containers)
                )

        monkeypatch.setattr(executor, "_ensure_connection", fake_ensure_connection)

        third_socket = sockets[2]

        with pytest.raises(DynamicBlockError, match="may have already executed"):
            executor._send_recv_with_retry(
                _pack({"request_id": "req-1"}), "test-ws", request_id="req-1"
            )

        # The third connection gets established (the guard is checked after
        # reconnecting), but NOTHING is written to it: the frame is never
        # handed to a container that could not answer it from a dedup record.
        assert third_socket.sent == []


class TestResponseIdGuard:
    def test_mismatched_request_id_drops_connection(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        with pytest.raises(DynamicBlockError, match="stale"):
            executor._check_response_id({"request_id": "other"}, "req-1")

        assert executor._ws is None

    def test_matching_id_passes(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        executor._check_response_id({"request_id": "req-1"}, "req-1")

    def test_v2_requires_the_echoed_id(self) -> None:
        # A late heartbeat_ack landing in the execution recv slot has no
        # request_id. Accepting it would surface as a fabricated
        # "RuntimeError: Unknown error" against the user's block and leave
        # the real response queued, desyncing the next request too.
        executor = _executor_with_ws(_FakeWS())

        with pytest.raises(DynamicBlockError, match="in-flight request id"):
            executor._check_response_id({"success": True}, "req-1")

        assert executor._ws is None

    def test_v2_rejects_a_control_frame(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        with pytest.raises(DynamicBlockError, match="in-flight request id"):
            executor._check_response_id(
                {"_kind": "heartbeat_ack", "request_id": "req-1"}, "req-1"
            )

    def test_v1_still_accepts_a_response_without_an_id(self) -> None:
        executor = _executor_with_ws(_FakeWS(), proto=1)

        executor._check_response_id({"success": True}, "req-1")

        assert executor._ws is not None


class TestHeartbeat:
    def test_interval_derives_from_server_idle_timeout(self) -> None:
        executor = _executor_with_ws(_FakeWS())
        executor._server = executor._server._replace(idle_timeout=9.0)

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


class TestSessionLossKillSwitch:
    def test_enforcement_can_be_disabled(self, monkeypatch: Any) -> None:
        # Prod needs to revert to the pre-v2 (silent continuation) behavior
        # without rolling the inference ref back across every consumer.
        monkeypatch.setattr(modal_executor, "WEBEXEC_WS_FAIL_ON_SESSION_LOSS", False)
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = True
        session = executor._session_id
        executor._ws = _FakeWS([_hello_reply(session_known=False)])

        executor._handshake()

        assert executor._server.proto == 2
        # The session is still rotated: state really is gone, so the next
        # reconnect must not silently pass the check on the old id.
        assert executor._session_id != session
        assert executor._had_success is False

    def test_enforcement_on_by_default(self) -> None:
        assert modal_executor.WEBEXEC_WS_FAIL_ON_SESSION_LOSS is True

    def test_session_lost_publishes_server_info_before_raising(self) -> None:
        # The keepalive thread reads _server unlocked; leaving it on the v1
        # default after a v2 handshake would restore the 25s interval
        # against a 10s-idle server.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._had_success = True
        executor._ws = _FakeWS([_hello_reply(session_known=False)])

        with pytest.raises(WebexecSessionLostError):
            executor._handshake()

        assert executor._server.proto == 2
        assert executor._server.idle_timeout == 10.0


def _run_one_execution(executor: WebSocketModalExecutor, response: dict) -> Any:
    """Drive _execute_ws once with a canned server response."""

    def _fake_send_recv(
        _frame: bytes, _workspace: str, request_id: Any = None
    ) -> bytes:
        # Echo the id the executor generated, as a real v2 server does.
        echoed = dict(response)
        if executor._server.proto == 2:
            echoed["request_id"] = request_id
        return _pack(echoed)

    executor._send_recv_with_retry = _fake_send_recv  # type: ignore
    return executor._execute_ws(
        "MyBlock",
        SimpleNamespace(
            run_function_code="def run(x):\n    return x\n",
            run_function_name="run",
            imports=[],
        ),
        {},
        "ws",
        msgpack,
        {},
    )


class TestHadSuccessLatch:
    def test_v1_success_does_not_arm_the_v2_session_check(self) -> None:
        # Client rolls out before the Modal app: the executor talks v1 and
        # succeeds. If that armed the latch, the first reconnect onto an
        # upgraded v2 container would hard-fail every workspace executor
        # that had ever succeeded.
        executor = _executor_with_ws(_FakeWS(), proto=1)

        _run_one_execution(executor, {"success": True, "result": {}})

        assert executor._had_success is False

    def test_v2_success_arms_it(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        _run_one_execution(executor, {"success": True, "result": {}})

        assert executor._had_success is True


class TestIdleReleaseDisable:
    def test_non_positive_value_disables_idle_release(self, monkeypatch: Any) -> None:
        # <= 0 disables, matching WEBEXEC_MODAL_EXECUTOR_IDLE_TTL_SECONDS.
        # Treating it as "always release" would drop the connection after one
        # heartbeat interval AND bypass the session check (rotation clears the
        # latch), silently resetting stateful blocks.
        monkeypatch.setattr(modal_executor, "WEBEXEC_WS_IDLE_RELEASE_SECONDS", 0)
        ws = _FakeWS([_pack({"_kind": "heartbeat_ack"})])
        executor = _executor_with_ws(ws)
        executor._had_success = True
        old_session = executor._session_id
        executor._last_activity = _time.monotonic() - 100_000

        executor._keepalive_loop(_single_pass_stop_event())

        assert executor._ws is ws
        assert not ws.closed
        assert executor._session_id == old_session
        assert executor._had_success is True


class TestIdleTimeoutCoercion:
    @pytest.mark.parametrize(
        "advertised, expected",
        [
            (None, 10.0),
            ("not-a-number", 10.0),
            (0, 10.0),
            (-5, 10.0),
            (float("nan"), 10.0),
            # Never clamped UP: believing the server allows more idle time
            # than it does is what reinstates the idle-close this protocol
            # fixes. The floor lives on the derived interval instead.
            (1, 1.0),
            (2, 2.0),
            (10_000, 300.0),  # clamped down: must stay under the idle release
            (30, 30.0),
        ],
    )
    def test_advisory_field_never_bricks_the_transport(
        self, advertised: Any, expected: float
    ) -> None:
        assert modal_executor._coerce_idle_timeout(advertised) == expected

    @pytest.mark.parametrize("advertised", [1.5, 2, 3, 10, 30, 300])
    def test_heartbeat_gap_stays_under_the_advertised_timeout(
        self, advertised: float
    ) -> None:
        # The keepalive skips a tick when ``idle < interval``, so the real
        # worst case between two app-level frames is 2 x interval. If that
        # ever reaches the server's deadline the connection dies on idle --
        # exactly the bug protocol v2 exists to fix.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._server = _ServerInfo(
            proto=2, idle_timeout=modal_executor._coerce_idle_timeout(advertised)
        )

        assert 2 * executor._heartbeat_interval() < advertised

    def test_handshake_survives_a_garbage_idle_timeout(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS([_pack({"_kind": "hello", "idle_timeout_s": "soon"})])

        executor._handshake()

        assert executor._server.proto == 2
        assert executor._server.idle_timeout == 10.0


class TestPostLoopRetryExhaustion:
    def test_delivered_frame_reports_possible_execution(self, monkeypatch: Any) -> None:
        # Attempt 1 delivers the frame and loses the response; attempts 2-3
        # reconnect to the same container and fail too. The final error must
        # keep the "may have already executed" wording, or a job runner
        # re-runs the block and duplicates its side effects.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._server = _ServerInfo(
            proto=2, idle_timeout=10.0, container_id="container-1"
        )

        class _Sock:
            def send_binary(self, frame: bytes) -> None:
                pass

            def recv(self) -> Any:
                raise ConnectionError("boom")

            def close(self) -> None:
                pass

        def _fake_ensure(_workspace: str) -> None:
            executor._ws = _Sock()
            executor._server = executor._server._replace(container_id="container-1")

        monkeypatch.setattr(executor, "_ensure_connection", _fake_ensure)

        with pytest.raises(DynamicBlockError) as excinfo:
            executor._send_recv_with_retry(_pack({"request_id": "r"}), "ws", "r")

        assert "may have already executed" in excinfo.value.public_message
        assert "websocket_response" in excinfo.value.context

    def test_never_delivered_frame_reports_a_connect_failure(
        self, monkeypatch: Any
    ) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")

        def _fake_ensure(_workspace: str) -> None:
            raise ConnectionError("no route")

        monkeypatch.setattr(executor, "_ensure_connection", _fake_ensure)

        with pytest.raises(DynamicBlockError) as excinfo:
            executor._send_recv_with_retry(_pack({"request_id": "r"}), "ws", "r")

        assert "failed after retry" in excinfo.value.public_message
        assert "websocket_connection" in excinfo.value.context


class TestServerInfrastructureErrors:
    @pytest.mark.parametrize(
        "result",
        [
            {"success": False, "error_type": "ResponseNoLongerAvailable", "error": "x"},
            {"success": False, "error_type": "InvalidRequest", "error": "x"},
            {
                "success": False,
                "error_type": "ValueError",
                "error": "x",
                "server_error": True,
            },
        ],
    )
    def test_transport_failures_are_not_reported_as_user_code_errors(
        self, result: dict
    ) -> None:
        # DynamicBlockCodeError means "the user's Python raised". These
        # failures mean the block either never ran or ran fine.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        with pytest.raises(DynamicBlockError) as excinfo:
            executor._raise_server_error_if_infrastructure(result, "MyBlock")

        assert not isinstance(excinfo.value, DynamicBlockCodeError)
        assert "websocket_server_error" in excinfo.value.context

    def test_user_code_errors_still_fall_through(self) -> None:
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._raise_server_error_if_infrastructure(
            {"success": False, "error_type": "ZeroDivisionError", "error": "x"},
            "MyBlock",
        )

    def test_v2_ignores_a_forged_infrastructure_error_type(self) -> None:
        # error_type is an exception CLASS NAME chosen by untrusted user code.
        # On v2 the server stamps server_error on its own responses, so a
        # block raising ``class InvalidRequest(Exception)`` must still be
        # reported as the user's code failing -- with its traceback intact.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._server = _ServerInfo(proto=2, idle_timeout=10.0)

        executor._raise_server_error_if_infrastructure(
            {"success": False, "error_type": "InvalidRequest", "error": "x"},
            "MyBlock",
        )

    def test_v1_still_trusts_the_error_type_names(self) -> None:
        # A v1 server cannot stamp the flag, so there the names must still
        # classify -- otherwise every legacy transport failure is reported as
        # the user's block raising.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._server = _ServerInfo(proto=1)

        with pytest.raises(DynamicBlockError):
            executor._raise_server_error_if_infrastructure(
                {"success": False, "error_type": "InvalidRequest", "error": "x"},
                "MyBlock",
            )


class TestChunkedResponseValidation:
    @pytest.mark.parametrize("chunk_count", [0, -1, 10**9, "many", True])
    def test_bogus_chunk_header_is_connection_death(self, chunk_count: Any) -> None:
        # A negative count yields an empty join and explodes in unpackb far
        # from any handler; a huge one stalls the read timeout per attempt.
        ws = _FakeWS([_pack({"_chunked": chunk_count})])
        executor = _executor_with_ws(ws)

        with pytest.raises(ConnectionError, match="chunk count"):
            executor._recv_reassembled(msgpack)


class TestMalformedResponse:
    def test_undecodable_response_becomes_a_transport_error(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        with pytest.raises(DynamicBlockError, match="could not be"):
            executor._unpack_response(b"\xc1not-msgpack", msgpack)

        assert executor._ws is None

    def test_non_map_response_becomes_a_transport_error(self) -> None:
        executor = _executor_with_ws(_FakeWS())

        with pytest.raises(DynamicBlockError, match="not a"):
            executor._unpack_response(_pack([1, 2, 3]), msgpack)

        assert executor._ws is None


class TestCloseTearsDownKeepaliveFirst:
    def test_close_stops_and_joins_the_keepalive_thread(self) -> None:
        # close() runs on a request thread when the executor cache evicts
        # this executor. websocket-client's close() does a recv_frame() that
        # bypasses the socket read lock, so the keepalive must be gone
        # before the fd is freed.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS()
        executor._server = _ServerInfo(
            proto=2, idle_timeout=10.0, container_id="container-1"
        )
        executor._ensure_keepalive_thread()
        thread = executor._keepalive_thread
        assert thread is not None and thread.is_alive()

        executor.close()

        assert not thread.is_alive()
        assert executor._ws is None
        assert executor._server.container_id is None

    def test_close_does_not_block_on_an_in_flight_execution(self) -> None:
        # _io_lock can be held for the whole read timeout by a running block.
        # Executor-cache eviction must not wait that long.
        executor = WebSocketModalExecutor(workspace_id="test-ws")
        executor._ws = _FakeWS()
        executor._CLOSE_LOCK_TIMEOUT_SECONDS = 0.05
        executor._io_lock.acquire()
        try:
            started = _time.monotonic()
            executor.close()
            elapsed = _time.monotonic() - started
        finally:
            executor._io_lock.release()

        assert elapsed < 1.0
        assert executor._ws is None
