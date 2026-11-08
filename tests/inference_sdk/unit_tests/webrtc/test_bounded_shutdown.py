"""Bounded shutdown through the real SDK event-loop and ownership paths."""

import asyncio
import inspect
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inference_sdk.webrtc import session as session_module
from inference_sdk.webrtc.session import WebRTCSession

WATCHDOG = 4


class _Transport:
    def __init__(self):
        self.closed = asyncio.Event()
        self.abort_calls = 0

    def abort(self):
        self.abort_calls += 1
        self.closed.set()


class _Peer:
    def __init__(self):
        self.transports = [_Transport(), _Transport()]
        self.started = threading.Event()
        self.close_calls = 0
        self.finished = False
        self.ice = [
            SimpleNamespace(
                _connection=SimpleNamespace(
                    _protocols=[SimpleNamespace(transport=transport)]
                )
            )
            for transport in self.transports
        ]
        self.sctp = SimpleNamespace(transport=SimpleNamespace(transport=self.ice[0]))

    def getTransceivers(self):  # noqa: N802
        # Include the SCTP ICE transport again to verify de-duplication.
        return [
            SimpleNamespace(
                receiver=SimpleNamespace(transport=SimpleNamespace(transport=ice))
            )
            for ice in self.ice
        ]

    async def close(self):
        self.close_calls += 1
        self.started.set()
        for transport in self.transports:
            await transport.closed.wait()
        self.finished = True


class _Source:
    def __init__(self):
        self.calls = 0
        self.release = None
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def cleanup(self):
        self.calls += 1
        self.started.set()
        if self.release is not None:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise


@pytest.fixture
def owned_session(monkeypatch):
    monkeypatch.setattr(session_module, "WEBRTC_CLEANUP_TIMEOUT", 0.03, raising=False)
    monkeypatch.setattr(session_module, "WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT", 0.05)
    peer = _Peer()
    source = _Source()
    session = WebRTCSession(
        api_url="http://127.0.0.1:9",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config={},
        stream_config=MagicMock(),
    )

    async def initialize():
        session._pc = peer

    monkeypatch.setattr(session, "_init", initialize)
    session._ensure_started()
    callers = []
    yield session, peer, source, callers

    # Parent failures must also release every gate and reap the real owner loop.
    if session._loop.is_running():
        for transport in peer.transports:
            session._loop.call_soon_threadsafe(transport.closed.set)
        if source.release is not None:
            session._loop.call_soon_threadsafe(source.release.set)
    for caller in callers:
        caller.join(timeout=WATCHDOG)
        assert not caller.is_alive()
    if session._loop.is_running():
        session._loop.call_soon_threadsafe(session._loop.stop)
    session._loop_thread.join(timeout=WATCHDOG)
    assert not session._loop_thread.is_alive()


def _close_in_thread(session, callers):
    finished = threading.Event()
    errors = []

    def close():
        try:
            session.close()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    caller = threading.Thread(target=close, daemon=True)
    callers.append(caller)
    caller.start()
    return finished, errors


def test_timeout_aborts_all_and_only_peer_owned_udp_transports(owned_session):
    session, peer, source, callers = owned_session
    unrelated = _Transport()
    finished, errors = _close_in_thread(session, callers)
    assert peer.started.wait(timeout=WATCHDOG)

    assert finished.wait(
        timeout=WATCHDOG
    ), "SDK close waited indefinitely for UDP close"
    assert errors == []
    assert peer.finished
    assert peer.close_calls == 1
    assert [transport.abort_calls for transport in peer.transports] == [1, 1]
    assert unrelated.abort_calls == 0
    assert source.calls == 1
    assert session._loop.is_closed()
    assert not session._loop_thread.is_alive()
    session.close()


def test_graceful_peer_close_preserves_transports(owned_session):
    session, peer, source, _ = owned_session
    for transport in peer.transports:
        session._loop.call_soon_threadsafe(transport.closed.set)

    session.close()

    assert peer.close_calls == 1
    assert peer.finished
    assert [transport.abort_calls for transport in peer.transports] == [0, 0]
    assert source.calls == 1
    assert session._loop.is_closed()


def test_callback_and_concurrent_callers_share_bounded_cleanup(owned_session):
    session, peer, source, callers = owned_session
    callback_returned = threading.Event()

    def callback():
        session.close()
        callback_returned.set()

    session._loop.call_soon_threadsafe(callback)
    assert callback_returned.wait(timeout=WATCHDOG)
    assert peer.started.wait(timeout=WATCHDOG)
    first_done, first_errors = _close_in_thread(session, callers)
    second_done, second_errors = _close_in_thread(session, callers)

    assert first_done.wait(timeout=WATCHDOG), "First duplicate close did not finish"
    assert second_done.wait(timeout=WATCHDOG), "Second duplicate close did not finish"
    assert first_errors == second_errors == []
    assert peer.close_calls == source.calls == 1
    assert [transport.abort_calls for transport in peer.transports] == [1, 1]
    assert session._loop.is_closed()


def test_source_cleanup_timeout_is_visible_and_cancels_owned_task(owned_session):
    session, peer, source, callers = owned_session
    for transport in peer.transports:
        session._loop.call_soon_threadsafe(transport.closed.set)
    source.release = asyncio.Event()
    finished, errors = _close_in_thread(session, callers)
    assert source.started.wait(timeout=WATCHDOG)

    assert finished.wait(timeout=WATCHDOG), "Source cleanup blocked close indefinitely"
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert source.cancelled.is_set()
    assert source.calls == 1
    assert session._loop.is_closed()
    with pytest.raises(TimeoutError):
        session.close()


def test_peer_error_still_aborts_owned_transports_and_cleans_source(
    owned_session, monkeypatch
):
    session, peer, source, _ = owned_session

    async def fail():
        raise RuntimeError("peer shutdown failed")

    monkeypatch.setattr(peer, "close", fail)
    with pytest.raises(RuntimeError, match="peer shutdown failed"):
        session.close()

    assert [transport.abort_calls for transport in peer.transports] == [1, 1]
    assert source.calls == 1
    assert session._loop.is_closed()


@pytest.mark.parametrize(
    "setting", ["WEBRTC_CLEANUP_TIMEOUT", "WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_cleanup_budget_fails_before_starting_resources(
    monkeypatch, setting, value
):
    monkeypatch.setattr(session_module, setting, value)
    with pytest.raises(ValueError, match="finite and greater than zero"):
        WebRTCSession(
            api_url="http://127.0.0.1:9",
            api_key="test-key",
            source=_Source(),
            image_input_name="image",
            workflow_config={},
            stream_config=MagicMock(),
        )


def test_real_aiortc_and_stun_close_finish_after_owned_abort(owned_session):
    from aioice.ice import StunProtocol
    from aiortc import RTCConfiguration, RTCPeerConnection

    session, _, source, callers = owned_session
    carriers = []
    holder = {}

    class StalledCarrier:
        def __init__(self, protocol, loop):
            self.protocol = protocol
            self.loop = loop
            self.abort_calls = 0

        def close(self):
            # Model a kernel transport which does not deliver connection_lost.
            pass

        def abort(self):
            self.abort_calls += 1
            self.loop.call_soon(self.protocol.connection_lost, None)

    async def initialize_real_peer():
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peer.createDataChannel("owned-test")
        connection = peer.sctp.transport.transport._connection
        for _ in range(2):
            protocol = StunProtocol(connection)
            carrier = StalledCarrier(protocol, asyncio.get_running_loop())
            protocol.connection_made(carrier)
            connection._protocols.append(protocol)
            carriers.append(carrier)
        session._pc = peer
        holder["peer"] = peer
        holder["connection"] = connection

    asyncio.run_coroutine_threadsafe(initialize_real_peer(), session._loop).result(
        timeout=WATCHDOG
    )
    finished, errors = _close_in_thread(session, callers)
    try:
        assert finished.wait(timeout=WATCHDOG)
        assert errors == []
        assert [carrier.abort_calls for carrier in carriers] == [1, 1]
        assert holder["peer"].connectionState == "closed"
        assert holder["connection"]._closed
        assert holder["connection"]._protocols == []
        assert source.calls == 1
        assert session._loop.is_closed()
    finally:
        if session._loop.is_running():
            for carrier in carriers:
                session._loop.call_soon_threadsafe(carrier.abort)


@pytest.mark.parametrize("carrier_kind", ["udp", "tcp"])
def test_real_turn_wrapper_releases_protocol_through_its_owned_carrier(carrier_kind):
    from aioice.ice import Connection, StunProtocol
    from aioice.turn import TurnClientTcpProtocol, TurnClientUdpProtocol, TurnTransport

    from inference_sdk.webrtc._cleanup import _abort_peer_transports

    async def exercise():
        connection = Connection(ice_controlling=False)
        protocol = StunProtocol(connection)
        cls = TurnClientUdpProtocol if carrier_kind == "udp" else TurnClientTcpProtocol
        inner = cls(
            server=("127.0.0.1", 9),
            username="test",
            password="test",
            lifetime=600,
            channel_refresh_time=500,
        )
        inner.receiver = protocol

        class Carrier:
            def __init__(self):
                self.calls = 0

            def abort(self):
                self.calls += 1
                inner.connection_lost(None)

        carrier = Carrier()
        inner.connection_made(carrier)
        # aioice 0.9 (aiortc's supported floor) also takes the receiver here.
        wrapper = (
            TurnTransport(protocol, inner)
            if "protocol" in inspect.signature(TurnTransport).parameters
            else TurnTransport(inner)
        )
        protocol.connection_made(wrapper)
        connection._protocols.append(protocol)
        ice = SimpleNamespace(_connection=connection)
        peer = SimpleNamespace(
            sctp=SimpleNamespace(transport=SimpleNamespace(transport=ice))
        )
        _abort_peer_transports(peer)
        assert carrier.calls == 1
        closed = getattr(protocol, "_StunProtocol__closed")
        assert closed.done() and not closed.cancelled()

    asyncio.run(exercise())


def test_one_abort_failure_does_not_strand_other_owned_transports(
    owned_session, monkeypatch
):
    session, peer, source, _ = owned_session

    def fail_abort():
        peer.transports[0].abort_calls += 1
        raise OSError("controlled abort failure")

    monkeypatch.setattr(peer.transports[0], "abort", fail_abort)
    with pytest.raises(RuntimeError, match="Failed to abort a WebRTC ICE transport"):
        session.close()
    assert [transport.abort_calls for transport in peer.transports] == [1, 1]
    assert source.calls == 1
    assert session._loop.is_closed()


def test_unrecognized_turn_layout_fails_visibly_and_preserves_unrelated_wrapper():
    from aioice.turn import TurnTransport

    from inference_sdk.webrtc._cleanup import _abort_peer_transports

    malformed = object.__new__(TurnTransport)
    unrelated = SimpleNamespace(transport=_Transport())
    ice = SimpleNamespace(
        _connection=SimpleNamespace(_protocols=[SimpleNamespace(transport=malformed)])
    )
    peer = SimpleNamespace(
        sctp=SimpleNamespace(transport=SimpleNamespace(transport=ice))
    )
    with pytest.raises(
        RuntimeError, match="Failed to abort a WebRTC ICE transport"
    ) as error:
        _abort_peer_transports(peer)
    assert "Unsupported WebRTC TURN transport layout" in str(error.value.__cause__)
    assert unrelated.transport.abort_calls == 0


@pytest.mark.parametrize("stall", [False, True])
def test_async_generator_shutdown_reports_incomplete_finalization(owned_session, stall):
    session, peer, _, _ = owned_session
    for transport in peer.transports:
        session._loop.call_soon_threadsafe(transport.closed.set)
    observed = {"started": False, "finished": False}

    async def generator():
        try:
            yield 1
        finally:
            observed["started"] = True
            if stall:
                await asyncio.Event().wait()
            observed["finished"] = True

    async def open_generator():
        session.retained_generator = generator()
        await session.retained_generator.__anext__()

    asyncio.run_coroutine_threadsafe(open_generator(), session._loop).result(WATCHDOG)
    if stall:
        with pytest.raises(TimeoutError, match="async generators") as first:
            session.close()
        with pytest.raises(TimeoutError) as duplicate:
            session.close()
        assert duplicate.value is first.value
    else:
        session.close()
        session.close()
    assert observed == {"started": True, "finished": not stall}
    assert session._loop.is_closed()
    assert not session._loop_thread.is_alive()


def test_secondary_cleanup_error_does_not_replace_first_failure(
    owned_session, monkeypatch, caplog
):
    session, peer, source, _ = owned_session
    primary = RuntimeError("original peer failure")

    async def fail_peer():
        raise primary

    async def fail_source():
        source.calls += 1
        raise TimeoutError("secondary source failure")

    monkeypatch.setattr(peer, "close", fail_peer)
    monkeypatch.setattr(source, "cleanup", fail_source)
    with pytest.raises(RuntimeError, match="original peer failure") as first:
        session.close()
    assert first.value is primary
    with pytest.raises(RuntimeError) as duplicate:
        session.close()
    assert duplicate.value is primary
    assert "secondary source failure" in caplog.text
    assert source.calls == 1
    assert session._loop.is_closed()


@pytest.mark.parametrize("resource", ["task", "async_generator"])
def test_finalizer_exception_is_visible_to_first_and_duplicate_close(
    owned_session, resource
):
    session, peer, _, _ = owned_session
    for transport in peer.transports:
        session._loop.call_soon_threadsafe(transport.closed.set)
    primary = RuntimeError("finalizer failed")

    async def generator():
        try:
            yield 1
        finally:
            raise primary

    async def worker():
        try:
            await asyncio.Event().wait()
        finally:
            raise primary

    async def create_resource():
        if resource == "task":
            session.retained_worker = asyncio.create_task(worker())
            await asyncio.sleep(0)
        else:
            session.retained_generator = generator()
            await session.retained_generator.__anext__()

    asyncio.run_coroutine_threadsafe(create_resource(), session._loop).result(WATCHDOG)
    with pytest.raises(RuntimeError, match="finalizer failed") as first:
        session.close()
    with pytest.raises(RuntimeError) as duplicate:
        session.close()
    assert first.value is duplicate.value is primary
    assert session._loop.is_closed()
    assert not session._loop_thread.is_alive()


def test_asyncgen_error_capture_preserves_existing_loop_exception_handler(
    owned_session,
):
    session, peer, _, _ = owned_session
    for transport in peer.transports:
        session._loop.call_soon_threadsafe(transport.closed.set)
    contexts = []
    primary = RuntimeError("asyncgen finalizer failed")
    unrelated = ValueError("unrelated handler context")

    def existing_handler(loop, context):
        contexts.append(context)

    async def generator():
        try:
            yield 1
        finally:
            asyncio.get_running_loop().call_exception_handler({"exception": unrelated})
            raise primary

    async def open_generator():
        session._loop.set_exception_handler(existing_handler)
        session.retained_generator = generator()
        await session.retained_generator.__anext__()

    asyncio.run_coroutine_threadsafe(open_generator(), session._loop).result(WATCHDOG)
    with pytest.raises(RuntimeError) as error:
        session.close()
    assert error.value is primary
    assert [context["exception"] for context in contexts] == [unrelated, primary]
    assert session._loop.get_exception_handler() is existing_handler


def test_cancelled_real_startup_cannot_allocate_peer_after_close(
    owned_session, monkeypatch
):
    from aiortc import RTCConfiguration

    session, _, source, _ = owned_session
    session._pc = None
    entered = threading.Event()
    release = asyncio.Event()

    async def get_config():
        return RTCConfiguration(iceServers=[])

    async def delayed_probe(config):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        return config

    async def cleanup():
        source.calls += 1
        release.set()
        await asyncio.sleep(0)

    constructor = MagicMock(side_effect=AssertionError("Peer allocated after close"))
    monkeypatch.setattr("aiortc.RTCPeerConnection", constructor)
    monkeypatch.setattr(session, "_init", WebRTCSession._init.__get__(session))
    monkeypatch.setattr(session, "_get_turn_config", get_config)
    monkeypatch.setattr(session_module, "prefer_reachable_turn", delayed_probe)
    monkeypatch.setattr(source, "cleanup", cleanup)
    startup = asyncio.run_coroutine_threadsafe(session._run_startup(), session._loop)
    assert entered.wait(WATCHDOG)
    with pytest.raises(TimeoutError, match="startup cancellation"):
        session.close()
    assert startup.done()
    constructor.assert_not_called()
    assert session._pc is None
    assert source.calls == 1
    assert session._loop.is_closed()


def test_outer_wait_handles_pre_311_future_timeout_class(owned_session, monkeypatch):
    session, _, _, _ = owned_session

    class LegacyFutureTimeout(Exception):
        pass

    future = MagicMock()
    future.result.side_effect = LegacyFutureTimeout()
    future.done.return_value = False

    def stalled_submission(coro, loop):
        coro.close()
        return future

    monkeypatch.setattr(
        session_module, "FutureTimeoutError", LegacyFutureTimeout, raising=False
    )
    monkeypatch.setattr(
        session_module.asyncio, "run_coroutine_threadsafe", stalled_submission
    )
    with pytest.raises(TimeoutError, match="cleanup event loop did not respond"):
        session.close()
    future.cancel.assert_called_once_with()
    assert session._loop.is_closed()
    assert not session._loop_thread.is_alive()
