import atexit
import multiprocessing.util
import os
import signal
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from asyncua.ua.uaerrors import BadMaxConnectionsReached, BadTooManySessions

from inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer import v1
from inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    OPCUAConnectionManager,
    SessionExhaustedError,
)

V1 = "inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer.v1"
URL = "opc.tcp://localhost:4840/freeopcua/server/"


@pytest.fixture
def manager():
    """Provide a fresh connection manager, leaving global state as it was found."""
    previous_instance = OPCUAConnectionManager._instance
    previous_global = v1._connection_manager
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    OPCUAConnectionManager._instance = None
    v1._connection_manager = None

    manager = OPCUAConnectionManager()

    yield manager

    atexit.unregister(manager.shutdown)
    if manager._finalizer is not None:
        manager._finalizer.cancel()
    signal.signal(signal.SIGTERM, previous_sigterm_handler)
    OPCUAConnectionManager._instance = previous_instance
    v1._connection_manager = previous_global


def test_create_client_requests_short_session_timeout(manager) -> None:
    # given
    created_client = MagicMock()

    # when
    with patch(f"{V1}.Client", return_value=created_client), patch.object(
        manager, "_get_tloop", return_value=MagicMock()
    ):
        client = manager._create_client(
            url=URL, user_name=None, password=None, timeout=2
        )

    # then - asyncua defaults to 1h, which servers clamp to ~50min for orphaned sessions
    assert client is created_client
    assert created_client.aio_obj.session_timeout == int(
        v1.SESSION_TIMEOUT_SECONDS * 1000
    )


def test_session_timeout_defaults_to_one_minute() -> None:
    # given - asserted against a cleared environment, so a deployment that legitimately
    # sets OPC_SESSION_TIMEOUT_SECONDS does not fail the suite
    with patch.dict(v1.os.environ):
        v1.os.environ.pop("OPC_SESSION_TIMEOUT_SECONDS", None)

        # when
        default = v1._positive_float_from_env("OPC_SESSION_TIMEOUT_SECONDS", 60.0)

    # then
    assert default == 60.0


@pytest.mark.parametrize("error", [BadTooManySessions(), BadMaxConnectionsReached()])
def test_connect_with_retry_fails_fast_when_server_is_out_of_sessions(
    manager, error
) -> None:
    # given
    client = MagicMock()
    client.connect.side_effect = error

    # when
    started_at = time.monotonic()
    with pytest.raises(SessionExhaustedError):
        manager._connect_with_retry(
            client=client, url=URL, max_retries=3, base_backoff=5.0
        )
    elapsed = time.monotonic() - started_at

    # then - a slot only frees up when another session times out, so retrying cannot help
    assert client.connect.call_count == 1
    assert elapsed < 1.0


def test_connect_with_retry_still_retries_transient_errors(manager) -> None:
    # given
    client = MagicMock()
    client.connect.side_effect = [ConnectionRefusedError("nope"), None]

    # when
    started_at = time.monotonic()
    manager._connect_with_retry(client=client, url=URL, max_retries=3, base_backoff=0.1)
    elapsed = time.monotonic() - started_at

    # then
    assert client.connect.call_count == 2
    assert elapsed >= 0.1


def test_get_connection_backs_off_for_seconds_after_session_limit(manager) -> None:
    # when
    with patch.object(
        manager, "_create_client", return_value=MagicMock()
    ), patch.object(
        manager, "_connect_with_retry", side_effect=SessionExhaustedError("full")
    ) as connect_mock:
        with pytest.raises(SessionExhaustedError):
            manager.get_connection(url=URL, user_name=None, password=None, timeout=2)

        # then - we stay away for tens of seconds, not the generic 2s circuit breaker
        _, timeout_seconds = manager._server_backoff[URL]
        assert timeout_seconds >= v1.SESSION_EXHAUSTION_BACKOFF_SECONDS
        assert (
            timeout_seconds
            <= v1.SESSION_EXHAUSTION_BACKOFF_SECONDS
            + v1.SESSION_EXHAUSTION_BACKOFF_JITTER_SECONDS
        )

        # and the next call does not touch the server at all
        with pytest.raises(SessionExhaustedError) as error:
            manager.get_connection(url=URL, user_name=None, password=None, timeout=2)
        assert "SESSION LIMIT ERROR" in str(error.value)
        assert connect_mock.call_count == 1


def test_get_connection_uses_short_circuit_timeout_for_other_failures(manager) -> None:
    # given
    key = manager._get_connection_key(URL, None)

    # when
    with patch.object(
        manager, "_create_client", return_value=MagicMock()
    ), patch.object(
        manager, "_connect_with_retry", side_effect=Exception("NETWORK ERROR")
    ):
        with pytest.raises(Exception):
            manager.get_connection(url=URL, user_name=None, password=None, timeout=2)

    # then
    _, timeout_seconds = manager._connection_failures[key]
    assert timeout_seconds == manager.CIRCUIT_BREAKER_TIMEOUT_SECONDS


def test_circuit_breaker_clears_once_its_timeout_elapsed(manager) -> None:
    # given
    key = manager._get_connection_key(URL, None)
    manager._connection_failures[key] = (time.monotonic() - 6.0, 5.0)

    # when
    remaining = manager._circuit_open_seconds_remaining(key)

    # then
    assert remaining is None
    assert key not in manager._connection_failures


def test_circuit_breaker_reports_time_left_while_open(manager) -> None:
    # given
    key = manager._get_connection_key(URL, None)
    manager._connection_failures[key] = (time.monotonic(), 30.0)

    # when
    remaining = manager._circuit_open_seconds_remaining(key)

    # then
    assert remaining is not None
    assert 29.0 <= remaining <= 30.0


def test_shutdown_closes_pooled_sessions_and_refuses_new_ones(manager) -> None:
    # given
    pooled_client = MagicMock()
    key = manager._get_connection_key(URL, None)
    manager._connections[key] = pooled_client
    manager._connection_metadata[key] = {
        "url": URL,
        "user_name": None,
        "password": None,
        "timeout": 2,
        "connected_at": v1.datetime.now(),
    }

    # when
    manager.shutdown()

    # then
    pooled_client.disconnect.assert_called_once()
    assert manager.get_pool_stats()["total_connections"] == 0
    with pytest.raises(Exception) as error:
        manager.get_connection(url=URL, user_name=None, password=None, timeout=2)
    assert "SHUTTING DOWN" in str(error.value)


def test_connection_opened_during_shutdown_is_closed_without_holding_the_pool_lock(
    manager,
) -> None:
    """`close_all` needs that lock to enforce its budget, so no OPC I/O may hold it."""
    # given - a connect that lands just after shutdown began
    opened_client = MagicMock()
    lock_free_per_disconnect = []

    def probe_lock_then_disconnect(client):
        # RLock is reentrant, so only a *different* thread can tell us whether the lock
        # is actually free while this disconnect is in flight. Every disconnect is
        # recorded, so an extra one made under the lock cannot be masked by a later one.
        outcome = []

        def try_acquire():
            acquired = manager._global_lock.acquire(blocking=False)
            outcome.append(acquired)
            if acquired:
                manager._global_lock.release()

        prober = threading.Thread(target=try_acquire)
        prober.start()
        prober.join(timeout=5)
        lock_free_per_disconnect.append(outcome == [True])

    def shutdown_starts_mid_connect(*args, **kwargs):
        manager._shutting_down = True

    # when
    with patch.object(
        manager, "_create_client", return_value=opened_client
    ), patch.object(
        manager, "_connect_with_retry", side_effect=shutdown_starts_mid_connect
    ), patch.object(
        manager, "_safe_disconnect", side_effect=probe_lock_then_disconnect
    ):
        with pytest.raises(Exception) as error:
            manager.get_connection(url=URL, user_name=None, password=None, timeout=2)

    # then - the session is handed back, and shutdown was never blocked behind it
    assert "SHUTTING DOWN" in str(error.value)
    assert lock_free_per_disconnect == [True]
    assert manager.get_pool_stats()["total_connections"] == 0


def test_invalidate_connection_survives_the_pool_being_cleared_mid_disconnect(
    manager,
) -> None:
    """A concurrent `close_all` sweep must not turn an invalidation into a KeyError."""
    # given
    key = manager._get_connection_key(URL, None)
    pooled_client = MagicMock()
    manager._connections[key] = pooled_client
    manager._connection_metadata[key] = {
        "url": URL,
        "user_name": None,
        "password": None,
        "timeout": 2,
        "connected_at": v1.datetime.now(),
        "pid": os.getpid(),
    }

    def sweep_the_pool_during_disconnect(client):
        # Exactly what `close_all` does while a disconnect is in flight. Disconnecting
        # before detaching would leave the removal reaching for a key that is now gone.
        manager._connections.clear()
        manager._connection_metadata.clear()

    # when
    with patch.object(
        manager, "_safe_disconnect", side_effect=sweep_the_pool_during_disconnect
    ):
        manager.invalidate_connection(URL, None)  # must not raise

    # then
    assert manager.get_pool_stats()["total_connections"] == 0


def test_invalidated_client_stays_visible_to_a_shutdown_sweep(manager) -> None:
    """Detaching before the disconnect would let shutdown stop the loop and strand it."""
    # given
    key = manager._get_connection_key(URL, None)
    pooled_client = MagicMock()
    manager._connections[key] = pooled_client
    manager._connection_metadata[key] = {
        "url": URL,
        "user_name": None,
        "password": None,
        "timeout": 2,
        "connected_at": v1.datetime.now(),
        "pid": os.getpid(),
    }
    seen_by_sweep = []

    def sweep_while_disconnecting(client):
        # `close_all` runs its sweep at the moment this disconnect is in flight. It must
        # still find the client, otherwise it stops the ThreadLoop and no CloseSession
        # ever reaches the server.
        seen_by_sweep.append(list(manager._connections))

    # when
    with patch.object(
        manager, "_safe_disconnect", side_effect=sweep_while_disconnecting
    ):
        manager.invalidate_connection(URL, None)

    # then
    assert seen_by_sweep == [[key]]
    assert manager.get_pool_stats()["total_connections"] == 0


def test_invalidation_holds_the_key_lock_until_the_old_session_is_gone(manager) -> None:
    """A replacement opened before teardown finishes doubles this key's server slots."""
    # given
    key = manager._get_connection_key(URL, None)
    manager._connections[key] = MagicMock()
    manager._connection_metadata[key] = {
        "url": URL,
        "user_name": None,
        "password": None,
        "timeout": 2,
        "connected_at": v1.datetime.now(),
        "pid": os.getpid(),
    }
    key_lock_free_during_disconnect = []

    def probe_key_lock(client):
        outcome = []
        key_lock = manager._get_connection_lock(key)

        def probe():
            acquired = key_lock.acquire(blocking=False)
            outcome.append(acquired)
            if acquired:
                key_lock.release()

        prober = threading.Thread(target=probe)
        prober.start()
        prober.join(timeout=5)
        key_lock_free_during_disconnect.append(outcome == [True])

    # when
    with patch.object(manager, "_safe_disconnect", side_effect=probe_key_lock):
        manager.invalidate_connection(URL, None)

    # then - another writer to this server cannot slip in mid-teardown
    assert key_lock_free_during_disconnect == [False]


def test_concurrent_first_use_does_not_reset_a_live_pool() -> None:
    """`__init__` runs on every construction: a second caller must not wipe the first's pool."""
    # given - the singleton is unbuilt, and two threads reach it at the same moment
    previous_instance = OPCUAConnectionManager._instance
    previous_global = v1._connection_manager
    previous_handler = signal.getsignal(signal.SIGTERM)
    OPCUAConnectionManager._instance = None
    v1._connection_manager = None

    lock_free_during_init = []
    real_reset = OPCUAConnectionManager._reset_process_local_state
    built = []

    def probing_reset(self):
        real_reset(self)
        # A second thread must not be able to enter initialization while this one is
        # mid-reset - that is exactly what would swap the pool out from under it.
        outcome = []

        def probe():
            acquired = OPCUAConnectionManager._lock.acquire(blocking=False)
            outcome.append(acquired)
            if acquired:
                OPCUAConnectionManager._lock.release()

        prober = threading.Thread(target=probe)
        prober.start()
        prober.join(timeout=5)
        lock_free_during_init.append(outcome == [True])

    try:
        # when
        with patch.object(
            OPCUAConnectionManager, "_reset_process_local_state", probing_reset
        ):
            built.append(OPCUAConnectionManager())

        manager = built[0]
        key = manager._get_connection_key(URL, None)
        manager._connections[key] = MagicMock()
        again = OPCUAConnectionManager()

        # then - initialization was exclusive, and re-construction is a no-op that leaves
        # the live session tracked
        assert lock_free_during_init == [False]
        assert again is manager
        assert manager._initialized is True
        assert manager.get_pool_stats()["total_connections"] == 1
        assert manager._finalizer is not None
    finally:
        for candidate in built:
            atexit.unregister(candidate.shutdown)
            if getattr(candidate, "_finalizer", None) is not None:
                candidate._finalizer.cancel()
        signal.signal(signal.SIGTERM, previous_handler)
        OPCUAConnectionManager._instance = previous_instance
        v1._connection_manager = previous_global


def test_fork_hook_replaces_the_inherited_singleton_lock() -> None:
    """A lock held by a parent thread at fork time is unacquirable in the child forever."""
    # given - the construction lock is held, as it would be mid-`__new__` during a fork
    previous_lock = OPCUAConnectionManager._lock
    previous_instance = OPCUAConnectionManager._instance
    try:
        OPCUAConnectionManager._lock.acquire()
        held_lock = OPCUAConnectionManager._lock

        # when - the child hook runs
        v1._adopt_connection_manager_in_child()

        # then - the child gets a fresh, acquirable lock instead of the stuck one
        assert OPCUAConnectionManager._lock is not held_lock
        assert OPCUAConnectionManager._lock.acquire(blocking=False) is True
        OPCUAConnectionManager._lock.release()
    finally:
        if held_lock.locked():
            held_lock.release()
        OPCUAConnectionManager._lock = previous_lock
        OPCUAConnectionManager._instance = previous_instance


def test_sigterm_handler_is_installed_on_initialization(manager) -> None:
    # when
    handler = signal.getsignal(signal.SIGTERM)

    # then
    assert getattr(handler, "func", None) == manager._handle_sigterm


def test_sigterm_handler_leaves_the_pool_to_an_existing_shutdown_path(manager) -> None:
    """Handlers like uvicorn's flip a flag and return - the drain happens after they do."""
    # given
    pooled_client = MagicMock()
    manager._connections[manager._get_connection_key(URL, None)] = pooled_client
    calls = []
    previous_handler = lambda signum, frame: calls.append(("previous", signum))

    # when
    manager._handle_sigterm(previous_handler, signal.SIGTERM, None)

    # then - closing now would fail every write still being drained during the grace period
    assert calls == [("previous", signal.SIGTERM)]
    pooled_client.disconnect.assert_not_called()
    assert manager._shutting_down is False
    assert manager.get_pool_stats()["total_connections"] == 1


def test_sigterm_handler_lets_an_exiting_previous_handler_through(manager) -> None:
    # given
    pooled_client = MagicMock()
    manager._connections[manager._get_connection_key(URL, None)] = pooled_client

    def previous_handler(signum, frame):
        raise SystemExit(0)

    # when
    with pytest.raises(SystemExit):
        manager._handle_sigterm(previous_handler, signal.SIGTERM, None)

    # then - the exit unwinds the interpreter, and `atexit` is what closes the pool
    pooled_client.disconnect.assert_not_called()
    assert manager.get_pool_stats()["total_connections"] == 1


def test_sigterm_handler_closes_sessions_when_nothing_was_handling_it(manager) -> None:
    # given - SIG_DFL means the signal kills the process outright, so no exit hook runs
    pooled_client = MagicMock()
    manager._connections[manager._get_connection_key(URL, None)] = pooled_client

    # when
    with patch(f"{V1}.os.kill") as kill_mock:
        manager._handle_sigterm(signal.SIG_DFL, signal.SIGTERM, None)

    # then - this is the only chance to hand the sessions back
    pooled_client.disconnect.assert_called_once()
    assert manager.get_pool_stats()["total_connections"] == 0
    # and the signal must still terminate the process the way it would have
    kill_mock.assert_called_once_with(v1.os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_an_ignored_sigterm_is_left_alone_and_the_manager_stays_usable() -> None:
    """A process that ignores SIGTERM must not have its pool torn down behind its back."""
    # given
    previous_instance = OPCUAConnectionManager._instance
    previous_global = v1._connection_manager
    previous_handler = signal.getsignal(signal.SIGTERM)
    OPCUAConnectionManager._instance = None
    v1._connection_manager = None
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        # when
        manager = OPCUAConnectionManager()

        # then - the disposition is untouched, so SIGTERM never reaches us and writes
        # keep working
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert manager._shutting_down is False
        with patch.object(
            manager, "_create_client", return_value=MagicMock()
        ), patch.object(manager, "_connect_with_retry"):
            assert (
                manager.get_connection(
                    url=URL, user_name=None, password=None, timeout=2
                )
                is not None
            )
    finally:
        atexit.unregister(manager.shutdown)
        if manager._finalizer is not None:
            manager._finalizer.cancel()
        signal.signal(signal.SIGTERM, previous_handler)
        OPCUAConnectionManager._instance = previous_instance
        v1._connection_manager = previous_global


def test_shutdown_hooks_cover_processes_that_never_run_atexit(manager) -> None:
    """Forked workers leave via os._exit(), which skips atexit but runs mp finalizers."""
    # then
    assert manager._finalizer is not None
    assert manager._finalizer.still_active()


def test_shared_threadloop_is_daemonised(manager) -> None:
    """A non-daemon loop would be joined before atexit, so the hook could never run."""
    # when
    tloop = manager._get_tloop()

    # then
    try:
        assert tloop.daemon is True
    finally:
        with manager._global_lock:
            manager._stop_tloop()


def test_connection_opened_while_shutting_down_is_discarded(manager) -> None:
    """Shutdown can start mid-connect: the new session must not land in a swept pool."""
    # given
    fresh_client = MagicMock()

    def start_shutdown(*args, **kwargs):
        manager._shutting_down = True

    # when - shutdown lands between creating the client and publishing it
    with patch.object(
        manager, "_create_client", return_value=fresh_client
    ), patch.object(manager, "_connect_with_retry", side_effect=start_shutdown):
        with pytest.raises(Exception) as error:
            manager.get_connection(url=URL, user_name=None, password=None, timeout=2)

    # then
    assert "SHUTTING DOWN" in str(error.value)
    fresh_client.disconnect.assert_called_once()
    assert manager.get_pool_stats()["total_connections"] == 0


def test_sigterm_cleanup_does_not_deadlock_against_a_held_global_lock(manager) -> None:
    """Signal handlers run on the main thread, which may already hold the pool lock."""
    # given
    pooled_client = MagicMock()
    manager._connections[manager._get_connection_key(URL, None)] = pooled_client

    # when - SIGTERM arrives while this thread is inside a locked section, on the path
    # that actually closes the pool (nothing else was handling the signal)
    with patch(f"{V1}.os.kill"):
        with manager._global_lock:
            manager._handle_sigterm(signal.SIG_DFL, signal.SIGTERM, None)

    # then - it completed instead of blocking on the lock it already holds
    pooled_client.disconnect.assert_called_once()
    assert manager.get_pool_stats()["total_connections"] == 0


def test_pooled_connection_survives_a_session_limit_hit_on_another_credential(
    manager,
) -> None:
    """A healthy session must not be torn down because the server is full for someone else."""
    # given - one credential already has a working pooled connection
    pooled_client = MagicMock()
    pooled_key = manager._get_connection_key(URL, "operator")
    manager._connections[pooled_key] = pooled_client

    # when - another credential is refused because the server is out of slots
    with patch.object(
        manager, "_create_client", return_value=MagicMock()
    ), patch.object(
        manager, "_connect_with_retry", side_effect=SessionExhaustedError("full")
    ):
        with pytest.raises(SessionExhaustedError):
            manager.get_connection(
                url=URL, user_name="engineer", password="other", timeout=2
            )

    # then - the established session is still handed out, not refused and invalidated
    assert (
        manager.get_connection(
            url=URL, user_name="operator", password="secret", timeout=2
        )
        is pooled_client
    )
    pooled_client.disconnect.assert_not_called()


def test_shutdown_deadline_is_shared_across_every_pooled_connection(manager) -> None:
    """Grace periods do not grow with the number of unresponsive servers we pooled."""
    # given - several connections whose disconnects each burn a chunk of the deadline
    tloop = MagicMock()
    manager._tloop = tloop
    budgets = []
    for index in range(4):
        client = MagicMock()
        client.disconnect.side_effect = lambda: budgets.append(tloop.timeout)
        manager._connections[f"opc.tcp://server-{index}|"] = client

    # when
    with patch.object(manager, "_stop_tloop"):
        manager.close_all()

    # then - each disconnect gets what is left of one budget, never a fresh one
    assert len(budgets) == 4
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[0] <= v1.SHUTDOWN_CALL_TIMEOUT_SECONDS


def test_shutdown_stops_waiting_on_servers_once_the_budget_is_gone(manager) -> None:
    """An exhausted deadline must not still cost a little more per remaining connection."""
    # given - an unresponsive first server burns the whole shutdown budget
    manager._tloop = MagicMock()
    disconnected = []

    first_client = MagicMock()
    first_client.disconnect.side_effect = lambda: (
        disconnected.append("first"),
        time.sleep(0.15),
    )
    manager._connections["opc.tcp://slow|"] = first_client
    for index in range(3):
        client = MagicMock()
        client.disconnect.side_effect = lambda: disconnected.append("later")
        manager._connections[f"opc.tcp://later-{index}|"] = client

    # when
    with patch.object(manager, "_stop_tloop"), patch.object(
        v1, "SHUTDOWN_CALL_TIMEOUT_SECONDS", 0.05
    ):
        manager.close_all()

    # then - the rest are dropped rather than each waiting out a fresh floor
    assert disconnected == ["first"]
    assert manager.get_pool_stats()["total_connections"] == 0


def test_forked_child_finalizer_is_rearmed_after_bootstrap_clears_it(manager) -> None:
    """multiprocessing wipes the finalizer registry while bootstrapping the child."""
    # given - the child's bootstrap drops whatever the fork hook had registered. Cancel
    # just ours rather than clearing the process-wide registry, which also holds
    # finalizers belonging to queues, pools and anything else the suite has running.
    manager._rearm_child_finalizer()
    manager._finalizer.cancel()
    assert not manager._finalizer.still_active()

    # when - the after-fork callbacks run, which is the point bootstrap reaches next
    manager._rearm_child_finalizer()

    # then
    assert manager._finalizer.still_active()


def test_no_new_event_loop_is_started_once_shutdown_began(manager) -> None:
    # given
    manager._shutting_down = True

    # when / then
    with pytest.raises(Exception) as error:
        manager._get_tloop()
    assert "SHUTTING DOWN" in str(error.value)
    assert manager._tloop is None


def test_manager_inherited_through_fork_is_re_armed(manager) -> None:
    """A forked child inherits a pool it cannot use and a finalizer that will not fire."""
    # given - state that looks like it was inherited from a parent process
    parent_client = MagicMock()
    key = manager._get_connection_key(URL, None)
    manager._connections[key] = parent_client
    manager._connection_metadata[key] = {"url": URL, "pid": os.getpid() - 1}
    manager._tloop = MagicMock()
    manager._shutting_down = True
    parent_finalizer = manager._finalizer
    manager._pid = os.getpid() - 1

    # when
    manager._ensure_process_local_state()

    # then - the parent's sessions are dropped, never closed from here
    parent_client.disconnect.assert_not_called()
    assert manager.get_pool_stats()["total_connections"] == 0
    assert manager._tloop is None
    assert manager._shutting_down is False
    assert manager._pid == os.getpid()
    # and this process gets a finalizer of its own, since the inherited one never fires
    assert manager._finalizer is not parent_finalizer
    assert manager._finalizer.still_active()


def test_session_limit_backoff_applies_to_every_credential_on_that_server(
    manager,
) -> None:
    """A session/connection limit belongs to the server, not to one set of credentials."""
    # given
    with patch.object(
        manager, "_create_client", return_value=MagicMock()
    ), patch.object(
        manager, "_connect_with_retry", side_effect=SessionExhaustedError("full")
    ) as connect_mock:
        with pytest.raises(SessionExhaustedError):
            manager.get_connection(
                url=URL, user_name="operator", password="secret", timeout=2
            )

        # when - a different user targets the same full server
        with pytest.raises(SessionExhaustedError) as error:
            manager.get_connection(
                url=URL, user_name="engineer", password="other", timeout=2
            )

    # then - it waits the backoff out rather than taking another run at the server
    assert "SESSION LIMIT ERROR" in str(error.value)
    assert connect_mock.call_count == 1


def test_close_all_leaves_connections_inherited_from_another_process_alone(
    manager,
) -> None:
    """A forked child must not close a session whose socket belongs to its parent."""
    # given
    inherited_client = MagicMock()
    key = manager._get_connection_key(URL, None)
    manager._connections[key] = inherited_client
    manager._connection_metadata[key] = {
        "url": URL,
        "user_name": None,
        "password": None,
        "timeout": 2,
        "connected_at": v1.datetime.now(),
        "pid": os.getpid() + 1,
    }

    # when
    manager.close_all()

    # then
    inherited_client.disconnect.assert_not_called()
    assert manager.get_pool_stats()["total_connections"] == 0


def test_release_connection_keeps_connection_pooled_by_default(manager) -> None:
    # given
    pooled_client = MagicMock()
    key = manager._get_connection_key(URL, None)
    manager._connections[key] = pooled_client

    # when
    manager.release_connection(url=URL, user_name=None)

    # then
    assert manager._connections[key] is pooled_client
    pooled_client.disconnect.assert_not_called()

    # and when
    manager.release_connection(url=URL, user_name=None, force_close=True)

    # then
    assert key not in manager._connections
    pooled_client.disconnect.assert_called_once()


@pytest.mark.parametrize("raw_value", ["not-a-number", "0", "-1", "nan", "inf", "-inf"])
def test_positive_float_from_env_falls_back_to_default(raw_value: str) -> None:
    # when
    with patch.dict(v1.os.environ, {"OPC_TEST_VALUE": raw_value}):
        value = v1._positive_float_from_env("OPC_TEST_VALUE", 60.0)

    # then
    assert value == 60.0


def test_positive_float_from_env_reads_override() -> None:
    # when
    with patch.dict(v1.os.environ, {"OPC_TEST_VALUE": "15"}):
        value = v1._positive_float_from_env("OPC_TEST_VALUE", 60.0)

    # then
    assert value == 15.0
