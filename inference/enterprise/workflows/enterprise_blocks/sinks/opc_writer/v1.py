import atexit
import logging
import math
import multiprocessing.util
import os
import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from asyncua.client import Client as AsyncClient
from asyncua.sync import Client, ThreadLoop, sync_async_client_method
from asyncua.ua import VariantType
from asyncua.ua.uaerrors import (
    BadMaxConnectionsReached,
    BadNoMatch,
    BadTooManySessions,
    BadTypeMismatch,
    BadUserAccessDenied,
)
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.core_steps.sinks.noop import disabled_sink_message


def _positive_float_from_env(variable_name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to `default` if unusable."""
    raw_value = os.getenv(variable_name)
    if raw_value is None:
        return default
    try:
        parsed_value = float(raw_value)
    except ValueError:
        logger.warning(
            f"OPC UA sink ignoring invalid {variable_name}='{raw_value}', using {default}"
        )
        return default
    if not math.isfinite(parsed_value) or parsed_value <= 0:
        # `float()` happily accepts "nan" and "inf": a non-finite timeout breaks every
        # connection attempt, and a non-finite backoff never lets the circuit close.
        logger.warning(
            f"OPC UA sink ignoring non-positive {variable_name}='{raw_value}', using {default}"
        )
        return default
    return parsed_value


# How long the server is asked to keep a session alive without activity. asyncua defaults to
# 1 hour, which servers typically clamp to 50 minutes - long enough for a session left behind
# by a crashed process to squat a server slot for the better part of an hour. asyncua keeps
# pooled idle sessions alive on its own (its watchdog reads `server_state` once a second,
# independently of `cooldown_seconds`), so a short timeout does not cause surprise disconnects.
SESSION_TIMEOUT_SECONDS = _positive_float_from_env("OPC_SESSION_TIMEOUT_SECONDS", 60.0)

# Errors a server returns when it cannot accept another session/connection. Retrying these
# milliseconds apart only hammers a server that is already full, so we fail fast and keep the
# circuit open for tens of seconds instead. Jitter de-synchronises pods that all failed at once.
SESSION_EXHAUSTION_ERRORS = (BadTooManySessions, BadMaxConnectionsReached)
SESSION_EXHAUSTION_BACKOFF_SECONDS = _positive_float_from_env(
    "OPC_SESSION_EXHAUSTION_BACKOFF_SECONDS", 30.0
)
SESSION_EXHAUSTION_BACKOFF_JITTER_SECONDS = 15.0

# Upper bound on calls posted to the shared ThreadLoop while tearing the pool down, so a dead
# server cannot stall process shutdown for the ThreadLoop's regular (much longer) timeout.
SHUTDOWN_CALL_TIMEOUT_SECONDS = 5.0


class SessionExhaustedError(Exception):
    """Raised when the OPC UA server refuses CreateSession because it is at capacity."""


class OPCUAConnectionManager:
    """
    Thread-safe connection manager for OPC UA clients with connection pooling
    and circuit breaker pattern.

    Maintains a pool of connections keyed by (url, user_name) to avoid creating
    new connections for every write operation. Uses circuit breaker to fail fast
    when servers are unreachable.
    """

    _instance: Optional["OPCUAConnectionManager"] = None
    _lock = threading.Lock()

    # Circuit breaker: how long to wait before trying a failed server again
    CIRCUIT_BREAKER_TIMEOUT_SECONDS = 2.0

    def __new__(cls) -> "OPCUAConnectionManager":
        """Singleton pattern to ensure one connection manager across the application."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # `__new__` serialises allocation, but Python calls `__init__` on every construction
        # of the singleton. Two first writes racing here would both see `_initialized` False
        # and both reset: the loser's `_reset_process_local_state()` would swap out the pool
        # and the global lock underneath a thread already connecting, dropping a live client
        # without closing it - an orphaned session, which is the whole point of this change.
        with self._lock:
            if self._initialized:
                return
            self._reset_process_local_state()
            self._register_shutdown_hooks()
            # Last, so nobody can observe a manager whose hooks are not registered yet.
            self._initialized = True
        logger.debug("OPC UA Connection Manager initialized")

    def _reset_process_local_state(self) -> None:
        """Start from an empty pool owned by the current process."""
        self._connections: Dict[str, Client] = {}
        self._connection_locks: Dict[str, threading.Lock] = {}
        self._connection_metadata: Dict[str, dict] = {}
        self._connection_failures: Dict[str, Tuple[float, float]] = (
            {}
        )  # key -> (timestamp of last failure, how long to stay away)
        self._server_backoff: Dict[str, Tuple[float, float]] = (
            {}
        )  # url -> (timestamp, how long to stay away) when the server is out of sessions
        # Reentrant: a SIGTERM handler runs on the main thread and may interrupt that
        # thread while it already holds this lock, then close the pool underneath it.
        self._global_lock = threading.RLock()
        self._tloop: Optional[ThreadLoop] = None
        self._shutting_down = False
        self._finalizer: Optional[multiprocessing.util.Finalize] = None
        self._pid = os.getpid()

    def _rearm_child_finalizer(self) -> None:
        """
        Give the child back the finalizer that its own bootstrap threw away.

        `multiprocessing` clears the whole finalizer registry while bootstrapping a child -
        after the `os.register_at_fork` hook above has already run - so the finalizer that
        hook created is gone by the time the child executes anything. Callbacks registered
        with `register_after_fork` run just *after* that clear, which is the only point
        where a child-owned finalizer survives. Without it a worker that returns normally
        leaves through `os._exit()` with its sessions still open.
        """
        self._finalizer = multiprocessing.util.Finalize(
            None, self.shutdown, exitpriority=16
        )

    def _adopt_in_child(self) -> None:
        """
        Re-arm the manager in a process that inherited it through `fork()`.

        A forked child inherits the singleton fully initialized, but none of it is usable:
        the ThreadLoop thread does not survive `fork()`, so inherited clients cannot talk to
        anything, their sockets belong to the parent, and `multiprocessing.util.Finalize`
        deliberately skips callbacks registered in a different process - so the child would
        run with no shutdown hook at all. Drop the inherited state (without touching the
        parent's sessions) and register hooks owned by this process.

        Deliberately takes no lock. Only the forking thread exists in the child, so there is
        nothing to race with - and the inherited lock may well be held by a parent thread
        that does not exist here, which would block us forever.
        """
        inherited = len(self._connections)
        self._reset_process_local_state()
        self._register_shutdown_hooks()
        logger.debug(
            f"OPC UA Connection Manager re-initialized after fork, dropped "
            f"{inherited} connection(s) belonging to the parent process"
        )

    def _ensure_process_local_state(self) -> None:
        """Re-arm after a fork that somehow bypassed the `os.register_at_fork` hook."""
        if self._pid != os.getpid():
            self._adopt_in_child()

    def _register_shutdown_hooks(self) -> None:
        """
        Make sure pooled sessions are handed back when the process goes away.

        Three hooks, because no single one covers every way this block is deployed:

        * `atexit` - normal interpreter exit.
        * `multiprocessing.util.Finalize` - a forked worker (the video pipeline runs inside
          a `multiprocessing.Process`) leaves through `os._exit()`, which skips `atexit`
          entirely but still runs multiprocessing's own finalizers.
        * a chained `SIGTERM` handler - container restarts and redeploys, where the default
          disposition kills the process before any exit hook runs. Only installable when the
          manager is first used on the main thread; the other two hooks cover the rest.

        Any handler that was already installed still runs, and it keeps ownership of the
        shutdown: when one exists, `_handle_sigterm` chains to it and leaves the pool alone
        so writes being drained during the grace period still succeed, and the exit hooks
        above do the closing. The handler only closes the pool itself when nothing else was
        handling SIGTERM, because then no exit hook will run at all.

        Known gap, accepted deliberately: this manager is created lazily on first write, and
        with the default `fire_and_forget=True` that write runs on a thread pool worker,
        where `signal.signal` is not allowed. A process that both fails to install the
        handler here *and* leaves SIGTERM on its default disposition runs no cleanup at all.

        The two main deployments are covered - the video pipeline runs in a
        `multiprocessing.Process` that installs its own SIGTERM handler and leaves through
        the finalizer above, and the HTTP server lets uvicorn handle SIGTERM and then exits
        through `atexit` - but `inference_cli`'s `process_video_with_workflow` is not: it
        runs `InferencePipeline` directly and installs no SIGTERM handler, so a `docker stop`
        or Ctrl-C-then-TERM there can still leave a session behind. What bounds that is the
        session timeout requested at connect: the server reclaims the slot in ~60s instead of
        the ~50min that caused the incident. Closing the gap outright means calling
        `shutdown()` from explicit application lifecycle code on the main thread (a FastAPI
        shutdown event, the CLI's pipeline teardown), which is outside this block; see
        ENT-1626 for the follow-up.
        """
        atexit.register(self.shutdown)
        # Runs in forked children, where `atexit` does not.
        self._finalizer = multiprocessing.util.Finalize(
            None, self.shutdown, exitpriority=16
        )
        multiprocessing.util.register_after_fork(
            self, type(self)._rearm_child_finalizer
        )
        previous_handler = signal.getsignal(signal.SIGTERM)
        if previous_handler == signal.SIG_IGN:
            # The process deliberately ignores SIGTERM. Taking the slot would turn a signal
            # it wants ignored into a teardown of a pool it is still writing to.
            logger.debug(
                "OPC UA Connection Manager leaving SIGTERM ignored as the process configured it"
            )
            return
        try:
            signal.signal(
                signal.SIGTERM, partial(self._handle_sigterm, previous_handler)
            )
        except (ValueError, OSError) as exc:
            # Signal handlers can only be installed from the main thread of the main
            # interpreter - elsewhere the exit hooks above are what close the pool.
            logger.debug(
                f"OPC UA Connection Manager could not install SIGTERM handler ({exc}), "
                f"relying on exit hooks to close pooled sessions"
            )

    def _handle_sigterm(
        self, previous_handler: Union[Callable, int, None], signum: int, frame
    ) -> None:
        """
        Release pooled sessions on SIGTERM without cutting a graceful shutdown short.

        Whether we close the pool here depends entirely on what was handling SIGTERM before
        us, because that determines whether any exit hook will get to run:

        * A callable handler means the process has its own shutdown path, and it will leave
          through an ordinary exit that runs `atexit` and the multiprocessing finalizer. We
          must NOT tear the pool down here: such handlers commonly just flip a flag and
          return (uvicorn's `handle_exit` does exactly that), so closing now would fail every
          write still being drained during the grace period.
        * `SIG_DFL` (or a handler installed outside Python, which we cannot call) means the
          signal terminates the process outright and no exit hook ever runs, so this is the
          only chance to hand the sessions back.
        """
        if callable(previous_handler):
            logger.debug(
                "OPC UA Connection Manager deferring pool shutdown to the process' own "
                "SIGTERM handler; exit hooks will close pooled sessions"
            )
            previous_handler(signum, frame)
            return
        if previous_handler == signal.SIG_IGN:
            # The process wants SIGTERM ignored and keeps running - closing its pool would
            # be a surprise. We decline to install over SIG_IGN, so this is belt and braces.
            return
        logger.info("OPC UA Connection Manager closing pooled sessions on SIGTERM")
        self.shutdown()
        # Restore the default disposition and let the signal terminate the process the way
        # it would have without our handler.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def shutdown(self) -> None:
        """Close every pooled session and refuse to open new ones."""
        self._shutting_down = True
        self.close_all()

    def _get_tloop(self) -> ThreadLoop:
        """
        Get or create the shared ThreadLoop for all clients.

        Guarded by the global lock: per-server locks do not serialise this, so two first
        connections to different servers could otherwise each start a loop and leave one
        of them unreferenced - and therefore impossible to stop later.
        """
        with self._global_lock:
            if self._shutting_down:
                # Otherwise a connection attempt that started just before the sweep would
                # leave a fresh loop running behind it.
                raise Exception(
                    "SHUTTING DOWN: Connection manager is closing, not starting a new event loop."
                )
            if self._tloop is None or not self._tloop.is_alive():
                logger.debug("OPC UA Connection Manager creating shared ThreadLoop")
                tloop = ThreadLoop(timeout=120)
                # A ThreadLoop inherits its daemon flag from whichever thread created it,
                # and it runs its event loop forever. Created from a non-daemon thread
                # (a `ThreadPoolExecutor` worker, say) it would block interpreter exit -
                # and non-daemon threads are joined *before* `atexit` runs, so the hook
                # that stops it would never get to run. Daemonise it to break that cycle.
                tloop.daemon = True
                tloop.start()
                self._tloop = tloop
            return self._tloop

    def _stop_tloop(self) -> None:
        """Stop the shared ThreadLoop if it exists. Caller must hold the global lock."""
        if self._tloop is not None and self._tloop.is_alive():
            logger.debug("OPC UA Connection Manager stopping shared ThreadLoop")
            try:
                self._tloop.loop.call_soon_threadsafe(self._tloop.loop.stop)
                self._tloop.join(timeout=2.0)
                if not self._tloop.is_alive():
                    self._tloop.loop.close()
            except Exception as exc:
                logger.debug(f"OPC UA Connection Manager ThreadLoop stop error: {exc}")
            self._tloop = None

    def _get_connection_key(self, url: str, user_name: Optional[str]) -> str:
        """Generate a unique key for connection pooling."""
        return f"{url}|{user_name or ''}"

    def _get_connection_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a specific connection."""
        with self._global_lock:
            if key not in self._connection_locks:
                self._connection_locks[key] = threading.Lock()
            return self._connection_locks[key]

    def _create_client(
        self,
        url: str,
        user_name: Optional[str],
        password: Optional[str],
        timeout: int,
    ) -> Client:
        """Create and configure a new OPC UA client using the shared ThreadLoop."""
        logger.debug(f"OPC UA Connection Manager creating client for {url}")
        tloop = self._get_tloop()
        client = Client(url=url, tloop=tloop, sync_wrapper_timeout=timeout)
        # Ask for a short session timeout instead of the asyncua default of 1 hour, so a
        # session orphaned by a crash stops holding a server slot for the better part of an
        # hour. The server has the last word: it may revise the request upwards, and asyncua
        # adopts whatever it returns.
        client.aio_obj.session_timeout = int(SESSION_TIMEOUT_SECONDS * 1000)
        if user_name and password:
            client.set_user(user_name)
            client.set_password(password)
        return client

    def _connect_with_retry(
        self,
        client: Client,
        url: str,
        max_retries: int = 3,
        base_backoff: float = 1.0,
    ) -> None:
        """
        Connect to OPC UA server with retry logic and exponential backoff.

        Args:
            client: The OPC UA client to connect
            url: Server URL (for logging)
            max_retries: Maximum number of connection attempts
            base_backoff: Base delay between retries (seconds), doubles each retry

        Raises:
            SessionExhaustedError: If the server refused to open another session
            Exception: If all connection attempts fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"OPC UA Connection Manager connecting to {url} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                client.connect()
                logger.info(
                    f"OPC UA Connection Manager successfully connected to {url}"
                )
                return
            except BadUserAccessDenied as exc:
                # Auth errors should not be retried - they will keep failing
                logger.error(f"OPC UA Connection Manager authentication failed: {exc}")
                raise Exception(f"AUTH ERROR: {exc}")
            except SESSION_EXHAUSTION_ERRORS as exc:
                # The server is out of session slots. Retrying immediately cannot help - a
                # slot only frees up when another session times out - and every other pod is
                # hitting the same wall, so fail fast and let the circuit breaker hold us off.
                #
                # ENT-1626 asked for a seconds-scale backoff here. We deliberately do not
                # sleep: with `fire_and_forget=False` this runs on the pipeline thread, so a
                # multi-second sleep would stall video processing. The jittered server-wide
                # backoff recorded by the caller achieves the same thing for the server - one
                # attempt per pod per ~30-45s, de-synchronised - at no cost to the pipeline.
                logger.error(
                    f"OPC UA Connection Manager server {url} refused a new session: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise SessionExhaustedError(
                    f"SESSION LIMIT ERROR: server {url} refused a new session "
                    f"({type(exc).__name__}: {exc}). The server is at its session/connection "
                    f"limit - raise it, or reduce the number of clients."
                ) from exc
            except OSError as exc:
                last_exception = exc
                logger.warning(
                    f"OPC UA Connection Manager network error on attempt {attempt + 1}: {exc}"
                )
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"OPC UA Connection Manager connection error on attempt {attempt + 1}: "
                    f"{type(exc).__name__}: {exc}"
                )

            # Don't sleep after the last attempt
            if attempt < max_retries - 1:
                backoff_time = base_backoff * (2**attempt)
                logger.debug(
                    f"OPC UA Connection Manager waiting {backoff_time}s before retry"
                )
                time.sleep(backoff_time)

        # All retries exhausted
        logger.error(
            f"OPC UA Connection Manager failed to connect to {url} "
            f"after {max_retries} attempts"
        )
        if isinstance(last_exception, OSError):
            raise Exception(
                f"NETWORK ERROR: Failed to connect after {max_retries} attempts. Last error: {last_exception}"
            )
        raise Exception(
            f"CONNECTION ERROR: Failed to connect after {max_retries} attempts. Last error: {last_exception}"
        )

    def _circuit_open_seconds_remaining(self, key: str) -> Optional[float]:
        """
        Check if circuit breaker is open (server recently failed).
        Returns the number of seconds left before the next attempt, or None if we may connect.
        """
        if key not in self._connection_failures:
            return None

        failed_at, timeout_seconds = self._connection_failures[key]
        time_since_failure = time.monotonic() - failed_at
        if time_since_failure < timeout_seconds:
            return timeout_seconds - time_since_failure

        # Timeout expired, clear the failure record
        del self._connection_failures[key]
        return None

    def _record_failure(
        self, key: str, timeout_seconds: Optional[float] = None
    ) -> None:
        """Record a connection failure for circuit breaker."""
        if timeout_seconds is None:
            timeout_seconds = self.CIRCUIT_BREAKER_TIMEOUT_SECONDS
        self._connection_failures[key] = (time.monotonic(), timeout_seconds)

    @staticmethod
    def _session_exhaustion_timeout() -> float:
        """Seconds to stay away from a server that is out of session slots (jittered)."""
        return SESSION_EXHAUSTION_BACKOFF_SECONDS + random.uniform(
            0.0, SESSION_EXHAUSTION_BACKOFF_JITTER_SECONDS
        )

    def _record_server_backoff(self, url: str, timeout_seconds: float) -> None:
        """Record that a server is out of session slots, for every credential alike."""
        with self._global_lock:
            self._server_backoff[url] = (time.monotonic(), timeout_seconds)

    def _server_backoff_seconds_remaining(self, url: str) -> Optional[float]:
        """Seconds left before this server should be approached again, or None if now."""
        with self._global_lock:
            if url not in self._server_backoff:
                return None

            failed_at, timeout_seconds = self._server_backoff[url]
            time_since_failure = time.monotonic() - failed_at
            if time_since_failure < timeout_seconds:
                return timeout_seconds - time_since_failure

            del self._server_backoff[url]
            return None

    def _clear_failure(self, key: str) -> None:
        """Clear failure record after successful connection."""
        if key in self._connection_failures:
            del self._connection_failures[key]

    def get_connection(
        self,
        url: str,
        user_name: Optional[str],
        password: Optional[str],
        timeout: int,
        max_retries: int = 1,
        base_backoff: float = 0.0,
    ) -> Client:
        """
        Get a connection from the pool or create a new one.

        This method is thread-safe and will reuse existing healthy connections.
        Uses circuit breaker pattern to fail fast for recently failed servers.

        Args:
            url: OPC UA server URL
            user_name: Optional username for authentication
            password: Optional password for authentication
            timeout: Connection timeout in seconds
            max_retries: Maximum number of connection attempts (default 1)
            base_backoff: Base delay between retries (default 0)

        Returns:
            A connected OPC UA client

        Raises:
            Exception: If connection fails or circuit breaker is open
        """
        self._ensure_process_local_state()
        if self._shutting_down:
            raise Exception(
                f"SHUTTING DOWN: Connection manager is closing, refusing new session to {url}."
            )

        key = self._get_connection_key(url, user_name)
        lock = self._get_connection_lock(key)

        with lock:
            # Check if we have an existing connection. This comes first: both gates below
            # exist to stop us *opening* a session, and a healthy pooled one costs the
            # server nothing. Refusing it here would be worse than useless - the caller
            # treats the refusal as a connection fault and invalidates the very session it
            # was reusing, so a capacity incident on one credential would tear down
            # working streams on another.
            if key in self._connections:
                logger.debug(f"OPC UA Connection Manager reusing connection for {url}")
                return self._connections[key]

            # Server-wide backoff: a session/connection limit belongs to the server, not to
            # one set of credentials, so every key for this URL waits it out.
            server_seconds_remaining = self._server_backoff_seconds_remaining(url)
            if server_seconds_remaining is not None:
                logger.debug(
                    f"OPC UA Connection Manager still backing off from {url}, "
                    f"failing fast (will retry in {server_seconds_remaining:.1f}s)"
                )
                raise SessionExhaustedError(
                    f"SESSION LIMIT ERROR: Server {url} recently refused a new session, "
                    f"skipping connection attempt. Will retry after "
                    f"{server_seconds_remaining:.1f}s."
                )

            # Circuit breaker: fail fast if server recently failed
            seconds_remaining = self._circuit_open_seconds_remaining(key)
            if seconds_remaining is not None:
                logger.debug(
                    f"OPC UA Connection Manager circuit breaker open for {url}, "
                    f"failing fast (will retry in {seconds_remaining:.1f}s)"
                )
                raise Exception(
                    f"CIRCUIT OPEN: Server {url} recently failed, skipping connection attempt. "
                    f"Will retry after {seconds_remaining:.1f}s."
                )

            # Create new connection
            discarded = False
            try:
                client = self._create_client(url, user_name, password, timeout)
                self._connect_with_retry(client, url, max_retries, base_backoff)

                # Success - clear any failure record and store in pool. Publishing under
                # the global lock (the same one `close_all` sweeps under) closes the race
                # where shutdown starts mid-connect: without it this session would land in
                # the pool just after the sweep and never be closed.
                #
                # One narrower ordering is left open on purpose: shutdown can stop the
                # ThreadLoop between `connect()` returning and this block running, and the
                # undo below then cannot post a clean `disconnect()`. Making shutdown wait
                # for in-flight connects would mean blocking a signal handler on an
                # operation that retries with backoff - worse than the exposure, which the
                # requested session timeout caps at ~60s rather than the ~50min this whole
                # change exists to avoid.
                with self._global_lock:
                    self._clear_failure(key)
                    self._connections[key] = client
                    self._connection_metadata[key] = {
                        "url": url,
                        "user_name": user_name,
                        "password": password,
                        "timeout": timeout,
                        "connected_at": datetime.now(),
                        # Which process opened it - a forked child inherits the pool but
                        # must not close sessions that belong to its parent.
                        "pid": os.getpid(),
                    }
                    if self._shutting_down:
                        # Shutdown started while this connection was being opened. Checking
                        # *after* publishing covers both orderings, including a SIGTERM
                        # handler that runs on this very thread and re-enters the lock to
                        # sweep the pool. Undo, so the session is not left behind.
                        self._connections.pop(key, None)
                        self._connection_metadata.pop(key, None)
                        discarded = True

                if discarded:
                    # Disconnect outside the lock. `close_all` needs that lock to apply its
                    # own shutdown budget, so blocking OPC I/O while holding it would pin
                    # the whole teardown to however long an unresponsive server takes to
                    # answer - long enough to burn a container's grace period and get
                    # SIGKILLed, which is the orphaned-session outcome this change exists
                    # to prevent. Cap the wait the same way the sweep does.
                    logger.debug(
                        f"OPC UA Connection Manager discarding connection to {url} "
                        f"opened while shutting down"
                    )
                    if self._tloop is not None:
                        self._tloop.timeout = min(
                            self._tloop.timeout, SHUTDOWN_CALL_TIMEOUT_SECONDS
                        )
                    self._safe_disconnect(client)
                    raise Exception(
                        f"SHUTTING DOWN: Connection manager is closing, "
                        f"discarded new session to {url}."
                    )

                return client
            except SessionExhaustedError:
                # Server is out of session slots - stay away for tens of seconds so a hundred
                # pods do not keep re-attempting against an already-full server.
                timeout_seconds = self._session_exhaustion_timeout()
                self._record_server_backoff(url, timeout_seconds)
                logger.warning(
                    f"OPC UA Connection Manager backing off from {url} for "
                    f"{timeout_seconds:.1f}s after session limit refusal"
                )
                raise
            except Exception as exc:
                # Record failure for circuit breaker
                self._record_failure(key)
                raise

    def _safe_disconnect(self, client: Client) -> None:
        """Safely disconnect a client, swallowing any errors."""
        try:
            logger.debug("OPC UA Connection Manager disconnecting client")
            client.disconnect()
        except Exception as exc:
            logger.debug(
                f"OPC UA Connection Manager disconnect error (non-fatal): {exc}"
            )

    def release_connection(
        self, url: str, user_name: Optional[str], force_close: bool = False
    ) -> None:
        """
        Release a connection back to the pool.

        By default, connections are kept alive for reuse. Set force_close=True
        to immediately close the connection.

        Note that pooled connections are closed on process shutdown (see `shutdown`), so
        keeping one for reuse does not leak a server session past the life of the process.

        Args:
            url: OPC UA server URL
            user_name: Optional username used for the connection
            force_close: If True, close the connection instead of keeping it
        """
        if not force_close:
            # Connection stays in pool for reuse
            return

        self.invalidate_connection(url=url, user_name=user_name)

    def invalidate_connection(self, url: str, user_name: Optional[str]) -> None:
        """
        Invalidate a connection, forcing it to be recreated on next use.

        Call this when a connection error occurs during an operation to ensure
        the next operation gets a fresh connection.

        Args:
            url: OPC UA server URL
            user_name: Optional username used for the connection
        """
        key = self._get_connection_key(url, user_name)
        lock = self._get_connection_lock(key)

        with lock:
            client = self._connections.get(key)
            if client is None:
                return
            # Disconnect while still holding the key lock, and while the client is still
            # listed in the pool. Both matter:
            #
            # * holding the key lock stops another writer opening a replacement session for
            #   this key before the old one has released its slot - exactly the overlap that
            #   pushes a full server into `BadTooManySessions`;
            # * staying listed keeps the client visible to a concurrent `close_all` sweep, so
            #   if shutdown wins the race it closes this session while the ThreadLoop is
            #   still running, instead of stopping the loop and stranding the disconnect.
            #
            # A sweep landing mid-call may close the same client twice, which `_safe_disconnect`
            # swallows, and may clear the pool underneath us - hence `pop`, not `del`, below.
            # This blocks only writers to this one server; `close_all` takes the global lock
            # alone, so shutdown is never queued behind this.
            self._safe_disconnect(client)
            self._connections.pop(key, None)
            self._connection_metadata.pop(key, None)
            logger.debug(f"OPC UA Connection Manager invalidated connection for {url}")

    def close_all(self) -> None:
        """Close all connections in the pool and stop the shared ThreadLoop."""
        current_pid = os.getpid()
        # One deadline for the whole sweep, not one per connection: this runs inside the
        # SIGTERM handler, and a container's termination grace period does not grow with
        # the number of unresponsive servers we happen to be pooling.
        deadline = time.monotonic() + SHUTDOWN_CALL_TIMEOUT_SECONDS
        with self._global_lock:
            for key, client in list(self._connections.items()):
                owner_pid = self._connection_metadata.get(key, {}).get("pid")
                if owner_pid is not None and owner_pid != current_pid:
                    # Inherited across a fork: the socket belongs to the parent, and
                    # closing it here would end a session the parent is still using.
                    logger.debug(
                        f"OPC UA Connection Manager dropping connection inherited from "
                        f"process {owner_pid} without closing it"
                    )
                    continue
                budget = deadline - time.monotonic()
                if budget <= 0:
                    # Out of time. Stop waiting on servers that are not answering: the
                    # process is on its way out and the sockets go with it. Anything left
                    # open is bounded by the session timeout we request at connect.
                    logger.warning(
                        f"OPC UA Connection Manager out of shutdown budget, dropping "
                        f"remaining connection {key} without a clean close"
                    )
                    continue
                if self._tloop is not None:
                    self._tloop.timeout = budget
                self._safe_disconnect(client)
            self._connections.clear()
            self._connection_metadata.clear()
            self._stop_tloop()
            logger.info("OPC UA Connection Manager closed all connections")

    def get_pool_stats(self) -> dict:
        """Get statistics about the connection pool."""
        with self._global_lock:
            return {
                "total_connections": len(self._connections),
                "connections": [
                    {
                        "url": meta["url"],
                        "user_name": meta["user_name"],
                        "connected_at": meta["connected_at"].isoformat(),
                    }
                    for meta in self._connection_metadata.values()
                ],
            }


# Global connection manager instance
_connection_manager: Optional[OPCUAConnectionManager] = None


def _adopt_connection_manager_in_child() -> None:
    """
    Re-arm the pooled manager in a freshly forked child.

    Runs in the child right after `fork()`, while it is still single-threaded, so it can
    replace the inherited lock before anything tries to acquire it. Waiting until first
    use would be too late: the parent may have been holding that lock at the moment of
    the fork, and the thread that would release it does not exist here.
    """
    # First, unconditionally - and before touching `_instance`, which may not exist yet.
    # `fork()` copies locks in whatever state they were in, and the thread that would have
    # released this one does not exist here. A parent thread inside `__new__`/`__init__` at
    # the moment of the fork would otherwise leave this child's first manager construction
    # blocked forever on a lock nobody can release.
    OPCUAConnectionManager._lock = threading.Lock()

    manager = OPCUAConnectionManager._instance
    if manager is not None and getattr(manager, "_initialized", False):
        manager._adopt_in_child()


if hasattr(os, "register_at_fork"):  # not available on Windows
    os.register_at_fork(after_in_child=_adopt_connection_manager_in_child)


def get_connection_manager() -> OPCUAConnectionManager:
    """Get the global OPC UA connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = OPCUAConnectionManager()
    return _connection_manager


class UnsupportedTypeError(Exception):
    """Raised when an unsupported value type is specified"""

    pass


# Exception types that should NOT invalidate the connection (user configuration errors)
USER_CONFIG_ERROR_TYPES = (
    BadTypeMismatch,  # Wrong data type - configuration error
    UnsupportedTypeError,  # Invalid value_type parameter
    ValueError,  # Value range validation errors
)


from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_API_KEY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    TOP_CLASS_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

BLOCK_TYPE = "roboflow_enterprise/opc_writer_sink@v1"
LONG_DESCRIPTION = """
The **OPC UA Writer** block enables you to write data to a variable on an OPC UA server, leveraging the 
[asyncua](https://github.com/FreeOpcUa/opcua-asyncio) library for seamless communication.

### Supported Data Types
This block supports writing the following data types to OPC UA server variables:
- Numbers (integers, floats)
- Booleans
- Strings

**Note:** The data type you send must match the expected type of the target OPC UA variable.

### Node Lookup Mode
The block supports two methods for locating OPC UA nodes via the `node_lookup_mode` parameter:

- **`hierarchical` (default)**: Uses standard OPC UA hierarchical path navigation. The block navigates
  through the address space using `get_child()`. Each component in the `object_name` path is
  automatically prefixed with the namespace index.
  - **Example**: `object_name="Roboflow/Crane_11"` → path `0:Objects/2:Roboflow/2:Crane_11/2:Variable`
  - **Best for**: Traditional OPC UA servers with hierarchical address spaces

- **`direct`**: Uses direct NodeId string access. The block constructs a NodeId as
  `ns={namespace};s={object_name}/{variable_name}` and accesses it directly via `get_node()`.
  - **Example**: `object_name="[Sample_Tags]/Ramp"` → NodeId `ns=2;s=[Sample_Tags]/Ramp/South_Person_Count`
  - **Best for**: Ignition SCADA systems and other servers using string-based NodeId identifiers

### Cooldown
To prevent excessive traffic to the OPC UA server, the block includes a `cooldown_seconds` parameter, 
which defaults to **5 seconds**. During the cooldown period:
- Consecutive executions of the block will set the `throttling_status` output to `True`.
- No data will be sent to the server.

You can customize the `cooldown_seconds` parameter based on your needs. Setting it to `0` disables 
the cooldown entirely.

### Asynchronous Execution
The block provides a `fire_and_forget` property for asynchronous execution:
- **When `fire_and_forget=True`**: The block sends data in the background, allowing the Workflow to 
  proceed immediately. However, the `error_status` output will always be set to `False`, so we do not 
  recommend this mode for debugging.
- **When `fire_and_forget=False`**: The block waits for confirmation before proceeding, ensuring errors 
  are captured in the `error_status` output.

### Disabling the Block Dynamically
You can disable the **OPC UA Writer** block during execution by linking the `disable_sink` parameter 
to a Workflow input. By providing a specific input value, you can dynamically prevent the block from 
executing.

### Connection Pooling
The block uses a connection pool to efficiently manage OPC UA connections. Instead of creating a new
connection for each write operation, connections are reused across multiple writes to the same server.
This significantly reduces latency and resource usage for high-frequency write scenarios.

- Connections are automatically pooled per server URL and username combination
- If a connection fails during a write operation, it is automatically invalidated and a new connection
  is established on the next write attempt
- Pooled connections are closed when the process shuts down, so a restart or redeploy usually hands
  its sessions back to the server instead of leaving them to time out. This covers the Roboflow
  Inference server and video pipelines, on normal exit and on `SIGTERM`. It does *not* cover every
  host: a process that is killed outright (`SIGKILL`, power loss), or one that leaves `SIGTERM` on
  its default disposition and first writes from a background thread (as `inference workflows
  process-video` does), has no opportunity to close anything. The session timeout below is what
  bounds those cases
- Sessions are opened asking for a short server-side timeout (60 seconds by default, override with
  the `OPC_SESSION_TIMEOUT_SECONDS` environment variable) so that a session orphaned by a crash is
  reclaimed quickly rather than after the better part of an hour. This is a *request*: the server
  returns the timeout it will actually honour, and may revise it upwards, so the effective value is
  ultimately the server's to decide

### Retry Logic
The block includes configurable retry logic with exponential backoff for handling transient connection failures:

- `max_retries`: Number of connection attempts before giving up (default: 3)
- `retry_backoff_seconds`: Base delay between retries in seconds (default: 1.0). The delay doubles
  after each failed attempt (exponential backoff).

**Note:** Authentication errors (wrong username/password) are not retried as they will continue to fail.

**Note:** If the server refuses a new session because it has reached its session or connection limit,
the block does not retry - a slot only frees up when another session times out. It stops attempting to
connect to that server for roughly 30 seconds (jittered, override with the
`OPC_SESSION_EXHAUSTION_BACKOFF_SECONDS` environment variable) so that many clients failing at once do
not keep hammering a server that is already full.

### Cooldown Limitations
!!! warning "Cooldown Limitations"
    The cooldown feature is optimized for workflows involving video processing.
    - In other contexts, such as Workflows triggered by HTTP services (e.g., Roboflow Hosted API,
      Dedicated Deployment, or self-hosted `Inference` server), the cooldown timer will not be applied effectively.
"""

QUERY_PARAMS_KIND = [
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_API_KEY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
    BOOLEAN_KIND,
    TOP_CLASS_KIND,
]
HEADER_KIND = [
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_API_KEY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    BOOLEAN_KIND,
    TOP_CLASS_KIND,
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OPC UA Writer Sink",
            "version": "v1",
            "short_description": "Writes data to an OPC UA server using the [asyncua](https://github.com/FreeOpcUa/opcua-asyncio) library for communication.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-industry",
                "blockPriority": 11,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal[BLOCK_TYPE]
    url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="URL of the OPC UA server to which data will be written.",
        examples=["opc.tcp://localhost:4840/freeopcua/server/", "$inputs.opc_url"],
    )
    namespace: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The OPC UA namespace URI or index used to locate objects and variables.",
        examples=["http://examples.freeopcua.github.io", "2", "$inputs.opc_namespace"],
    )
    user_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional username for authentication when connecting to the OPC UA server.",
        examples=["John", "$inputs.opc_user_name"],
    )
    password: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional password for authentication when connecting to the OPC UA server.",
        examples=["secret", "$inputs.opc_password"],
    )
    object_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The name of the target object in the namespace to search for.",
        examples=["Line1", "$inputs.opc_object_name"],
    )
    variable_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The name of the variable within the target object to be updated.",
        examples=[
            "InspectionSuccess",
            "$inputs.opc_variable_name",
        ],
    )
    value: Union[
        Selector(kind=[BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND, STRING_KIND]),
        str,
        bool,
        float,
        int,
    ] = Field(
        description="The value to be written to the target variable on the OPC UA server.",
        examples=["running", "$other_block.result"],
    )
    value_type: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "Boolean",
            "Double",
            "Float",
            "Int16",
            "Int32",
            "Int64",
            "Integer",
            "SByte",
            "String",
            "UInt16",
            "UInt32",
            "UInt64",
        ],
    ] = Field(
        default="String",
        description="The type of the value to be written to the target variable on the OPC UA server. "
        "Supported types: Boolean, Double, Float, Int16, Int32, Int64, Integer (Int64 alias), SByte, String, UInt16, UInt32, UInt64.",
        examples=["Boolean", "Double", "Float", "Int32", "Int64", "String"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="The number of seconds to wait for a response from the OPC UA server before timing out.",
        examples=[10, "$inputs.timeout"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
        examples=[True, "$inputs.fire_and_forget"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable block execution.",
        examples=[False, "$inputs.disable_opc_writers"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="The minimum number of seconds to wait between consecutive updates to the OPC UA server.",
        json_schema_extra={
            "always_visible": True,
        },
        examples=[10, "$inputs.cooldown_seconds"],
    )
    node_lookup_mode: Union[
        Selector(kind=[STRING_KIND]),
        Literal["hierarchical", "direct"],
    ] = Field(
        default="hierarchical",
        description="Method to locate the OPC UA node: 'hierarchical' uses path navigation, "
        "'direct' uses NodeId strings (for Ignition-style string-based tags).",
        examples=["hierarchical", "direct"],
    )
    max_retries: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=3,
        description="Maximum number of connection attempts before giving up. "
        "Default is 3 with exponential backoff starting at 15ms.",
        examples=[1, 3, "$inputs.max_retries"],
        ge=1,
    )
    retry_backoff_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.015,
        description="Base delay between retry attempts in seconds (doubles each retry). "
        "Default is 0.015 (15ms) for fast exponential backoff.",
        examples=[0.015, 0.5, 1.0, "$inputs.retry_backoff"],
        ge=0.0,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="disabled", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OPCWriterSinkBlockV1(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        disable_sinks: bool = False,
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._disable_sinks = disable_sinks
        self._last_notification_fired: Optional[datetime] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor", "disable_sinks"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        url: str,
        namespace: str,
        user_name: Optional[str],
        password: Optional[str],
        object_name: str,
        variable_name: str,
        value: Union[str, bool, float, int],
        value_type: Literal[
            "Boolean",
            "Double",
            "Float",
            "Int16",
            "Int32",
            "Int64",
            "Integer",
            "SByte",
            "String",
            "UInt16",
            "UInt32",
            "UInt64",
        ] = "String",
        timeout: int = 2,
        fire_and_forget: bool = True,
        disable_sink: bool = False,
        cooldown_seconds: int = 5,
        node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.015,
    ) -> BlockResult:
        if self._disable_sinks or disable_sink:
            message = disabled_sink_message(
                disabled_by_execution_policy=self._disable_sinks
            )
            logger.debug(message)
            return {
                "disabled": True,
                "throttling_status": False,
                "error_status": False,
                "message": message,
            }
        seconds_since_last_notification = cooldown_seconds
        if self._last_notification_fired is not None:
            seconds_since_last_notification = (
                datetime.now() - self._last_notification_fired
            ).total_seconds()
        if seconds_since_last_notification < cooldown_seconds:
            logger.info(f"Activated `{BLOCK_TYPE}` cooldown.")
            return {
                "disabled": False,
                "throttling_status": True,
                "error_status": False,
                "message": "Sink cooldown applies",
            }

        if value_type in [BOOLEAN_KIND, "Boolean"] and isinstance(value, str):
            # handle boolean conversion explicitly if value is a string
            decoded_value = value.strip().lower() in ("true", "1")
        else:
            # Use value directly - OPC UA library will convert based on type specification
            decoded_value = value

        logger.debug(
            f"OPC Writer prepared value '{decoded_value}' for type {value_type}"
        )

        opc_writer_handler = partial(
            opc_connect_and_write_value,
            url=url,
            namespace=namespace,
            user_name=user_name,
            password=password,
            object_name=object_name,
            variable_name=variable_name,
            value=decoded_value,
            value_type=value_type,
            timeout=timeout,
            node_lookup_mode=node_lookup_mode,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            logger.debug("OPC Writer submitting write task to background tasks")
            self._background_tasks.add_task(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to the OPC UA server in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            logger.debug("OPC Writer submitting write task to thread pool executor")
            self._thread_pool_executor.submit(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to the OPC UA server in the background task",
            }
        logger.debug("OPC Writer executing synchronous write")
        error_status, message = opc_writer_handler()
        logger.debug(
            f"OPC Writer write completed: error_status={error_status}, message={message}"
        )
        return {
            "disabled": False,
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def get_available_namespaces(client: Client) -> List[str]:
    """
    Get list of available namespaces from OPC UA server.
    Returns empty list if unable to fetch namespaces.
    """
    try:
        get_namespace_array = sync_async_client_method(AsyncClient.get_namespace_array)(
            client
        )
        return get_namespace_array()
    except Exception as exc:
        logger.info(f"Failed to get namespace array (non-fatal): {exc}")
        return ["<unable to fetch namespaces>"]


def safe_disconnect(client: Client) -> None:
    """Safely disconnect from OPC UA server, swallowing any errors"""
    try:
        logger.debug("OPC Writer disconnecting from server")
        client.disconnect()
    except Exception as exc:
        logger.debug(f"OPC Writer disconnect error (non-fatal): {exc}")


def get_node_data_type(var) -> str:
    """
    Get the data type of an OPC UA node.
    Returns a string representation of the type, or "Unknown" if unable to read.
    """
    try:
        return str(var.read_data_type_as_variant_type())
    except Exception as exc:
        logger.info(f"Unable to read node data type: {exc}")
        return "Unknown"


def opc_connect_and_write_value(
    url: str,
    namespace: str,
    user_name: Optional[str],
    password: Optional[str],
    object_name: str,
    variable_name: str,
    value: Union[bool, float, int, str],
    timeout: int,
    node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
    value_type: str = "String",
    max_retries: int = 1,
    retry_backoff_seconds: float = 0.0,
) -> Tuple[bool, str]:
    """
    Connect to OPC UA server and write a value using connection pooling.

    Uses the connection manager to reuse existing connections. If no connection
    exists, attempts to create one. Fails fast on connection errors to avoid
    blocking the pipeline.

    Args:
        url: OPC UA server URL
        namespace: Namespace URI or index
        user_name: Optional username for authentication
        password: Optional password for authentication
        object_name: Target object path
        variable_name: Variable to write
        value: Value to write
        timeout: Connection timeout in seconds
        node_lookup_mode: Path lookup strategy ('hierarchical' or 'direct')
        value_type: OPC UA data type for the value
        max_retries: Maximum number of connection attempts (default 1 = no retries)
        retry_backoff_seconds: Base delay between retries (default 0 = no delay)

    Returns:
        Tuple of (error_status, message)
    """
    logger.debug(
        f"OPC Writer attempting to write value={value} to {url}/{object_name}/{variable_name}"
    )

    connection_manager = get_connection_manager()

    try:
        # Get connection from pool (will create new if needed)
        client = connection_manager.get_connection(
            url=url,
            user_name=user_name,
            password=password,
            timeout=timeout,
            max_retries=max_retries,
            base_backoff=retry_backoff_seconds,
        )

        # Perform the write operation
        _opc_write_value(
            client=client,
            namespace=namespace,
            object_name=object_name,
            variable_name=variable_name,
            value=value,
            node_lookup_mode=node_lookup_mode,
            value_type=value_type,
        )

        logger.debug(
            f"OPC Writer successfully wrote value to {url}/{object_name}/{variable_name}"
        )
        return False, "Value set successfully"

    except Exception as exc:
        is_user_config_error = isinstance(exc, USER_CONFIG_ERROR_TYPES)

        # Check the exception chain for wrapped errors
        if not is_user_config_error and hasattr(exc, "__cause__") and exc.__cause__:
            is_user_config_error = isinstance(exc.__cause__, USER_CONFIG_ERROR_TYPES)

        if not is_user_config_error:
            logger.warning(
                f"OPC Writer error (invalidating connection): {type(exc).__name__}: {exc}"
            )
            connection_manager.invalidate_connection(url, user_name)
        else:
            # User configuration errors - connection is fine, just log the error
            logger.error(f"OPC Writer configuration error: {type(exc).__name__}: {exc}")

        return (
            True,
            f"Failed to write {value} to {object_name}:{variable_name} in {url}. Error: {exc}",
        )


def _opc_write_value(
    client: Client,
    namespace: str,
    object_name: str,
    variable_name: str,
    value: Union[bool, float, int, str],
    node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
    value_type: str = "String",
) -> None:
    """
    Write a value to an OPC UA variable using an existing connection.

    This is the core write logic, separated from connection management.
    Raises exceptions on failure which the caller should handle.

    Args:
        client: Connected OPC UA client
        namespace: Namespace URI or index
        object_name: Target object path
        variable_name: Variable to write
        value: Value to write
        node_lookup_mode: Path lookup strategy
        value_type: OPC UA data type for the value

    Raises:
        Exception: On any error during the write operation
    """
    get_namespace_index = sync_async_client_method(AsyncClient.get_namespace_index)(
        client
    )

    # Resolve namespace
    try:
        if namespace.isdigit():
            nsidx = int(namespace)
            logger.debug(f"OPC Writer using numeric namespace index: {nsidx}")
        else:
            nsidx = get_namespace_index(namespace)
    except ValueError as exc:
        namespaces = get_available_namespaces(client)
        logger.error(f"OPC Writer invalid namespace: {exc}")
        logger.error(f"Available namespaces: {namespaces}")
        raise Exception(
            f"WRONG NAMESPACE ERROR: {exc}. Available namespaces: {namespaces}"
        ) from exc
    except Exception as exc:
        namespaces = get_available_namespaces(client)
        logger.error(f"OPC Writer unhandled namespace error: {type(exc)} {exc}")
        logger.error(f"Available namespaces: {namespaces}")
        raise Exception(
            f"UNHANDLED ERROR: {type(exc)} {exc}. Available namespaces: {namespaces}"
        ) from exc

    # Locate the node
    if node_lookup_mode == "direct":
        # Direct NodeId access for string identifiers
        # If variable_name is empty, use object_name as the full identifier
        # This allows maximum flexibility for different server naming conventions
        try:
            if variable_name:
                node_id = f"ns={nsidx};s={object_name}/{variable_name}"
            else:
                node_id = f"ns={nsidx};s={object_name}"
            logger.debug(f"OPC Writer using direct NodeId access: {node_id}")
            var = client.get_node(node_id)
            logger.debug(
                f"OPC Writer successfully found variable node using direct NodeId"
            )
        except Exception as exc:
            logger.error(f"OPC Writer direct NodeId access failed: {exc}")
            raise Exception(
                f"WRONG OBJECT OR PROPERTY ERROR: Could not find node with direct NodeId '{node_id}'. Error: {exc}"
            ) from exc
    else:
        # Hierarchical path navigation (standard OPC UA)
        try:
            # Split object_name on "/" and prepend namespace index to each component
            object_components = object_name.split("/")
            object_path = "/".join([f"{nsidx}:{comp}" for comp in object_components])
            node_path = f"0:Objects/{object_path}/{nsidx}:{variable_name}"
            logger.debug(f"OPC Writer using hierarchical path: {node_path}")
            var = client.nodes.root.get_child(node_path)
            logger.debug(
                f"OPC Writer successfully found variable node using hierarchical path"
            )
        except BadNoMatch as exc:
            logger.error(f"OPC Writer hierarchical path not found: {exc}")
            raise Exception(
                f"WRONG OBJECT OR PROPERTY ERROR: Could not find node at hierarchical path '{node_path}'. Error: {exc}"
            ) from exc
        except Exception as exc:
            logger.error(f"OPC Writer unhandled node lookup error: {type(exc)} {exc}")
            raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}") from exc

    # Write the value
    try:
        logger.debug(
            f"OPC Writer writing value '{value}' to variable with type '{value_type}'"
        )
        # Convert to primitive types before setting value
        if value_type in [BOOLEAN_KIND, "Boolean"]:
            var.set_value(bool(value), VariantType.Boolean)
        elif value_type == "Double":
            var.set_value(float(value), VariantType.Double)
        elif value_type in [FLOAT_KIND, "Float"]:
            var.set_value(float(value), VariantType.Float)
        elif value_type == "Int16":
            int_val = int(value)
            if not (-32768 <= int_val <= 32767):
                raise ValueError(f"Value {int_val} out of range for Int16")
            var.set_value(int_val, VariantType.Int16)
        elif value_type == "Int32":
            int_val = int(value)
            if not (-2147483648 <= int_val <= 2147483647):
                raise ValueError(f"Value {int_val} out of range for Int32")
            var.set_value(int_val, VariantType.Int32)
        elif value_type in ["Int64", INTEGER_KIND, "Integer"]:
            int_val = int(value)
            if not (-9223372036854775808 <= int_val <= 9223372036854775807):
                raise ValueError(f"Value {int_val} out of range for Int64")
            var.set_value(int_val, VariantType.Int64)
        elif value_type == "SByte":
            int_val = int(value)
            if not (-128 <= int_val <= 127):
                raise ValueError(f"Value {int_val} out of range for SByte")
            var.set_value(int_val, VariantType.SByte)
        elif value_type in [STRING_KIND, "String"]:
            var.set_value(str(value), VariantType.String)
        elif value_type == "UInt16":
            int_val = int(value)
            if not (0 <= int_val <= 65535):
                raise ValueError(f"Value {int_val} out of range for UInt16")
            var.set_value(int_val, VariantType.UInt16)
        elif value_type == "UInt32":
            int_val = int(value)
            if not (0 <= int_val <= 4294967295):
                raise ValueError(f"Value {int_val} out of range for UInt32")
            var.set_value(int_val, VariantType.UInt32)
        elif value_type == "UInt64":
            int_val = int(value)
            if not (0 <= int_val <= 18446744073709551615):
                raise ValueError(f"Value {int_val} out of range for UInt64")
            var.set_value(int_val, VariantType.UInt64)
        else:
            logger.error(f"OPC Writer unsupported value type: {value_type}")
            raise UnsupportedTypeError(f"Value type '{value_type}' is not supported.")
        logger.info(
            f"OPC Writer successfully wrote '{value}' to variable at {object_name}/{variable_name}"
        )
    except UnsupportedTypeError:
        raise
    except BadTypeMismatch as exc:
        node_type = get_node_data_type(var)
        logger.error(
            f"OPC Writer type mismatch: tried to write value '{value}' (type: {type(value).__name__}) to node with data type {node_type}. Error: {exc}"
        )
        raise Exception(
            f"WRONG TYPE ERROR: Tried to write value '{value}' (type: {type(value).__name__}) but node expects type {node_type}. {exc}"
        ) from exc
    except Exception as exc:
        logger.error(f"OPC Writer unhandled write error: {type(exc)} {exc}")
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}") from exc
