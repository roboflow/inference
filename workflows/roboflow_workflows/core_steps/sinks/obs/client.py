"""Shared obs-websocket client handling for the OBS workflow blocks.

OBS lives on the machine driving the workflow and its websocket server drops
connections whenever OBS restarts, so clients are pooled per (host, port) and
every request is retried once against a freshly established connection.

Credentials are registered here by the OBS Connection block and looked up by
address, so the connection descriptor that travels between blocks - and can be
wired to a Workflow output - never carries the password.

The websocket client is not thread-safe (one socket, no locking around
send/receive), while a Workflow can drive the same OBS from several threads at
once: fire-and-forget actions on the thread pool, or two video pipelines in one
process. Every request therefore runs under a per-client lock.
"""

import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from roboflow_workflows.core_steps.sinks.obs.websocket_client import (
    OBSAuthenticationError,
    OBSRequestError,
    OBSWebSocketClient,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3

# OBS answered and refused (unknown scene, wrong password): reconnecting cannot
# change the outcome, so these fail immediately instead of costing a second round trip
NON_RETRYABLE_ERRORS = (OBSRequestError, OBSAuthenticationError)

ConnectionKey = Tuple[str, int]

_CLIENTS: Dict[ConnectionKey, Any] = {}
_CLIENT_LOCKS: Dict[ConnectionKey, threading.Lock] = {}
_CREDENTIALS: Dict[ConnectionKey, Tuple[Optional[str], int]] = {}
_REGISTRY_LOCK = threading.Lock()


def _connect(host: str, port: int, password: Optional[str], timeout: int) -> Any:
    client = OBSWebSocketClient(host, port, password=password, timeout=timeout)

    return client


def _close(client: Any) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001 - a dead socket must not break teardown
        pass


def register_connection(
    host: str, port: int, password: Optional[str], timeout: int
) -> None:
    """Record how to reach an OBS instance so later requests need only its address.

    Re-registering with a different password drops the pooled client, so the next
    request authenticates with the new credential instead of a stale socket.
    """
    key = (host, port)
    with _REGISTRY_LOCK:
        previous = _CREDENTIALS.get(key)
        _CREDENTIALS[key] = (password, timeout)
        if previous is not None and previous[0] != password and key in _CLIENTS:
            _close(_CLIENTS.pop(key))


def _resolve_credentials(
    key: ConnectionKey, password: Optional[str], timeout: Optional[int]
) -> Tuple[Optional[str], int]:
    registered_password, registered_timeout = _CREDENTIALS.get(key, (None, None))
    if password is None:
        password = registered_password
    if timeout is None:
        timeout = registered_timeout or DEFAULT_TIMEOUT
    return password, timeout


def get_client(
    host: str,
    port: int,
    password: Optional[str] = None,
    timeout: Optional[int] = None,
    force_reconnect: bool = False,
) -> Any:
    key = (host, port)
    with _REGISTRY_LOCK:
        if force_reconnect and key in _CLIENTS:
            _close(_CLIENTS.pop(key))
        client = _CLIENTS.get(key)
        if client is None:
            password, timeout = _resolve_credentials(key, password, timeout)
            client = _connect(host=host, port=port, password=password, timeout=timeout)
            _CLIENTS[key] = client
        lock = _CLIENT_LOCKS.setdefault(key, threading.Lock())
    return client, lock


def call_with_reconnect(
    host: str,
    port: int,
    operation: Callable[[Any], Any],
    password: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Any:
    """Run `operation(client)` under the client's lock, reconnecting once if the socket is dead."""
    try:
        client, lock = get_client(
            host=host, port=port, password=password, timeout=timeout
        )
        with lock:
            return operation(client)
    except NON_RETRYABLE_ERRORS:
        raise
    except (
        Exception
    ) as first_error:  # noqa: BLE001 - any socket fault is retryable once
        logger.warning(
            "OBS request failed (%s). Reconnecting to %s:%s and retrying once.",
            first_error,
            host,
            port,
        )
        client, lock = get_client(
            host=host,
            port=port,
            password=password,
            timeout=timeout,
            force_reconnect=True,
        )
        with lock:
            return operation(client)


def reset_clients() -> None:
    """Drop every pooled connection and credential. Used by tests and on teardown."""
    with _REGISTRY_LOCK:
        for client in _CLIENTS.values():
            _close(client)
        _CLIENTS.clear()
        _CLIENT_LOCKS.clear()
        _CREDENTIALS.clear()
