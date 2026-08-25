"""Shared obs-websocket client handling for the OBS workflow blocks.

OBS lives on the machine driving the workflow and its websocket server drops
connections whenever OBS restarts, so clients are pooled per (host, port) and
every request is retried once against a freshly established connection.

Credentials are registered here by the OBS Connection block and looked up by
address, so the connection descriptor that travels between blocks - and can be
wired to a Workflow output - never carries the password.

obsws-python's request client is not thread-safe (one websocket, no locking
around send/receive), while a Workflow can drive the same OBS from several
threads at once: fire-and-forget actions on the thread pool, or two video
pipelines in one process. Every request therefore runs under a per-client lock.
"""

import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from inference.core import logger

OBS_CLIENT_IMPORT_ERROR = (
    "OBS blocks require the `obsws-python` package, which is not installed in "
    "the environment running `inference`. Install it with `pip install obsws-python`."
)
DEFAULT_TIMEOUT = 3

ConnectionKey = Tuple[str, int]

_CLIENTS: Dict[ConnectionKey, Any] = {}
_CLIENT_LOCKS: Dict[ConnectionKey, threading.Lock] = {}
_CREDENTIALS: Dict[ConnectionKey, Tuple[Optional[str], int]] = {}
_REGISTRY_LOCK = threading.Lock()


def _import_obsws() -> Any:
    try:
        import obsws_python
    except ImportError as error:
        raise ImportError(OBS_CLIENT_IMPORT_ERROR) from error
    return obsws_python


def _request_error_type() -> Optional[type]:
    """The obsws exception meaning "OBS rejected this request" rather than "socket died"."""
    try:
        from obsws_python.error import OBSSDKRequestError
    except ImportError:
        return None
    return OBSSDKRequestError


_PASSWORD_LOGGING_SUPPRESSED = False


def _suppress_obsws_credential_logging() -> None:
    """obsws-python logs the websocket password verbatim at INFO on every connect.

    Left alone, the OBS password ends up in `inference` server logs. Raise that one
    logger to WARNING so connection failures still surface but credentials do not.
    """
    global _PASSWORD_LOGGING_SUPPRESSED
    if _PASSWORD_LOGGING_SUPPRESSED:
        return
    logging.getLogger("obsws_python.baseclient").setLevel(logging.WARNING)
    _PASSWORD_LOGGING_SUPPRESSED = True


def _connect(host: str, port: int, password: Optional[str], timeout: int) -> Any:
    obsws_python = _import_obsws()
    _suppress_obsws_credential_logging()
    return obsws_python.ReqClient(
        host=host,
        port=port,
        password=password or "",
        timeout=timeout,
    )


def _close(client: Any) -> None:
    try:
        client.disconnect()
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
    request_error_type = _request_error_type()
    try:
        client, lock = get_client(
            host=host, port=port, password=password, timeout=timeout
        )
        with lock:
            return operation(client)
    except ImportError:
        raise
    except (
        Exception
    ) as first_error:  # noqa: BLE001 - any socket fault is retryable once
        if request_error_type is not None and isinstance(
            first_error, request_error_type
        ):
            # OBS answered and refused the request (unknown scene, missing source, ...).
            # Reconnecting cannot change that, so fail immediately instead of paying for
            # a second round trip on every malformed action.
            raise
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
