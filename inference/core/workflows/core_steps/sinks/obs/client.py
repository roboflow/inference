"""Shared obs-websocket client handling for the OBS workflow blocks.

OBS lives on the machine driving the workflow and its websocket server drops
connections whenever OBS restarts, so clients are pooled per (host, port) and
every request is retried once against a freshly established connection.
"""

import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from inference.core import logger

OBS_CLIENT_IMPORT_ERROR = (
    "OBS blocks require the `obsws-python` package, which is not installed in "
    "the environment running `inference`. Install it with `pip install obsws-python`."
)

_CLIENTS: Dict[Tuple[str, int], Any] = {}
_CLIENTS_LOCK = threading.Lock()


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


def get_client(
    host: str,
    port: int,
    password: Optional[str],
    timeout: int = 3,
    force_reconnect: bool = False,
) -> Any:
    key = (host, port)
    with _CLIENTS_LOCK:
        if force_reconnect and key in _CLIENTS:
            _close(_CLIENTS.pop(key))
        client = _CLIENTS.get(key)
        if client is None:
            client = _connect(host=host, port=port, password=password, timeout=timeout)
            _CLIENTS[key] = client
        return client


def call_with_reconnect(
    host: str,
    port: int,
    password: Optional[str],
    timeout: int,
    operation: Callable[[Any], Any],
) -> Any:
    """Run `operation(client)`, reconnecting once if the pooled socket is dead."""
    request_error_type = _request_error_type()
    try:
        return operation(
            get_client(host=host, port=port, password=password, timeout=timeout)
        )
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
        client = get_client(
            host=host,
            port=port,
            password=password,
            timeout=timeout,
            force_reconnect=True,
        )
        return operation(client)


def reset_clients() -> None:
    """Drop every pooled connection. Used by tests and on interpreter teardown."""
    with _CLIENTS_LOCK:
        for client in _CLIENTS.values():
            _close(client)
        _CLIENTS.clear()
