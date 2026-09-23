"""Shared pieces of the MQTT Reader and MQTT Writer blocks.

Both blocks open an outbound connection to the broker named by the workflow.
`resolve_broker_address()` applies the operator's policy from
`MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` and
`MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (see `roboflow_workflows.environment`)
before either block builds a client, and `configure_tls()` enables
server-verified TLS on the client, gating a workflow-chosen CA path behind the
engine's file-system permission. `normalise_client_id()` turns the reader's
optional `client_id` into either a usable id or "unset".
"""

import logging
import os
from typing import Any, List, Optional, Tuple

from roboflow_workflows.environment import (
    MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST,
    MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS,
)

logger = logging.getLogger("inference")

DEFAULT_MQTT_PORT = 1883
# paho defaults to 60 s; a silent link loss (no TCP close) is only noticed
# through the keepalive, within about 1.5 times this value
MQTT_KEEPALIVE_SECONDS = 15


class ConfigurationError(ValueError):
    """The operator's policy or the deployment's permissions forbid the connection."""


def configure_tls(
    client: Any,
    *,
    use_tls: bool,
    ca_certificate_path: Optional[str],
    allow_access_to_file_system: bool,
) -> None:
    """Enable server-verified TLS on a paho client before it connects.

    Args:
        client: The paho ``Client`` to configure; left untouched when ``use_tls``
            is False or when an error is raised.
        use_tls: Whether the connection must be wrapped in TLS.
        ca_certificate_path: Optional PEM bundle for a broker whose issuer is not in
            the system trust store. Ignored when ``use_tls`` is False.
        allow_access_to_file_system: The engine's file-system permission; a CA path
            is a server-side path chosen by the workflow and is refused without it.

    Raises:
        ConfigurationError: When the CA path is set but file-system access is
            disabled, or when the CA bundle cannot be loaded.
    """
    if not use_tls:
        return
    if ca_certificate_path and not allow_access_to_file_system:
        # the path is chosen by the workflow and read by the server process
        raise ConfigurationError(
            "ca_certificate_path needs access to the local file system, which is "
            "disabled on this deployment (ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False "
            "or the engine's allow_access_to_file_system init parameter)."
        )
    if ca_certificate_path and not os.path.isfile(ca_certificate_path):
        # OpenSSL's loader blocks forever on a FIFO and errors on a directory;
        # refuse anything that is not a regular file before it is opened
        raise ConfigurationError(
            f"TLS could not be configured: ca_certificate_path {ca_certificate_path!r} "
            "is not a readable file."
        )
    try:
        if ca_certificate_path:
            client.tls_set(ca_certs=str(ca_certificate_path))
        else:
            # system trust store; hostname verification stays on
            client.tls_set()
    except Exception as e:
        raise ConfigurationError(
            f"TLS could not be configured: could not load CA bundle "
            f"{ca_certificate_path!r} ({e})."
            if ca_certificate_path
            else f"TLS could not be configured: {e}."
        ) from e


def normalise_client_id(value: Any) -> Optional[str]:
    """Normalise the reader's optional ``client_id`` input.

    A persistent MQTT session exists only for a fixed client id, so the presence
    of the id is the switch. An empty or whitespace-only value is "unset", so a
    stray space never creates a persistent session.

    Args:
        value: The workflow-provided ``client_id``: ``None``, a literal or a
            selector-resolved value.

    Returns:
        The id with surrounding whitespace removed, or ``None`` when unset.

    Raises:
        ConfigurationError: When the value is not a string (a wiring error).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(
            f"Invalid client_id: {value!r}. Must be a string or left empty."
        )
    value = value.strip()
    return value or None


def _normalise_host(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def split_host_port(entry: str) -> Tuple[str, Optional[int]]:
    """Split an allowlist entry into a normalised host and an optional port.

    Accepts ``host``, ``host:port``, ``[ipv6]``, ``[ipv6]:port`` and a bare
    IPv6 address. There is no DNS resolution.

    Args:
        entry: One operator-written allowlist entry.

    Returns:
        ``(host, port)`` with the host lower-cased, whitespace trimmed and IPv6
        brackets stripped; ``port`` is ``None`` when the entry names none.
    """
    entry = entry.strip()
    if entry.startswith("["):
        closing = entry.find("]")
        if closing == -1:
            return _normalise_host(entry), None
        rest = entry[closing + 1 :]
        port = int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
        return _normalise_host(entry[: closing + 1]), port
    if entry.count(":") == 1:
        host, port_text = entry.split(":")
        if port_text.strip().isdigit():
            return _normalise_host(host), int(port_text)
        return _normalise_host(entry), None
    # no colon: a plain host; several colons: a bare IPv6 address
    return _normalise_host(entry), None


def normalise_broker_address(host: str, port: int) -> str:
    """Return the ``host:port`` form used when comparing addresses.

    Args:
        host: Broker host as written in the workflow.
        port: Broker port.

    Returns:
        ``host:port`` with the host lower-cased and IPv6 brackets stripped.
    """
    return f"{_normalise_host(str(host))}:{int(port)}"


def _operator_entries(allowlist: Optional[List[str]]) -> List[str]:
    if allowlist is None:
        return []
    return [str(entry).strip() for entry in allowlist if str(entry).strip()]


def resolve_broker_address(
    host: str, port: int, log_override: bool = True
) -> Tuple[str, int]:
    """Apply the operator's broker policy to a workflow-provided address.

    Args:
        host: Broker host taken from the workflow.
        port: Broker port taken from the workflow, already validated as an int.
        log_override: Emit the warning when the workflow value is replaced by
            the operator's broker; blocks pass ``False`` after their first run
            so the warning is logged once per instance instead of once per frame.

    Returns:
        The ``(host, port)`` the block must connect to: the workflow value when
        permitted, or the operator's first allowlist entry when workflow-provided
        hosts are not allowed.

    Raises:
        ConfigurationError: When the policy forbids the connection. The message
            never echoes the allowlist.
    """
    # module attributes are read on every call so tests can patch them
    user_provided_allowed = MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST
    allowlist = MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS
    operator_entries = _operator_entries(allowlist)
    if not user_provided_allowed:
        if not operator_entries:
            raise ConfigurationError(
                "MQTT blocks are disabled on this deployment: a workflow-provided "
                "broker host is not allowed and the operator configured none."
            )
        operator_host, operator_port = split_host_port(operator_entries[0])
        if operator_port is None:
            operator_port = DEFAULT_MQTT_PORT
        if log_override:
            logger.warning(
                "MQTT blocks: the workflow-provided host/port was replaced by the "
                "operator-configured broker (%s:%s), because "
                "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST is False.",
                operator_host,
                operator_port,
            )
        return operator_host, operator_port
    if allowlist is None:
        return host, port

    wanted_host = _normalise_host(str(host))
    wanted_port = int(port)
    for entry in operator_entries:
        entry_host, entry_port = split_host_port(entry)
        if entry_host == wanted_host and entry_port in (None, wanted_port):
            return host, port
    # the allowlist holds the operator's internal hostnames: never echo it
    raise ConfigurationError(
        f"Broker {host!r} port {wanted_port} is not permitted on this deployment: "
        "the operator restricts the MQTT brokers Workflow blocks may connect to."
    )
