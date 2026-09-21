"""Shared pieces of the MQTT Reader and MQTT Writer blocks.

Both blocks open an outbound connection to the broker named by the workflow.
`resolve_broker_address()` applies the operator's policy from
`MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` and
`MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (see `roboflow_workflows.environment`)
before either block builds a client.
"""

import logging
from typing import List, Optional, Tuple

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
    """The operator's policy forbids the requested broker connection."""


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
