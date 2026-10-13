"""PLC reader and writer blocks: read and write PLC tags.

Ports of V1 ``roboflow_core/plc_reader@v1`` and ``roboflow_core/plc_writer@v1``::

    step parameters ─▶ PlcTarget(mode, address, ports, slot, timeout)
                          │
    plc_client resource?  ├ given ─▶ the host's client; borrowed, never closed here
                          └ none  ─▶ PlcConnections() for this call only,
                                     closed in ``finally`` on success and failure
                          ▼
        relay        POST <address>/read_batch | /write_batch  (requests)
        ethernet_ip  pylogix.PLC Read / Write                  (lazy import)
        modbus       pymodbus ModbusTcpClient                  (lazy import)
                          ▼
        {tag: value or "ReadFailure"} / "WriteSuccess" | "WriteFailure",
        error_status = any tag failed

Failures keep V1's contract: a transport or PLC error is logged and turns the
affected tags into the failure sentinel; it does not fail the step. Writes are
never retried. A failed write reports ``WriteFailure`` even when the PLC may
have applied it (e.g. a read timeout after the relay sent it); the caller
decides whether to write again.

Connection lifetime: the engine never closes a block, so a block keeps no
connection between calls. Without a ``plc_client`` resource each call opens
and closes its own connection: one TCP setup per call, which V1 avoided by
keeping it on the block. A host wanting V1's reuse creates ``PlcConnections``,
passes it as ``plc_client`` and closes it after its runs stop. No speed claim
is made for either path.

The relay address may be a ``secret`` (e.g. ``http://user:password@host:8007``
from a secrets store step). Its plaintext exists only inside the transport;
log lines are scrubbed of it. A replayed (unavailable) secret fails the step
before any connection is made.

Differences from V1 are listed in ``PlcReaderBlock`` and ``PlcWriterBlock``.
"""

import importlib
import logging
import threading
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)
from roboflow_workflows.core_steps.common.workload_presets import (
    PLC_LAN_ACCESS_ACTUAL_RESTRICTION,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret, reveal_secrets
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

LOGGER = logging.getLogger(__name__)

READER_TYPE = "v2/plc_reader@v1"
WRITER_TYPE = "v2/plc_writer@v1"

DEFAULT_RELAY_PORT = 8007
# The connect phase to the on-device relay is capped tightly so a down relay
# fails fast; request_timeout is the read budget for the relay's PLC batch.
RELAY_CONNECT_TIMEOUT = 3

READ_FAILURE = "ReadFailure"
WRITE_FAILURE = "WriteFailure"
WRITE_SUCCESS = "WriteSuccess"

# Exact pins of the ``enterprise`` extra in workflows/pyproject.toml.
DRIVER_REQUIREMENTS = {
    "pylogix": "pylogix==1.0.5",
    "pymodbus": "pymodbus>=3.6.9,<=3.8.3",
}

MODBUS_REGISTER_AREAS = {"holding", "input"}
MODBUS_BIT_AREAS = {"coil", "discrete"}
MODBUS_READONLY_AREAS = {"input", "discrete"}
MODBUS_AREAS = MODBUS_REGISTER_AREAS | MODBUS_BIT_AREAS

ConnectionMode = Literal["relay", "ethernet_ip", "modbus"]
TagValues = Dict[Any, Any]
TagResult = Tuple[TagValues, bool]
Result = TypeVar("Result")

PLC_METADATA = {
    "section": "industrial",
    "license": "Roboflow Enterprise License",
    "enterprise_only": True,
    "local_only": True,
}


class PlcDriverUnavailableError(RuntimeError):
    """A direct connection mode needs an optional driver that is not installed."""


@dataclass(frozen=True)
class PlcTarget:
    """Where and how one call reaches the PLC.

    Constructing a target with an unavailable secret address raises, so a
    replayed secret fails before any connection exists.

    Args:
        connection_mode: ``relay``, ``ethernet_ip`` or ``modbus``.
        address: Relay host or URL (relay) or PLC IP address (direct modes);
            a ``Secret`` when it holds credentials.
        relay_port: Relay port, used when ``address`` is not a URL.
        request_timeout: Relay read timeout in seconds.
        processor_slot: EtherNet/IP processor slot.
        modbus_port: Modbus TCP port.
        modbus_unit_id: Modbus unit (slave) id.

    Raises:
        SecretUnavailableError: When ``address`` is an unavailable secret.
    """

    connection_mode: ConnectionMode
    address: Union[str, Secret]
    relay_port: int = DEFAULT_RELAY_PORT
    request_timeout: int = 10
    processor_slot: int = 0
    modbus_port: int = 502
    modbus_unit_id: int = 1

    def __post_init__(self) -> None:
        reveal_secrets(self.address)

    def reveal_address(self) -> str:
        """Return the plaintext address; call only where it is used.

        Returns:
            The address.
        """
        address = reveal_secrets(self.address).value

        return address

    def scrub(self, text: str) -> str:
        """Replace the plaintext of a secret address in ``text`` by its marker.

        Args:
            text: A message, e.g. a library error.

        Returns:
            ``text`` without the secret address; unchanged for a plain address.
        """
        scrubbed = reveal_secrets(self.address).scrub(text)

        return scrubbed


class PlcClient(Protocol):
    """Reads and writes PLC tags for a target; ``PlcConnections`` is one.

    A host-provided client is borrowed: blocks never close it. Calls from
    several steps, pulses and sessions may arrive at once.
    """

    def read_tags(self, target: PlcTarget, tags: List[Any]) -> TagResult:
        """Return ``({tag: value or READ_FAILURE}, any_failed)``."""

    def write_tags(
        self, target: PlcTarget, tags_to_write: Mapping[Any, Any]
    ) -> TagResult:
        """Return ``({tag: WRITE_SUCCESS or WRITE_FAILURE}, any_failed)``."""


class PlcConnections:
    """PLC connections kept across calls until ``close``; a ``PlcClient``.

    Keeps one relay HTTP session, one EtherNet/IP client and one Modbus
    client, as V1 kept them per block. A direct client is replaced when the
    target address, port or slot changes. One lock serializes calls, so a
    shared instance does one PLC operation at a time; give steps their own
    instances for parallel I/O. Whoever creates an instance closes it, after
    the runs using it stopped; later calls raise ``RuntimeError``.

    Args:
        make_session: Builds the relay session; default ``requests.Session``.
        make_eip_client: Builds an EtherNet/IP client from ``(address,
            processor_slot)``; default ``pylogix.PLC``.
        make_modbus_client: Builds a Modbus client from ``(address, port)``;
            default ``pymodbus.client.ModbusTcpClient``.
    """

    def __init__(
        self,
        *,
        make_session: Optional[Callable[[], Any]] = None,
        make_eip_client: Optional[Callable[[str, int], Any]] = None,
        make_modbus_client: Optional[Callable[[str, int], Any]] = None,
    ) -> None:
        self._make_session = make_session or _new_session
        self._make_eip_client = make_eip_client or _new_eip_client
        self._make_modbus_client = make_modbus_client or _new_modbus_client
        self._lock = threading.Lock()
        self._closed = False
        self._session: Any = None
        self._eip: Any = None
        self._eip_key: Optional[Tuple[str, int]] = None
        self._modbus: Any = None
        self._modbus_key: Optional[Tuple[str, int]] = None

    def read_tags(self, target: PlcTarget, tags: List[Any]) -> TagResult:
        """Read ``tags`` from ``target``.

        Args:
            target: The PLC to read.
            tags: Tag names, or Modbus ``area:address`` strings.

        Returns:
            ``({tag: value or READ_FAILURE}, any_failed)``; ``({}, False)``
            without connecting when ``tags`` is empty.

        Raises:
            PlcDriverUnavailableError: When a direct mode's driver is missing.
            RuntimeError: After ``close``.
        """
        tags = list(tags)
        if not tags:
            return {}, False

        with self._lock:
            self._check_open()
            if target.connection_mode == "ethernet_ip":
                result = ethernet_read_tags(self._eip_client(target), target, tags)
            elif target.connection_mode == "modbus":
                result = modbus_read_tags(self._modbus_client(target), target, tags)
            else:
                result = relay_read_tags(self._relay_session(), target, tags)

        return result

    def write_tags(
        self, target: PlcTarget, tags_to_write: Mapping[Any, Any]
    ) -> TagResult:
        """Write ``tags_to_write`` to ``target`` once, without retrying.

        Args:
            target: The PLC to write.
            tags_to_write: Values by tag.

        Returns:
            ``({tag: WRITE_SUCCESS or WRITE_FAILURE}, any_failed)``;
            ``({}, False)`` without connecting when there is nothing to write.

        Raises:
            PlcDriverUnavailableError: When a direct mode's driver is missing.
            RuntimeError: After ``close``.
        """
        if not tags_to_write:
            return {}, False

        with self._lock:
            self._check_open()
            if target.connection_mode == "ethernet_ip":
                client = self._eip_client(target)
                result = ethernet_write_tags(client, target, tags_to_write)
            elif target.connection_mode == "modbus":
                client = self._modbus_client(target)
                result = modbus_write_tags(client, target, tags_to_write)
            else:
                result = relay_write_tags(self._relay_session(), target, tags_to_write)

        return result

    def close(self) -> None:
        """Close every connection this instance made; waits for a running call.

        Close errors are ignored, as in V1. Calling it again does nothing.
        """
        with self._lock:
            self._closed = True
            _close_quietly(self._session, "close")
            _close_quietly(self._eip, "Close")
            _close_quietly(self._modbus, "close")
            self._session = self._eip = self._modbus = None
            self._eip_key = self._modbus_key = None

    def __enter__(self) -> "PlcConnections":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("PlcConnections is closed")

    def _relay_session(self) -> Any:
        if self._session is None:
            self._session = self._make_session()

        return self._session

    def _eip_client(self, target: PlcTarget) -> Any:
        key = (target.reveal_address(), target.processor_slot)
        if self._eip is not None and self._eip_key != key:
            _close_quietly(self._eip, "Close")
            self._eip = None
        if self._eip is None:
            self._eip = self._make_eip_client(*key)
            self._eip_key = key

        return self._eip

    def _modbus_client(self, target: PlcTarget) -> Any:
        key = (target.reveal_address(), target.modbus_port)
        if self._modbus is not None and self._modbus_key != key:
            _close_quietly(self._modbus, "close")
            self._modbus = None
        if self._modbus is None:
            self._modbus = self._make_modbus_client(*key)
            self._modbus_key = key

        return self._modbus


def with_plc_client(
    plc_client: Optional[PlcClient], operation: Callable[[PlcClient], Result]
) -> Result:
    """Run ``operation`` with the borrowed client, or a per-call one.

    Args:
        plc_client: The host's client, used as is and never closed; ``None``
            opens ``PlcConnections`` for this call only.
        operation: The read or write.

    Returns:
        The operation's result.
    """
    if plc_client is not None:
        borrowed_result = operation(plc_client)
        return borrowed_result

    connections = PlcConnections()
    try:
        result = operation(connections)
    finally:
        connections.close()

    return result


# ------------------------------ relay (HTTP) ---------------------------------


def relay_base_url(address: str, relay_port: int) -> str:
    """Build the relay base URL from a host or a full URL.

    Args:
        address: ``192.168.1.10`` or ``http://host:8007``.
        relay_port: Port added to a bare host.

    Returns:
        ``http://<host>:<port>``, or the URL without a trailing ``/``.
    """
    address = str(address).strip()
    if address.startswith("http://") or address.startswith("https://"):
        url = address.rstrip("/")
        return url

    url = f"http://{address}:{relay_port}"

    return url


def relay_read_tags(session: Any, target: PlcTarget, tags: List[Any]) -> TagResult:
    """Read a batch of tags with one ``POST /read_batch`` to the relay.

    A whole-batch rejection (non-200), an unreachable relay, a non-positive
    timeout or a malformed body fails every tag; a per-tag ``error`` fails
    that tag.

    Args:
        session: ``requests.Session``-like object; not closed here.
        target: The relay target.
        tags: Tags to read.

    Returns:
        ``({tag: value or READ_FAILURE}, any_failed)``.
    """
    response = _relay_post(session, target, "read_batch", {"tags": tags}, tags)
    if response is None:
        return _all_failed(tags, READ_FAILURE), True

    data = _response_object(response, target, tags)
    if data is None:
        return _all_failed(tags, READ_FAILURE), True

    by_name = _index_by_name(data.get("tags"))
    results: TagValues = {}
    had_failure = False
    for tag in tags:
        entry = by_name.get(tag)
        if entry is None:
            LOGGER.error("PLC Relay did not return a value for tag '%s'", tag)
            results[tag] = READ_FAILURE
            had_failure = True
        elif entry.get("error"):
            error = target.scrub(str(entry["error"]))
            LOGGER.error("Error reading tag '%s' from PLC: %s", tag, error)
            results[tag] = READ_FAILURE
            had_failure = True
        else:
            results[tag] = entry.get("value")

    return results, had_failure


def relay_write_tags(
    session: Any, target: PlcTarget, tags_to_write: Mapping[Any, Any]
) -> TagResult:
    """Write a batch of tags with one ``POST /write_batch`` to the relay.

    Failure rules are those of ``relay_read_tags``; a per-tag
    ``success: false`` fails that tag.

    Args:
        session: ``requests.Session``-like object; not closed here.
        target: The relay target.
        tags_to_write: Values by tag.

    Returns:
        ``({tag: WRITE_SUCCESS or WRITE_FAILURE}, any_failed)``.

    Raises:
        TypeError: When a value is not JSON data, as in V1.
    """
    names = list(tags_to_write)
    writes = [{"name": name, "value": value} for name, value in tags_to_write.items()]
    response = _relay_post(session, target, "write_batch", {"writes": writes}, names)
    if response is None:
        return _all_failed(names, WRITE_FAILURE), True

    data = _response_object(response, target, names)
    if data is None:
        return _all_failed(names, WRITE_FAILURE), True

    by_name = _index_by_name(data.get("results"))
    results: TagValues = {}
    had_failure = False
    for name, value in tags_to_write.items():
        entry = by_name.get(name)
        if entry is None:
            LOGGER.error("PLC Relay did not return a result for tag '%s'", name)
            results[name] = WRITE_FAILURE
            had_failure = True
        elif not entry.get("success", False):
            error = target.scrub(str(entry.get("error", WRITE_FAILURE)))
            LOGGER.error(
                "Error writing tag '%s' with value '%s': %s", name, value, error
            )
            results[name] = WRITE_FAILURE
            had_failure = True
        else:
            results[name] = WRITE_SUCCESS

    return results, had_failure


def _relay_post(
    session: Any, target: PlcTarget, endpoint: str, body: dict, tags: List[Any]
) -> Any:
    """POST ``body``; ``None`` (logged) when the request itself failed."""
    import requests

    url = f"{relay_base_url(target.reveal_address(), target.relay_port)}/{endpoint}"
    try:
        response = session.post(
            url,
            json=body,
            timeout=(RELAY_CONNECT_TIMEOUT, target.request_timeout),
        )
    except (requests.exceptions.RequestException, ValueError) as error:
        # ValueError: requests rejects a non-positive timeout before sending.
        LOGGER.error(
            "Failed to reach PLC Relay (%s) for tags %s: %s",
            endpoint,
            tags,
            target.scrub(str(error)),
        )
        return None

    return response


def _response_object(response: Any, target: PlcTarget, tags: List[Any]) -> Any:
    """The JSON object of a 200 response; ``None`` (logged) otherwise."""
    if response.status_code != 200:
        LOGGER.error(
            "PLC Relay rejected the batch for tags %s: HTTP %s: %s",
            tags,
            response.status_code,
            target.scrub(_error_detail(response)),
        )
        return None

    data = _json_object(response)
    if data is None:
        LOGGER.error(
            "Malformed PLC Relay success response for tags %s: %s",
            tags,
            target.scrub(response.text),
        )

    return data


def _json_object(response: Any) -> Optional[dict]:
    try:
        data = response.json()
    except ValueError:
        return None

    return data if isinstance(data, dict) else None


def _error_detail(response: Any) -> str:
    data = _json_object(response)
    if data is not None and "detail" in data:
        return str(data["detail"])

    return response.text


def _index_by_name(entries: Any) -> Dict[Any, dict]:
    if not isinstance(entries, list):
        return {}

    indexed = {
        entry["name"]: entry
        for entry in entries
        if isinstance(entry, dict) and "name" in entry
    }

    return indexed


def _all_failed(tags: Iterable[Any], sentinel: str) -> TagValues:
    return {tag: sentinel for tag in tags}


# --------------------------- EtherNet/IP (pylogix) ---------------------------


def ethernet_read_tags(comm: Any, target: PlcTarget, tags: List[Any]) -> TagResult:
    """Read tags one by one over EtherNet/IP; a failing tag does not stop others.

    Args:
        comm: ``pylogix.PLC``-like client; not closed here.
        target: The PLC, for scrubbing messages.
        tags: Tag names.

    Returns:
        ``({tag: value or READ_FAILURE}, any_failed)``.
    """
    results: TagValues = {}
    had_failure = False
    for tag in tags:
        value, ok = _eip_read_one(comm, target, tag)
        results[tag] = value
        had_failure = had_failure or not ok

    return results, had_failure


def ethernet_write_tags(
    comm: Any, target: PlcTarget, tags_to_write: Mapping[Any, Any]
) -> TagResult:
    """Write tags one by one over EtherNet/IP; strings suit Logix STRING tags.

    Args:
        comm: ``pylogix.PLC``-like client; not closed here.
        target: The PLC, for scrubbing messages.
        tags_to_write: Values by tag name.

    Returns:
        ``({tag: WRITE_SUCCESS or WRITE_FAILURE}, any_failed)``.
    """
    results: TagValues = {}
    had_failure = False
    for tag, value in tags_to_write.items():
        status, ok = _eip_write_one(comm, target, tag, value)
        results[tag] = status
        had_failure = had_failure or not ok

    return results, had_failure


def _eip_read_one(comm: Any, target: PlcTarget, tag: Any) -> Tuple[Any, bool]:
    try:
        response = comm.Read(tag)
        if response.Status == "Success":
            return response.Value, True

        LOGGER.error("Error reading EtherNet/IP tag '%s': %s", tag, response.Status)
        return READ_FAILURE, False
    except Exception as error:
        LOGGER.error(
            "Unhandled error reading EtherNet/IP tag '%s': %s",
            tag,
            target.scrub(str(error)),
        )
        return READ_FAILURE, False


def _eip_write_one(
    comm: Any, target: PlcTarget, tag: Any, value: Any
) -> Tuple[str, bool]:
    try:
        response = comm.Write(tag, value)
        if response.Status == "Success":
            return WRITE_SUCCESS, True

        LOGGER.error(
            "Error writing EtherNet/IP tag '%s' with value '%s': %s",
            tag,
            value,
            response.Status,
        )
        return WRITE_FAILURE, False
    except Exception as error:
        LOGGER.error(
            "Unhandled error writing EtherNet/IP tag '%s': %s",
            tag,
            target.scrub(str(error)),
        )
        return WRITE_FAILURE, False


# ------------------------------ Modbus (pymodbus) ----------------------------


def modbus_read_tags(client: Any, target: PlcTarget, tags: List[Any]) -> TagResult:
    """Read ``area:address`` tags over Modbus TCP, connecting first if needed.

    Registers read as ``int``, coils and discrete inputs as ``bool``. A bare
    number is a holding register.

    Args:
        client: ``ModbusTcpClient``-like client; not closed here.
        target: The PLC; ``modbus_unit_id`` is the slave id.
        tags: Tags such as ``holding:100`` or ``coil:0``.

    Returns:
        ``({tag: value or READ_FAILURE}, any_failed)``.
    """
    if not _modbus_connected(client):
        return _all_failed(tags, READ_FAILURE), bool(tags)

    results: TagValues = {}
    had_failure = False
    for tag in tags:
        value, ok = _modbus_read_one(client, target, tag)
        results[tag] = value
        had_failure = had_failure or not ok

    return results, had_failure


def modbus_write_tags(
    client: Any, target: PlcTarget, tags_to_write: Mapping[Any, Any]
) -> TagResult:
    """Write holding registers (0..65535) or coils (bool or 0/1) over Modbus TCP.

    Read-only areas, invalid tags and values that do not fit are rejected per
    tag before sending; nothing is coerced silently.

    Args:
        client: ``ModbusTcpClient``-like client; not closed here.
        target: The PLC; ``modbus_unit_id`` is the slave id.
        tags_to_write: Values by tag.

    Returns:
        ``({tag: WRITE_SUCCESS or WRITE_FAILURE}, any_failed)``.
    """
    if not _modbus_connected(client):
        return _all_failed(tags_to_write, WRITE_FAILURE), bool(tags_to_write)

    results: TagValues = {}
    had_failure = False
    for tag, value in tags_to_write.items():
        status, ok = _modbus_write_one(client, target, tag, value)
        results[tag] = status
        had_failure = had_failure or not ok

    return results, had_failure


def parse_modbus_tag(tag: Any) -> Tuple[str, int]:
    """Parse ``area:address`` or a bare address (a holding register).

    Args:
        tag: E.g. ``"holding:100"``, ``"coil:0"`` or ``"100"``.

    Returns:
        ``(area, address)``.

    Raises:
        ValueError: On an unknown area or a non-integer address.
    """
    text = str(tag).strip()
    if ":" in text:
        area, _, address = text.partition(":")
        area = area.strip().lower()
    else:
        area, address = "holding", text
    if area not in MODBUS_AREAS:
        raise ValueError(
            f"unknown Modbus area '{area}' (expected one of {sorted(MODBUS_AREAS)})"
        )

    parsed = (area, int(address.strip()))

    return parsed


def coerce_coil_value(value: Any) -> bool:
    """Accept a bool or 0/1 for a coil; reject anything else.

    Args:
        value: The value to write.

    Returns:
        The bit.

    Raises:
        ValueError: For any other value, e.g. ``"False"`` or ``2``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    raise ValueError(f"coil value must be a boolean or 0/1, got {value!r}")


def coerce_register_value(value: Any) -> int:
    """Accept an integral number in 0..65535 for a holding register.

    Args:
        value: The value to write.

    Returns:
        The register word.

    Raises:
        ValueError: For a bool, a string, a fractional float or out of range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"register value must be an integer in 0..65535, got {value!r}"
        )
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"register value must be an integer in 0..65535, got {value!r}"
        )

    word = int(value)
    if not 0 <= word <= 0xFFFF:
        raise ValueError(
            f"register value {word} out of range for a 16-bit register (0..65535)"
        )

    return word


def _modbus_connected(client: Any) -> bool:
    # As in V1, an exception from connect() is not caught: it fails the step.
    connected = bool(client.connected) or bool(client.connect())
    if not connected:
        LOGGER.error("Failed to connect to Modbus PLC")

    return connected


def _modbus_read_one(client: Any, target: PlcTarget, tag: Any) -> Tuple[Any, bool]:
    try:
        area, address = parse_modbus_tag(tag)
    except (ValueError, TypeError) as error:
        LOGGER.error("Invalid Modbus tag '%s': %s", tag, error)
        return READ_FAILURE, False

    unit_id = target.modbus_unit_id
    readers = {
        "holding": client.read_holding_registers,
        "input": client.read_input_registers,
        "coil": client.read_coils,
        "discrete": client.read_discrete_inputs,
    }
    try:
        response = readers[area](address, count=1, slave=unit_id)
        if response.isError():
            LOGGER.error("Error reading Modbus tag '%s': %s", tag, response)
            return READ_FAILURE, False

        if area in MODBUS_REGISTER_AREAS:
            return response.registers[0], True

        return bool(response.bits[0]), True
    except Exception as error:
        LOGGER.error(
            "Unhandled error reading Modbus tag '%s': %s", tag, target.scrub(str(error))
        )
        return READ_FAILURE, False


def _modbus_write_one(
    client: Any, target: PlcTarget, tag: Any, value: Any
) -> Tuple[str, bool]:
    try:
        area, address = parse_modbus_tag(tag)
    except (ValueError, TypeError) as error:
        LOGGER.error("Invalid Modbus tag '%s': %s", tag, error)
        return WRITE_FAILURE, False

    if area in MODBUS_READONLY_AREAS:
        LOGGER.error("Modbus tag '%s' is read-only (area '%s')", tag, area)
        return WRITE_FAILURE, False

    try:
        if area == "coil":
            payload: Any = coerce_coil_value(value)
        else:
            payload = coerce_register_value(value)
    except (ValueError, TypeError) as error:
        LOGGER.error("Invalid value for Modbus tag '%s': %s", tag, error)
        return WRITE_FAILURE, False

    unit_id = target.modbus_unit_id
    try:
        if area == "coil":
            response = client.write_coil(address, payload, slave=unit_id)
        else:
            response = client.write_register(address, payload, slave=unit_id)
        if response.isError():
            LOGGER.error(
                "Error writing Modbus tag '%s' with value '%s': %s",
                tag,
                value,
                response,
            )
            return WRITE_FAILURE, False
    except Exception as error:
        LOGGER.error(
            "Unhandled error writing Modbus tag '%s': %s", tag, target.scrub(str(error))
        )
        return WRITE_FAILURE, False

    return WRITE_SUCCESS, True


# ------------------------------ default drivers ------------------------------


def _new_session() -> Any:
    import requests

    session = requests.Session()

    return session


def _new_eip_client(address: str, processor_slot: int) -> Any:
    pylogix = _import_driver("pylogix", package="pylogix", mode="ethernet_ip")
    comm = pylogix.PLC()
    comm.IPAddress = address
    comm.ProcessorSlot = processor_slot

    return comm


def _new_modbus_client(address: str, port: int) -> Any:
    client_module = _import_driver("pymodbus.client", package="pymodbus", mode="modbus")
    client = client_module.ModbusTcpClient(address, port=port)

    return client


def _import_driver(module_name: str, *, package: str, mode: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise PlcDriverUnavailableError(
            f"PLC connection_mode '{mode}' needs the optional package "
            f"`{DRIVER_REQUIREMENTS[package]}`, which is not installed. It is part "
            "of the `enterprise` extra: pip install 'roboflow-workflows[enterprise]'. "
            "Relay mode needs no driver."
        ) from error

    return module


def _close_quietly(connection: Any, method: str) -> None:
    if connection is None:
        return

    try:
        getattr(connection, method)()
    except Exception:
        pass


# ---------------------------------- blocks -----------------------------------


def _check_positive_request_timeout(value: int) -> int:
    if value <= 0:
        raise ValueError("request_timeout must be a positive number of seconds.")

    return value


# Literal-only check: a selected timeout reaches the relay call, which fails
# every tag on a non-positive value, as in V1.
PositiveTimeout = Annotated[StrictInt, AfterValidator(_check_positive_request_timeout)]


def _address_field() -> Any:
    return Field(
        default="127.0.0.1",
        description=(
            "Relay host or URL (relay mode) or PLC IP address (direct modes). "
            "May be a secret, e.g. a relay URL holding credentials."
        ),
        examples=["127.0.0.1", "192.168.1.10", "$steps.secrets.plc_relay_url"],
    )


def _mode_field() -> Any:
    return Field(
        default="relay",
        validate_default=True,
        description=(
            "relay: the on-device PLC Relay over HTTP; ethernet_ip: pylogix; "
            "modbus: pymodbus. Direct modes need the enterprise extra."
        ),
        examples=["relay", "ethernet_ip", "modbus"],
    )


def _relay_port_field() -> Any:
    return Field(
        default=DEFAULT_RELAY_PORT,
        description="Relay port, used when ip_address is not a URL (relay mode).",
        examples=[8007],
    )


def _timeout_field() -> Any:
    return Field(
        default=10,
        description=(
            "Read timeout in seconds of one relay request; it must cover the "
            "relay's PLC batch. Connecting to the relay is capped at 3 s."
        ),
        examples=[10, 30],
    )


def _slot_field() -> Any:
    return Field(
        default=0,
        description="EtherNet/IP processor slot (ethernet_ip mode).",
        examples=[0],
    )


def _modbus_port_field() -> Any:
    return Field(
        default=502, description="Modbus TCP port (modbus mode).", examples=[502]
    )


def _unit_id_field() -> Any:
    return Field(
        default=1, description="Modbus unit / slave id (modbus mode).", examples=[1]
    )


def _discover_restrictions(params: BlockParams) -> list:
    if params.connection_mode in ("ethernet_ip", "modbus"):
        return [PLC_LAN_ACCESS_ACTUAL_RESTRICTION]

    return []


ERROR_STATUS = Output(
    BOOLEAN_KIND,
    description="True when any tag failed; the failures are logged.",
)


class PlcReaderBlock(Block):
    """Read PLC tags into ``tag_values``.

    Example: ``tags_to_read=["camera_msg", "sku_number"]`` through a relay
    gives ``{"camera_msg": 1, "sku_number": "ReadFailure"}`` and
    ``error_status=True`` when the relay reports an error for ``sku_number``.

    Differences from V1: no connection is kept on the block (see the module
    docstring); ``ip_address`` may also select a step output or a secret;
    numeric fields and tag names are strict (``"8007"`` is rejected, not
    converted); an empty tag list opens no Modbus connection; a missing
    direct driver fails the step with ``PlcDriverUnavailableError`` naming
    the pin (V1 could not import at all); messages log to this module's
    logger, not ``inference``.
    """

    type = READER_TYPE
    metadata = {
        "name": "PLC Reader",
        "port_of": "roboflow_core/plc_reader@v1",
        **PLC_METADATA,
    }
    outputs = {
        "tag_values": Output(
            DICTIONARY_KIND,
            description="Value per tag, or 'ReadFailure' for a tag that failed.",
        ),
        "error_status": ERROR_STATUS,
    }

    class Params(BlockParams):
        tags_to_read: List[StrictStr] | Ref(LIST_OF_VALUES_KIND) = Field(
            default_factory=list,
            description=(
                "Tags to read. Relay and ethernet_ip use tag names; modbus uses "
                "area:address with area holding, input, coil or discrete; a bare "
                "number is a holding register."
            ),
            examples=[["camera_msg", "sku_number"], ["holding:100", "coil:0"]],
        )
        ip_address: StrictStr | Ref(SECRET_KIND, STRING_KIND) = _address_field()
        connection_mode: ConnectionMode = _mode_field()
        relay_port: StrictInt | Ref(INTEGER_KIND) = _relay_port_field()
        request_timeout: PositiveTimeout | Ref(INTEGER_KIND) = _timeout_field()
        processor_slot: StrictInt | Ref(INTEGER_KIND) = _slot_field()
        modbus_port: StrictInt | Ref(INTEGER_KIND) = _modbus_port_field()
        modbus_unit_id: StrictInt | Ref(INTEGER_KIND) = _unit_id_field()

    def __init__(self, *, plc_client: Optional[PlcClient] = None) -> None:
        self._plc_client = plc_client

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> list:
        """Declare one external request per call."""
        return [WorkOperation.EXTERNAL_REQUEST]

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> list:
        """Direct modes need LAN access to the PLC; relay mode declares none."""
        restrictions = _discover_restrictions(params)

        return restrictions

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> list:
        """Declare no managed resources, as V1."""
        return []

    def run(
        self,
        *,
        tags_to_read: List[Any],
        ip_address: Union[str, Secret],
        connection_mode: str,
        relay_port: int,
        request_timeout: int,
        processor_slot: int,
        modbus_port: int,
        modbus_unit_id: int,
    ) -> Dict[str, Any]:
        """Read the tags once.

        Args:
            tags_to_read: Tags to read.
            ip_address: Relay or PLC address.
            connection_mode: ``relay``, ``ethernet_ip`` or ``modbus``.
            relay_port: Relay port.
            request_timeout: Relay read timeout in seconds.
            processor_slot: EtherNet/IP processor slot.
            modbus_port: Modbus TCP port.
            modbus_unit_id: Modbus unit id.

        Returns:
            ``tag_values`` and ``error_status``.

        Raises:
            SecretUnavailableError: When ``ip_address`` is a replayed secret.
            PlcDriverUnavailableError: When a direct mode's driver is missing.
        """
        target = PlcTarget(
            connection_mode=connection_mode,
            address=ip_address,
            relay_port=relay_port,
            request_timeout=request_timeout,
            processor_slot=processor_slot,
            modbus_port=modbus_port,
            modbus_unit_id=modbus_unit_id,
        )
        tags = list(tags_to_read)

        tag_values, had_failure = with_plc_client(
            self._plc_client, lambda client: client.read_tags(target, tags)
        )

        return {"tag_values": tag_values, "error_status": had_failure}


class PlcWriterBlock(Block):
    """Write one PLC tag; add one step per tag to write several.

    Example: ``tag="camera_fault"``, ``value="$steps.count.output"`` writes
    the count and gives ``write_result="WriteSuccess"``.

    Differences from V1: those of ``PlcReaderBlock``; ``depends_on`` selects
    a step output (``$steps.x.output``), not a bare step; the value check
    below applies only to literals, as V1 meant.
    """

    type = WRITER_TYPE
    metadata = {
        "name": "PLC Writer",
        "port_of": "roboflow_core/plc_writer@v1",
        **PLC_METADATA,
    }
    outputs = {
        "write_result": Output(
            STRING_KIND,
            description="'WriteSuccess', 'WriteFailure', or '' when disabled.",
        ),
        "error_status": ERROR_STATUS,
    }

    class Params(BlockParams):
        tag: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "The tag to write: a name (relay, ethernet_ip) or holding:N / "
                "coil:N (modbus; input and discrete are read-only)."
            ),
            examples=["camera_fault", "holding:100"],
        )
        value: StrictBool | StrictInt | StrictFloat | StrictStr | Ref() = Field(
            description=(
                "Value to write: bool, int or float; a literal string only in "
                "ethernet_ip mode (Logix STRING tags)."
            ),
            examples=[True, 5, "$steps.counter.count"],
        )
        depends_on: Optional[Ref(allow_none=True)] = Field(
            default=None,
            description=(
                "Output of a step this write must follow; it orders the step, "
                "its value is not read."
            ),
            examples=["$steps.reader.tag_values"],
        )
        ip_address: StrictStr | Ref(SECRET_KIND, STRING_KIND) = _address_field()
        # After ``value``: its validator reads it. Literal-only, so it never
        # reruns on a resolved value.
        connection_mode: ConnectionMode = _mode_field()
        relay_port: StrictInt | Ref(INTEGER_KIND) = _relay_port_field()
        request_timeout: PositiveTimeout | Ref(INTEGER_KIND) = _timeout_field()
        processor_slot: StrictInt | Ref(INTEGER_KIND) = _slot_field()
        modbus_port: StrictInt | Ref(INTEGER_KIND) = _modbus_port_field()
        modbus_unit_id: StrictInt | Ref(INTEGER_KIND) = _unit_id_field()
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip the write; write_result is ''.",
            examples=[False, "$inputs.disable_plc_writer"],
        )

        @field_validator("connection_mode")
        @classmethod
        def _literal_string_needs_ethernet_ip(
            cls, connection_mode: str, info: ValidationInfo
        ) -> str:
            value = info.data.get("value")
            is_literal_string = isinstance(value, str) and not value.startswith("$")
            if connection_mode != "ethernet_ip" and is_literal_string:
                raise ValueError(
                    "A literal string value is only supported in Direct "
                    f"(EtherNet/IP) mode; '{connection_mode}' mode accepts a "
                    "boolean, integer, or float (or a selector that resolves to "
                    "one)."
                )

            return connection_mode

    def __init__(
        self, *, plc_client: Optional[PlcClient] = None, disable_sinks: bool = False
    ) -> None:
        self._plc_client = plc_client
        self._disable_sinks = disable_sinks

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> list:
        """Declare one external request per call."""
        return [WorkOperation.EXTERNAL_REQUEST]

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> list:
        """Direct modes need LAN access to the PLC; relay mode declares none."""
        restrictions = _discover_restrictions(params)

        return restrictions

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> list:
        """Declare no managed resources, as V1."""
        return []

    def run(
        self,
        *,
        tag: Any,
        value: Any,
        depends_on: Any,
        ip_address: Union[str, Secret],
        connection_mode: str,
        relay_port: int,
        request_timeout: int,
        processor_slot: int,
        modbus_port: int,
        modbus_unit_id: int,
        disable_sink: bool,
    ) -> Dict[str, Any]:
        """Write the tag once, unless disabled; never retried.

        Disabled (``disable_sinks`` or ``disable_sink``) returns before the
        address is revealed or any client is used.

        Args:
            tag: Tag to write.
            value: Value to write.
            depends_on: Ordering only; not read.
            ip_address: Relay or PLC address.
            connection_mode: ``relay``, ``ethernet_ip`` or ``modbus``.
            relay_port: Relay port.
            request_timeout: Relay read timeout in seconds.
            processor_slot: EtherNet/IP processor slot.
            modbus_port: Modbus TCP port.
            modbus_unit_id: Modbus unit id.
            disable_sink: Skip this write.

        Returns:
            ``write_result`` and ``error_status``.

        Raises:
            SecretUnavailableError: When ``ip_address`` is a replayed secret.
            PlcDriverUnavailableError: When a direct mode's driver is missing.
            TypeError: When a relay value is not JSON data, as in V1.
        """
        if self._disable_sinks or disable_sink:
            return {"write_result": "", "error_status": False}

        target = PlcTarget(
            connection_mode=connection_mode,
            address=ip_address,
            relay_port=relay_port,
            request_timeout=request_timeout,
            processor_slot=processor_slot,
            modbus_port=modbus_port,
            modbus_unit_id=modbus_unit_id,
        )

        write_results, had_failure = with_plc_client(
            self._plc_client, lambda client: client.write_tags(target, {tag: value})
        )

        return {
            "write_result": write_results.get(tag, WRITE_FAILURE),
            "error_status": had_failure,
        }


BLOCKS = (PlcReaderBlock, PlcWriterBlock)
