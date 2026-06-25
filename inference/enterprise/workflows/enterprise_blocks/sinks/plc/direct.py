"""Direct (no-relay) PLC transports for the PLC Reader/Writer blocks.

Two protocols are supported when the block connects straight to the PLC instead of
going through the on-device PLC Relay service:

* **EtherNet/IP** via ``pylogix`` — tags are addressed by name (e.g. ``Program:Main.Tag``).
* **Modbus TCP** via ``pymodbus`` — tags are addressed as ``area:address`` strings
  (``holding:100``, ``coil:0``, ``input:5``, ``discrete:2``); a bare number defaults to a
  holding register (``100`` == ``holding:100``).

The connection object (``ModbusTcpClient`` / ``pylogix.PLC``) is created once via the
``make_*_client`` factories and then reused across frames by the block — opening and
closing a TCP connection on every frame is a throughput/reliability problem for the
high-FPS, frame-by-frame workflows these blocks target. The block owns the connection's
lifetime; the read/write functions below take a live client, (re)connect it lazily if
needed, and never close it.

Every read/write function returns ``(results, had_failure)`` with the same sentinels the
relay client uses, so the blocks can treat all three transports uniformly. A failure on
one tag is logged and does not stop the remaining tags.
"""

from typing import Any, Dict, List, Mapping, Tuple

import pylogix
from pymodbus.client import ModbusTcpClient

from inference.core.logger import logger
from inference.enterprise.workflows.enterprise_blocks.sinks.plc.client import (
    READ_FAILURE,
    WRITE_FAILURE,
    WRITE_SUCCESS,
)

# Modbus register/bit areas and whether they are writable.
MODBUS_REGISTER_AREAS = {"holding", "input"}
MODBUS_BIT_AREAS = {"coil", "discrete"}
MODBUS_READONLY_AREAS = {"input", "discrete"}
MODBUS_AREAS = MODBUS_REGISTER_AREAS | MODBUS_BIT_AREAS


# --------------------------- EtherNet/IP (pylogix) ---------------------------


def make_eip_client(ip_address: str, processor_slot: int) -> "pylogix.PLC":
    """Create a reusable pylogix EtherNet/IP connection for a PLC.

    The returned object is reused across frames by the block and closed (via ``Close()``)
    only when the block is torn down or the target changes; pylogix opens the underlying
    socket lazily on the first read/write.
    """
    comm = pylogix.PLC()
    comm.IPAddress = ip_address
    comm.ProcessorSlot = processor_slot
    return comm


def ethernet_read_tags(comm, tags: List[str]) -> Tuple[Dict[str, Any], bool]:
    """Read tags from a PLC over EtherNet/IP. Returns (tag_values, had_failure)."""
    results: Dict[str, Any] = {}
    had_failure = False
    for tag in tags:
        value, ok = _eip_read_one(comm, tag)
        results[tag] = value
        had_failure = had_failure or not ok
    return results, had_failure


def ethernet_write_tags(
    comm, tags_to_write: Mapping[str, Any]
) -> Tuple[Dict[str, str], bool]:
    """Write tags to a PLC over EtherNet/IP. Returns (write_results, had_failure)."""
    results: Dict[str, str] = {}
    had_failure = False
    for tag, value in tags_to_write.items():
        status, ok = _eip_write_one(comm, tag, value)
        results[tag] = status
        had_failure = had_failure or not ok
    return results, had_failure


def _eip_read_one(comm, tag: str) -> Tuple[Any, bool]:
    try:
        response = comm.Read(tag)
        if response.Status == "Success":
            return response.Value, True
        logger.error("Error reading EtherNet/IP tag '%s': %s", tag, response.Status)
        return READ_FAILURE, False
    except Exception as e:
        logger.error("Unhandled error reading EtherNet/IP tag '%s': %s", tag, e)
        return READ_FAILURE, False


def _eip_write_one(comm, tag: str, value: Any) -> Tuple[str, bool]:
    try:
        response = comm.Write(tag, value)
        if response.Status == "Success":
            return WRITE_SUCCESS, True
        logger.error(
            "Error writing EtherNet/IP tag '%s' with value '%s': %s",
            tag,
            value,
            response.Status,
        )
        return WRITE_FAILURE, False
    except Exception as e:
        logger.error("Unhandled error writing EtherNet/IP tag '%s': %s", tag, e)
        return WRITE_FAILURE, False


# ------------------------------ Modbus (pymodbus) ----------------------------


def make_modbus_client(ip_address: str, port: int) -> ModbusTcpClient:
    """Create a reusable Modbus TCP client for a PLC.

    The client is reused across frames by the block (and reconnected on demand by
    ``_ensure_modbus_connected``) rather than opened and closed every frame.
    """
    return ModbusTcpClient(ip_address, port=port)


def _ensure_modbus_connected(client: ModbusTcpClient) -> bool:
    """Connect the client if it is not already connected. Returns whether it is up."""
    if client.connected:
        return True
    return bool(client.connect())


def modbus_read_tags(
    client: ModbusTcpClient, unit_id: int, tags: List[str]
) -> Tuple[Dict[str, Any], bool]:
    """Read tags from a PLC over Modbus TCP. Returns (tag_values, had_failure)."""
    if not _ensure_modbus_connected(client):
        logger.error("Failed to connect to Modbus PLC")
        return {tag: READ_FAILURE for tag in tags}, bool(tags)
    results: Dict[str, Any] = {}
    had_failure = False
    for tag in tags:
        value, ok = _modbus_read_one(client, tag, unit_id)
        results[tag] = value
        had_failure = had_failure or not ok
    return results, had_failure


def modbus_write_tags(
    client: ModbusTcpClient, unit_id: int, tags_to_write: Mapping[str, Any]
) -> Tuple[Dict[str, str], bool]:
    """Write tags to a PLC over Modbus TCP. Returns (write_results, had_failure)."""
    if not _ensure_modbus_connected(client):
        logger.error("Failed to connect to Modbus PLC")
        return {tag: WRITE_FAILURE for tag in tags_to_write}, bool(tags_to_write)
    results: Dict[str, str] = {}
    had_failure = False
    for tag, value in tags_to_write.items():
        status, ok = _modbus_write_one(client, tag, value, unit_id)
        results[tag] = status
        had_failure = had_failure or not ok
    return results, had_failure


def _parse_modbus_tag(tag: str) -> Tuple[str, int]:
    """Parse a Modbus tag string into (area, address).

    Formats: ``area:address`` (e.g. ``holding:100``) or a bare address (defaults to a
    holding register). Raises ValueError on an unknown area or non-integer address.
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
    return area, int(address.strip())


def _coerce_coil_value(value: Any) -> bool:
    """Validate a value destined for a Modbus coil (a single bit).

    Coils accept only a real boolean or ``0``/``1``; anything else (a string like
    ``"False"``, a non-binary number) is rejected rather than silently coerced into a
    truthy/falsy bit.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    raise ValueError(f"coil value must be a boolean or 0/1, got {value!r}")


def _coerce_register_value(value: Any) -> int:
    """Validate a value destined for a Modbus holding register (a 16-bit word).

    Registers accept only an integral number in ``0..65535``; booleans, strings, and
    non-integral floats (e.g. ``2.9``) are rejected rather than silently truncated.
    """
    if isinstance(value, bool):
        raise ValueError(f"register value must be an integer in 0..65535, got {value!r}")
    if isinstance(value, int):
        int_value = value
    elif isinstance(value, float) and value.is_integer():
        int_value = int(value)
    else:
        raise ValueError(f"register value must be an integer in 0..65535, got {value!r}")
    if not 0 <= int_value <= 0xFFFF:
        raise ValueError(
            f"register value {int_value} out of range for a 16-bit register (0..65535)"
        )
    return int_value


def _modbus_read_one(client, tag: str, unit_id: int) -> Tuple[Any, bool]:
    try:
        area, address = _parse_modbus_tag(tag)
    except (ValueError, TypeError) as e:
        logger.error("Invalid Modbus tag '%s': %s", tag, e)
        return READ_FAILURE, False
    try:
        if area == "holding":
            response = client.read_holding_registers(address, count=1, slave=unit_id)
        elif area == "input":
            response = client.read_input_registers(address, count=1, slave=unit_id)
        elif area == "coil":
            response = client.read_coils(address, count=1, slave=unit_id)
        else:  # discrete
            response = client.read_discrete_inputs(address, count=1, slave=unit_id)
        if response.isError():
            logger.error("Error reading Modbus tag '%s': %s", tag, response)
            return READ_FAILURE, False
        if area in MODBUS_REGISTER_AREAS:
            return response.registers[0], True
        return bool(response.bits[0]), True
    except Exception as e:
        logger.error("Unhandled error reading Modbus tag '%s': %s", tag, e)
        return READ_FAILURE, False


def _modbus_write_one(client, tag: str, value: Any, unit_id: int) -> Tuple[str, bool]:
    try:
        area, address = _parse_modbus_tag(tag)
    except (ValueError, TypeError) as e:
        logger.error("Invalid Modbus tag '%s': %s", tag, e)
        return WRITE_FAILURE, False
    if area in MODBUS_READONLY_AREAS:
        logger.error("Modbus tag '%s' is read-only (area '%s')", tag, area)
        return WRITE_FAILURE, False
    try:
        payload = (
            _coerce_coil_value(value)
            if area == "coil"
            else _coerce_register_value(value)
        )
    except (ValueError, TypeError) as e:
        logger.error("Invalid value for Modbus tag '%s': %s", tag, e)
        return WRITE_FAILURE, False
    try:
        if area == "coil":
            response = client.write_coil(address, payload, slave=unit_id)
        else:  # holding register
            response = client.write_register(address, payload, slave=unit_id)
        if response.isError():
            logger.error(
                "Error writing Modbus tag '%s' with value '%s': %s",
                tag,
                value,
                response,
            )
            return WRITE_FAILURE, False
        return WRITE_SUCCESS, True
    except Exception as e:
        logger.error("Unhandled error writing Modbus tag '%s': %s", tag, e)
        return WRITE_FAILURE, False
