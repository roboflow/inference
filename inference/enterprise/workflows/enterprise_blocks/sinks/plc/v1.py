from typing import Any, Dict, List, Optional, Tuple, Type, Union

import requests
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Literal

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.plc.client import (
    DEFAULT_RELAY_PORT,
    WRITE_FAILURE,
    read_tags,
    relay_base_url,
    write_tags,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.plc.direct import (
    ethernet_read_tags,
    ethernet_write_tags,
    make_eip_client,
    make_modbus_client,
    modbus_read_tags,
    modbus_write_tags,
)

# Connection modes (the `connection_mode` selector) and per-mode field visibility.
RELAY_RELEVANT = {"connection_mode": {"values": ["relay"]}}
ETHERNET_RELEVANT = {"connection_mode": {"values": ["ethernet_ip"]}}
MODBUS_RELEVANT = {"connection_mode": {"values": ["modbus"]}}

CONNECTION_NOTE = """
This block can reach the PLC three ways, selected by **Connection mode** in the advanced
section:

- **Roboflow PLC Relay** (default): sends tags to the on-device **PLC Relay** service over
  HTTP. The relay owns the protocol (Allen-Bradley, Modbus, or Siemens S7), the device IP,
  and the tag schema, so the same Workflow runs unchanged across devices. Tags are sent in
  a single batch request per frame over a persistent keep-alive connection (high FPS).
- **Direct (EtherNet/IP)**: connects straight to the PLC with `pylogix`. Tags are
  addressed by name (e.g. `Program:MainProgram.Tag1`).
- **Direct (Modbus TCP)**: connects straight to the PLC with `pymodbus`. Tags are
  addressed as `area:address` (`holding:100`, `coil:0`, `input:5`, `discrete:2`); a bare
  number defaults to a holding register.

**Address** is the relay host in relay mode, or the PLC's IP address in either direct
mode. The advanced section exposes the relevant extras per mode (relay port, processor
slot, Modbus port / unit id).

On any failure the error is logged and that tag's entry in the output is set to
`"ReadFailure"` / `"WriteFailure"`; `error_status` is `True` if any tag failed.
""".strip()

READER_LONG_DESCRIPTION = f"""
The **PLC Reader** block reads tag values from a PLC and makes them available to the rest
of the Workflow.

{CONNECTION_NOTE}
"""

WRITER_LONG_DESCRIPTION = f"""
The **PLC Writer** block writes a single tag value to a PLC. To write several tags, add one
PLC Writer block per tag (each is its own request to the PLC).

{CONNECTION_NOTE}
"""

PLC_UI_MANIFEST_BASE = {
    "section": "industrial",
    "icon": "fal fa-microchip",
    "enterprise_only": True,
    "local_only": True,
}


def _is_selector(value: object) -> bool:
    """True if value is a workflow selector reference ($inputs.X / $steps.X.Y)."""
    return isinstance(value, str) and value.startswith("$")


def _connection_mode_field() -> Field:
    return Field(
        default="relay",
        description="How to reach the PLC: through the on-device PLC Relay, or directly over "
        "EtherNet/IP or Modbus TCP.",
        examples=["relay", "ethernet_ip", "modbus"],
        json_schema_extra={
            "additional_section": True,
            "values_metadata": {
                "relay": {
                    "name": "Roboflow PLC Relay",
                    "description": "Communicate through the on-device PLC Relay service over HTTP "
                    "(protocol-agnostic; the relay handles Allen-Bradley, Modbus, or Siemens S7).",
                },
                "ethernet_ip": {
                    "name": "Direct (EtherNet/IP)",
                    "description": "Connect directly to the PLC over EtherNet/IP using pylogix.",
                },
                "modbus": {
                    "name": "Direct (Modbus TCP)",
                    "description": "Connect directly to the PLC over Modbus TCP using pymodbus.",
                },
            },
        },
    )


def _ip_address_field() -> Field:
    return Field(
        default="127.0.0.1",
        description="Address of the PLC Relay (relay mode) or of the PLC itself (direct modes). "
        "A bare host/IP is accepted; in relay mode a full URL may also be given.",
        examples=["127.0.0.1", "192.168.1.10"],
        json_schema_extra={"always_visible": True},
    )


def _relay_port_field() -> Field:
    return Field(
        default=DEFAULT_RELAY_PORT,
        description="Port of the PLC Relay service (relay mode).",
        examples=[8007],
        json_schema_extra={"additional_section": True, "relevant_for": RELAY_RELEVANT},
    )


def _request_timeout_field() -> Field:
    return Field(
        default=10,
        description="Read timeout in seconds for each request to the PLC Relay service "
        "(relay mode). This must cover the relay's synchronous PLC batch transaction, which "
        "can run for seconds against a slow or disconnected PLC (especially Modbus / S7); if "
        "it is exceeded the request is abandoned and every tag in the batch is reported as a "
        "failure. Raise it for slow or flaky PLCs. (Connecting to the relay itself uses a "
        "separate short timeout, so a down relay still fails fast.)",
        examples=[10, 30],
        json_schema_extra={"additional_section": True, "relevant_for": RELAY_RELEVANT},
    )


def _processor_slot_field() -> Field:
    return Field(
        default=0,
        description="EtherNet/IP processor slot of the PLC (direct EtherNet/IP mode).",
        examples=[0],
        json_schema_extra={
            "additional_section": True,
            "relevant_for": ETHERNET_RELEVANT,
        },
    )


def _modbus_port_field() -> Field:
    return Field(
        default=502,
        description="Modbus TCP port of the PLC (direct Modbus mode).",
        examples=[502],
        json_schema_extra={"additional_section": True, "relevant_for": MODBUS_RELEVANT},
    )


def _modbus_unit_id_field() -> Field:
    return Field(
        default=1,
        description="Modbus unit / slave id of the PLC (direct Modbus mode).",
        examples=[1],
        json_schema_extra={"additional_section": True, "relevant_for": MODBUS_RELEVANT},
    )


class _PLCConnectionMixin:
    """Shared connection state + read/write dispatch for the PLC blocks.

    The connection for the active mode (relay HTTP session, Modbus TCP client, or pylogix
    EtherNet/IP client) is created once and reused across frames, so a high-FPS workflow
    does not pay a connect/teardown on every frame. A direct client is rebuilt only if the
    target (address/port/slot) changes, and torn down when the block is.
    """

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._modbus_client = None
        self._modbus_key: Optional[Tuple[str, int]] = None
        self._eip_client = None
        self._eip_key: Optional[Tuple[str, int]] = None

    def __del__(self):
        self._close_session()
        self._close_modbus()
        self._close_eip()

    def _close_session(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _close_modbus(self) -> None:
        if self._modbus_client is not None:
            try:
                self._modbus_client.close()
            except Exception:
                pass
            self._modbus_client = None
            self._modbus_key = None

    def _close_eip(self) -> None:
        if self._eip_client is not None:
            try:
                self._eip_client.Close()
            except Exception:
                pass
            self._eip_client = None
            self._eip_key = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _get_modbus_client(self, ip_address: str, port: int):
        key = (ip_address, port)
        if self._modbus_client is not None and self._modbus_key != key:
            self._close_modbus()
        if self._modbus_client is None:
            self._modbus_client = make_modbus_client(ip_address, port)
            self._modbus_key = key
        return self._modbus_client

    def _get_eip_client(self, ip_address: str, processor_slot: int):
        key = (ip_address, processor_slot)
        if self._eip_client is not None and self._eip_key != key:
            self._close_eip()
        if self._eip_client is None:
            self._eip_client = make_eip_client(ip_address, processor_slot)
            self._eip_key = key
        return self._eip_client

    def _read(
        self,
        connection_mode: str,
        ip_address: str,
        relay_port: int,
        processor_slot: int,
        modbus_port: int,
        modbus_unit_id: int,
        request_timeout: int,
        tags: List[str],
    ) -> Tuple[Dict[str, Any], bool]:
        if connection_mode == "ethernet_ip":
            return ethernet_read_tags(
                self._get_eip_client(ip_address, processor_slot), tags
            )
        if connection_mode == "modbus":
            return modbus_read_tags(
                self._get_modbus_client(ip_address, modbus_port), modbus_unit_id, tags
            )
        base_url = relay_base_url(ip_address, relay_port)
        return read_tags(self._get_session(), base_url, tags, request_timeout)

    def _write(
        self,
        connection_mode: str,
        ip_address: str,
        relay_port: int,
        processor_slot: int,
        modbus_port: int,
        modbus_unit_id: int,
        request_timeout: int,
        tags_to_write: Dict[str, Union[bool, int, float]],
    ) -> Tuple[Dict[str, str], bool]:
        if connection_mode == "ethernet_ip":
            return ethernet_write_tags(
                self._get_eip_client(ip_address, processor_slot), tags_to_write
            )
        if connection_mode == "modbus":
            return modbus_write_tags(
                self._get_modbus_client(ip_address, modbus_port),
                modbus_unit_id,
                tags_to_write,
            )
        base_url = relay_base_url(ip_address, relay_port)
        return write_tags(self._get_session(), base_url, tags_to_write, request_timeout)


class PLCReaderBlockManifest(WorkflowBlockManifest):
    """Manifest for a block that reads PLC tag values (relay or direct connection)."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC Reader",
            "version": "v1",
            "short_description": "Read PLC tag values via the PLC Relay or a direct EtherNet/IP / Modbus connection.",
            "long_description": READER_LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "transformation",
            "ui_manifest": {**PLC_UI_MANIFEST_BASE, "blockPriority": 12},
        }
    )

    type: Literal["roboflow_core/plc_reader@v1"]

    tags_to_read: Union[
        List[str],
        Selector(kind=[LIST_OF_VALUES_KIND]),
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]),
    ] = Field(
        default_factory=list,
        description="PLC tags to read, entered comma-separated (e.g. `camera_msg, sku_number`). "
        "Relay and Direct (EtherNet/IP) modes use tag names. Direct (Modbus TCP) mode uses "
        "`area:address`, where area is `holding`, `input` (read-only), `coil`, or `discrete` "
        "(read-only); a bare number means a holding register (`100` = `holding:100`). "
        "Example for Modbus: `holding:100, coil:0`.",
        examples=[["camera_msg", "sku_number"], ["holding:100", "coil:0"]],
        json_schema_extra={"always_visible": True},
    )

    ip_address: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        _ip_address_field()
    )
    connection_mode: Literal["relay", "ethernet_ip", "modbus"] = (
        _connection_mode_field()
    )
    relay_port: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _relay_port_field()
    )
    request_timeout: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _request_timeout_field()
    )
    processor_slot: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _processor_slot_field()
    )
    modbus_port: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _modbus_port_field()
    )
    modbus_unit_id: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _modbus_unit_id_field()
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="tag_values", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class PLCReaderBlockV1(_PLCConnectionMixin, WorkflowBlock):
    """Reads PLC tag values over the PLC Relay or a direct EtherNet/IP / Modbus connection."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PLCReaderBlockManifest

    def run(
        self,
        tags_to_read: List[str],
        ip_address: str = "127.0.0.1",
        connection_mode: str = "relay",
        relay_port: int = DEFAULT_RELAY_PORT,
        request_timeout: int = 10,
        processor_slot: int = 0,
        modbus_port: int = 502,
        modbus_unit_id: int = 1,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        """Read tags from the PLC. Returns `tag_values` and `error_status`."""
        tag_values, had_failure = self._read(
            connection_mode,
            ip_address,
            relay_port,
            processor_slot,
            modbus_port,
            modbus_unit_id,
            request_timeout,
            tags_to_read,
        )
        return {"tag_values": tag_values, "error_status": had_failure}


class PLCWriterBlockManifest(WorkflowBlockManifest):
    """Manifest for a block that writes PLC tag values (relay or direct connection)."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC Writer",
            "version": "v1",
            "short_description": "Write PLC tag values via the PLC Relay or a direct EtherNet/IP / Modbus connection.",
            "long_description": WRITER_LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
            "ui_manifest": {**PLC_UI_MANIFEST_BASE, "blockPriority": 13},
        }
    )

    type: Literal["roboflow_core/plc_writer@v1"]

    tag: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="The single PLC tag to write. Relay and Direct (EtherNet/IP) modes use a "
        "tag name (e.g. `camera_fault`); Direct (Modbus TCP) mode uses `area:address` "
        "(`holding:100`, `coil:0`; a bare number is a holding register, and only `holding` "
        "registers and `coil`s are writable, not the read-only `input` / `discrete` areas). "
        "To write several tags, add one PLC Writer block per tag.",
        examples=["camera_fault", "holding:100"],
        json_schema_extra={"always_visible": True},
    )

    # `value` accepts a literal or a selector (e.g. a previous step's output). `str` is allowed
    # in the schema for all modes; `_validate_write_value_type` then rejects *literal* string
    # values for relay and Modbus (the relay contract is bool/int/float; Modbus registers/coils
    # are numeric/boolean). Direct (EtherNet/IP) keeps `str` for Logix STRING tags. A selector
    # resolves at runtime and is not statically inspectable, so it is skipped at validation time.
    value: Union[bool, int, float, str, Selector()] = Field(
        description="The value to write to the tag. May be a fixed value or a reference to a "
        "workflow input or a previous step's output. Must be a boolean, integer, or float, "
        "except Direct (EtherNet/IP) mode, which also accepts strings (for Logix STRING tags).",
        examples=[True, 5, "$steps.counter.count"],
        json_schema_extra={"always_visible": True},
    )

    depends_on: Selector() = Field(
        description="Reference to the step output this block depends on.",
        examples=["$steps.some_previous_step"],
    )

    ip_address: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        _ip_address_field()
    )
    connection_mode: Literal["relay", "ethernet_ip", "modbus"] = (
        _connection_mode_field()
    )
    relay_port: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _relay_port_field()
    )
    request_timeout: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _request_timeout_field()
    )
    processor_slot: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _processor_slot_field()
    )
    modbus_port: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _modbus_port_field()
    )
    modbus_unit_id: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _modbus_unit_id_field()
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, skip the write to the PLC and return an empty result.",
        examples=[False, "$inputs.disable_plc_writer"],
        json_schema_extra={"additional_section": True},
    )

    @model_validator(mode="after")
    def _validate_write_value_type(self):
        # A literal string value is only meaningful for Direct (EtherNet/IP) (Logix STRING tags).
        # The relay accepts only bool/int/float, and Modbus registers/coils are numeric/boolean,
        # so reject a literal string for those modes at validation time. A selector resolves at
        # runtime and is not statically inspectable, so it is allowed (the relay / Modbus
        # transports reject a bad resolved value per tag).
        if self.connection_mode == "ethernet_ip":
            return self
        if isinstance(self.value, str) and not _is_selector(self.value):
            raise ValueError(
                f"A literal string value is only supported in Direct (EtherNet/IP) mode; "
                f"'{self.connection_mode}' mode accepts a boolean, integer, or float (or a "
                f"selector that resolves to one)."
            )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="write_result", kind=[STRING_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class PLCWriterBlockV1(_PLCConnectionMixin, WorkflowBlock):
    """Writes PLC tag values over the PLC Relay or a direct EtherNet/IP / Modbus connection."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PLCWriterBlockManifest

    def run(
        self,
        tag: str,
        value: Union[bool, int, float, str],
        depends_on: any,
        ip_address: str = "127.0.0.1",
        connection_mode: str = "relay",
        relay_port: int = DEFAULT_RELAY_PORT,
        request_timeout: int = 10,
        processor_slot: int = 0,
        modbus_port: int = 502,
        modbus_unit_id: int = 1,
        disable_sink: bool = False,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        """Write a single tag to the PLC. Returns `write_result` and `error_status`."""
        if disable_sink:
            return {"write_result": "", "error_status": False}

        write_results, had_failure = self._write(
            connection_mode,
            ip_address,
            relay_port,
            processor_slot,
            modbus_port,
            modbus_unit_id,
            request_timeout,
            {tag: value},
        )
        return {
            "write_result": write_results.get(tag, WRITE_FAILURE),
            "error_status": had_failure,
        }
