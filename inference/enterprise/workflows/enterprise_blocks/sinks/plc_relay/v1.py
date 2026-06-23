from typing import Dict, List, Optional, Type, Union

import requests
from pydantic import ConfigDict, Field
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
from inference.enterprise.workflows.enterprise_blocks.sinks.plc_relay.client import (
    DEFAULT_RELAY_URL,
    read_tags,
    write_tags,
)

RELAY_INDEPENDENCE_NOTE = """
**Only use this block on a device running the Roboflow PLC Relay service.** It talks to
the relay over HTTP and does **not** open a direct PLC connection, so it does nothing on
a device where the relay is not running. The relay owns the protocol (Allen-Bradley,
Modbus, or Siemens S7), the device IP, and the tag schema, so the same Workflow runs
unchanged across devices. Use the **PLC EthernetIP** block instead if you want to
connect to the PLC directly.

The relay defaults to `http://localhost:8007` (the relay running on the same device).
Use the advanced section to point at a relay on a different host or port. Because the
Workflow is independent of the device, tag names and values are entered free-form; the
relay validates each tag against its configured schema and the block logs a precise
error whenever a relay constraint is violated.

All tags are sent in a single batch request, so reading or writing N tags costs one
HTTP round-trip and one PLC transaction per frame (the relay connection is kept alive
across frames). The relay validates the whole batch up front: if it rejects the request
(for example an unknown or non-writable tag), the entire batch fails and every tag is
reported as a failure.
""".strip()

READER_LONG_DESCRIPTION = f"""
The **PLC Relay Reader** block reads tag values from a PLC through the on-device PLC
Relay service and makes them available to the rest of the Workflow.

{RELAY_INDEPENDENCE_NOTE}

When a tag cannot be read (the tag is not configured on the relay, the relay is
unreachable, or the PLC returns an error), the error is logged and that tag's value in
`tag_values` is set to `"ReadFailure"`. `error_status` is `True` if any tag failed.
"""

WRITER_LONG_DESCRIPTION = f"""
The **PLC Relay Writer** block writes tag values to a PLC through the on-device PLC
Relay service.

{RELAY_INDEPENDENCE_NOTE}

When a tag cannot be written (the tag is not configured on the relay, the tag is not
writable, or the value has the wrong type or is out of range for the tag's data type),
the error is logged and that tag's entry in `write_results` is set to `"WriteFailure"`.
`error_status` is `True` if any tag failed.
"""

RELAY_UI_MANIFEST_BASE = {
    "section": "industrial",
    "icon": "fal fa-microchip",
    "enterprise_only": True,
    "local_only": True,
}


def _relay_url_field() -> Field:
    return Field(
        default=DEFAULT_RELAY_URL,
        description="Base URL of the PLC Relay service. Defaults to the relay running on the same "
        "device (localhost). Override this to reach a relay on a different host or port.",
        examples=[DEFAULT_RELAY_URL, "http://192.168.1.10:8007"],
        json_schema_extra={"additional_section": True},
    )


def _request_timeout_field() -> Field:
    return Field(
        default=5,
        description="Timeout in seconds for each request to the PLC Relay service.",
        examples=[5, 10],
        json_schema_extra={"additional_section": True},
    )


class PLCRelayReaderBlockManifest(WorkflowBlockManifest):
    """Manifest for a block that reads PLC tag values through the PLC Relay service."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC Relay Reader",
            "version": "v1",
            "short_description": "Read PLC tag values through the on-device PLC Relay service.",
            "long_description": READER_LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "transformation",
            "ui_manifest": {**RELAY_UI_MANIFEST_BASE, "blockPriority": 12},
        }
    )

    type: Literal["roboflow_core/plc_relay_reader@v1"]

    tags_to_read: Union[
        List[str],
        Selector(kind=[LIST_OF_VALUES_KIND]),
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]),
    ] = Field(
        default_factory=list,
        description="List of PLC tag names to read. Tag names are free-form and must match tags "
        "configured on the relay.",
        examples=[["camera_msg", "sku_number"]],
        json_schema_extra={"always_visible": True},
    )

    relay_url: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        _relay_url_field()
    )
    request_timeout: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _request_timeout_field()
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


class PLCRelayReaderBlockV1(WorkflowBlock):
    """Reads PLC tag values through the PLC Relay service.

    All requested tags are read in a single ``/read_batch`` call (one HTTP round-trip
    and one PLC transaction per frame). The HTTP session is reused across frames so the
    relay connection is kept alive at high FPS. If the relay rejects the batch, every
    tag is reported as ``"ReadFailure"`` and the relay's error detail is logged.
    """

    def __init__(self):
        self._session: Optional[requests.Session] = None

    def __del__(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PLCRelayReaderBlockManifest

    def run(
        self,
        tags_to_read: List[str],
        relay_url: str = DEFAULT_RELAY_URL,
        request_timeout: int = 5,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        """Read tags through the PLC Relay service.

        Args:
            tags_to_read (List[str]): Tags to read.
            relay_url (str): Base URL of the PLC Relay service.
            request_timeout (int): Request timeout in seconds for the batch call.
            image (Optional[WorkflowImageData]): Not required for this block.
            metadata (Optional[VideoMetadata]): Not required for this block.

        Returns:
            dict: A dictionary with `tag_values` (mapping each tag to its value, or
            `"ReadFailure"` on error) and `error_status` (True if any read failed).
        """
        tag_values, had_failure = read_tags(
            self._get_session(), relay_url.rstrip("/"), tags_to_read, request_timeout
        )
        return {"tag_values": tag_values, "error_status": had_failure}


class PLCRelayWriterBlockManifest(WorkflowBlockManifest):
    """Manifest for a block that writes PLC tag values through the PLC Relay service."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PLC Relay Writer",
            "version": "v1",
            "short_description": "Write PLC tag values through the on-device PLC Relay service.",
            "long_description": WRITER_LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
            "ui_manifest": {**RELAY_UI_MANIFEST_BASE, "blockPriority": 13},
        }
    )

    type: Literal["roboflow_core/plc_relay_writer@v1"]

    tags_to_write: Union[
        Dict[str, Union[bool, int, float]],
        Selector(kind=[DICTIONARY_KIND]),
        WorkflowParameterSelector(kind=[DICTIONARY_KIND]),
    ] = Field(
        default_factory=dict,
        description="Dictionary mapping tag names to the values to write. Tag names are free-form, "
        "and values must be a boolean, integer, or float (the value types the relay accepts); the "
        "relay validates the type and range against each tag's configured data type.",
        examples=[{"camera_fault": True, "defect_count": 5}],
        json_schema_extra={"always_visible": True},
    )

    depends_on: Selector() = Field(
        description="Reference to the step output this block depends on.",
        examples=["$steps.some_previous_step"],
    )

    relay_url: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        _relay_url_field()
    )
    request_timeout: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        _request_timeout_field()
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, skip all communication with the PLC Relay and return empty results.",
        examples=[False, "$inputs.disable_plc_relay"],
        json_schema_extra={"additional_section": True},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="write_results", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class PLCRelayWriterBlockV1(WorkflowBlock):
    """Writes PLC tag values through the PLC Relay service.

    All tags are written in a single ``/write_batch`` call (one HTTP round-trip and one
    PLC transaction per frame). The HTTP session is reused across frames so the relay
    connection is kept alive at high FPS. The relay validates the whole batch up front,
    so if it rejects the request (an unknown tag, a non-writable tag, or a bad value)
    every tag is reported as ``"WriteFailure"`` and the relay's error detail is logged.
    """

    def __init__(self):
        self._session: Optional[requests.Session] = None

    def __del__(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PLCRelayWriterBlockManifest

    def run(
        self,
        tags_to_write: Dict[str, Union[bool, int, float]],
        depends_on: any,
        relay_url: str = DEFAULT_RELAY_URL,
        request_timeout: int = 5,
        disable_sink: bool = False,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> dict:
        """Write tags through the PLC Relay service.

        Args:
            tags_to_write (Dict[str, Union[bool, int, float]]): Tags and values to write.
            depends_on (any): The step output this block depends on.
            relay_url (str): Base URL of the PLC Relay service.
            request_timeout (int): Request timeout in seconds for the batch call.
            disable_sink (bool): If True, skip all relay communication.
            image (Optional[WorkflowImageData]): Not required for this block.
            metadata (Optional[VideoMetadata]): Not required for this block.

        Returns:
            dict: A dictionary with `write_results` (mapping each tag to `"WriteSuccess"`
            or `"WriteFailure"`) and `error_status` (True if any write failed).
        """
        if disable_sink:
            return {"write_results": {}, "error_status": False}

        write_results, had_failure = write_tags(
            self._get_session(), relay_url.rstrip("/"), tags_to_write, request_timeout
        )
        return {"write_results": write_results, "error_status": had_failure}
