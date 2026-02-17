import json
import logging
import os.path
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Store workflow events in an in-memory event store with optional persistence to disk,
enabling event logging, audit trails, and time-series data collection from workflow executions.

## How This Block Works

This block captures workflow data as events and stores them in an in-memory event store. The block:

1. Takes a generic payload from any workflow step and an event type label as input
2. Validates filesystem access permissions if disk persistence is enabled
3. Checks cooldown period to prevent event flooding (similar to webhook sink throttling)
4. Creates an event with a unique ID, timestamp, event type, and the provided payload
5. Stores the event in memory for fast access and optional downstream querying
6. Optionally persists events to a JSONL file on disk for durability
7. Returns the event ID, error status, throttling status, and a message

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of
events. Please adjust it according to your needs, setting `0` indicates no cooldown.

During the cooldown period, consecutive runs of the step will cause `throttling_status` output to be set
`True` and no event will be stored.

### Disk Persistence

When `persist_to_disk` is enabled, events are written to a JSONL file in the specified
`target_directory`. The file is named using the `file_name_prefix` and a timestamp. This requires
local filesystem access permissions.

### Disabling the sink

Set `disable_sink` to `True` to skip event storage entirely. This is useful for toggling
event collection via a workflow input parameter.

## Common Use Cases

- **Workflow Audit Trails**: Record each step's output as events for compliance and debugging
- **Detection Event Logging**: Store object detection results as timestamped events for analysis
- **Time-Series Data Collection**: Accumulate workflow metrics over time for reporting
- **Integration with Downstream Systems**: Collect events in memory for batch export or streaming
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Event Store Sink",
            "version": "v1",
            "short_description": "Store workflow events in an event store.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-database",
                "blockPriority": 4,
            },
        }
    )
    type: Literal["roboflow_core/event_store_sink@v1"]
    payload: Selector(kind=[WILDCARD_KIND]) = Field(
        description="Data payload to store as an event. Accepts any workflow output.",
        examples=["$steps.model.predictions"],
    )
    event_type: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="workflow_event",
        description="Label categorizing the event. Used to organize and filter stored events.",
        examples=["detection", "classification", "$inputs.event_type"],
    )
    persist_to_disk: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Whether to persist events to a JSONL file on disk. "
        "Requires local filesystem access when enabled.",
        examples=[True, "$inputs.persist_to_disk"],
    )
    target_directory: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="event_store_output",
        description="Directory path where event files will be saved when persist_to_disk is enabled. "
        "Parent directories are created automatically if they don't exist.",
        examples=["some/location"],
        json_schema_extra={
            "relevant_for": {
                "persist_to_disk": {
                    "values": [True],
                    "required": True,
                },
            }
        },
    )
    file_name_prefix: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="events",
        description="Prefix for the JSONL event file name when persisting to disk.",
        examples=["my_events"],
        json_schema_extra={
            "relevant_for": {
                "persist_to_disk": {
                    "values": [True],
                    "required": True,
                },
            }
        },
    )
    cooldown_seconds: Union[int, Selector(kind=[STRING_KIND])] = Field(
        default=5,
        description="Number of seconds to wait between storing consecutive events. "
        "Set to 0 to disable cooldown.",
        json_schema_extra={
            "always_visible": True,
        },
        examples=["$inputs.cooldown_seconds", 10],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable event storage entirely.",
        examples=[False, "$inputs.disable_event_store"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="event_id", kind=[STRING_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class EventStoreSinkBlockV1(WorkflowBlock):

    def __init__(
        self,
        allow_access_to_file_system: bool,
        allowed_write_directory: Optional[str],
    ):
        self._allow_access_to_file_system = allow_access_to_file_system
        self._allowed_write_directory = allowed_write_directory
        self._last_event_fired: Optional[datetime] = None
        self._event_store = _get_or_create_event_store()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["allow_access_to_file_system", "allowed_write_directory"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        payload: Any,
        event_type: str,
        persist_to_disk: bool,
        target_directory: str,
        file_name_prefix: str,
        cooldown_seconds: int,
        disable_sink: bool,
    ) -> BlockResult:
        if disable_sink:
            return {
                "event_id": "",
                "error_status": False,
                "throttling_status": False,
                "message": "Sink was disabled by parameter `disable_sink`",
            }
        seconds_since_last_event = cooldown_seconds
        if self._last_event_fired is not None:
            seconds_since_last_event = (
                datetime.now() - self._last_event_fired
            ).total_seconds()
        if seconds_since_last_event < cooldown_seconds:
            logging.info(
                "Activated `roboflow_core/event_store_sink@v1` cooldown."
            )
            return {
                "event_id": "",
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        try:
            serialized_payload = _serialize_payload(payload)
            event_id = self._event_store.append(
                event_type=event_type,
                payload=serialized_payload,
            )
            self._last_event_fired = datetime.now()
        except Exception as error:
            logging.warning(
                f"Could not store event in event store: {error}"
            )
            return {
                "event_id": "",
                "error_status": True,
                "throttling_status": False,
                "message": f"Failed to store event: {error}",
            }
        if persist_to_disk:
            disk_result = self._persist_to_disk(
                target_directory=target_directory,
                file_name_prefix=file_name_prefix,
            )
            if disk_result is not None:
                return {
                    "event_id": event_id,
                    "error_status": True,
                    "throttling_status": False,
                    "message": disk_result,
                }
        return {
            "event_id": event_id,
            "error_status": False,
            "throttling_status": False,
            "message": "Event stored successfully",
        }

    def _persist_to_disk(
        self,
        target_directory: str,
        file_name_prefix: str,
    ) -> Optional[str]:
        if not self._allow_access_to_file_system:
            return (
                "Cannot persist to disk - local file system usage is forbidden. "
                "Use self-hosted `inference` or Roboflow Dedicated Deployment."
            )
        if self._allowed_write_directory is not None:
            if not _path_is_within_specified_directory(
                path=target_directory,
                specified_directory=self._allowed_write_directory,
            ):
                return (
                    f"Cannot persist to `{target_directory}` - not a sub-directory of "
                    f"allowed write directory: {self._allowed_write_directory}"
                )
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            file_name = f"{file_name_prefix}_{timestamp}.jsonl"
            file_path = os.path.abspath(
                os.path.join(target_directory, file_name)
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._event_store.flush_to_disk(file_path=file_path)
            return None
        except Exception as error:
            logging.warning(f"Could not persist events to disk: {error}")
            return f"Failed to persist events to disk: {error}"


def _serialize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    try:
        return {"value": json.loads(json.dumps(payload, default=str))}
    except Exception:
        return {"value": str(payload)}


def _path_is_within_specified_directory(
    path: str,
    specified_directory: str,
) -> bool:
    absolute_path = os.path.abspath(path)
    if not absolute_path.endswith(os.sep):
        absolute_path = f"{absolute_path}{os.sep}"
    specified_directory = os.path.abspath(specified_directory)
    if not specified_directory.endswith(os.sep):
        specified_directory = f"{specified_directory}{os.sep}"
    return absolute_path.startswith(specified_directory)


def _get_or_create_event_store():
    """Lazily import and create an EventStore instance."""
    from inference.core.workflows.core_steps.sinks.event_store.event_store import (
        EventStore,
    )

    return EventStore()
