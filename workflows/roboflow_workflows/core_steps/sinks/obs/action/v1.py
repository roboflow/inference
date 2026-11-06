import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, model_validator
from roboflow_workflows.core_steps.common.detections_fields import (
    detections_boxes_and_confidences,
    detections_count,
)
from roboflow_workflows.core_steps.common.workload_presets import (
    COOLDOWN_ACTUAL_RESTRICTION,
)
from roboflow_workflows.core_steps.sinks.noop import disabled_sink_message
from roboflow_workflows.core_steps.sinks.obs.client import call_with_reconnect
from roboflow_workflows.core_steps.sinks.obs.restrictions import (
    OBS_LAN_ACCESS_RESTRICTION,
)
from roboflow_workflows.core_steps.sinks.obs.websocket_client import OBSRequestError
from roboflow_workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    OBS_CONNECTION_KIND,
    STRING_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    COOLDOWN_HTTP_SOFT_RESTRICTION,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
)

logger = logging.getLogger(__name__)

SET_SCENE = "set_scene"
SET_SOURCE_VISIBILITY = "set_source_visibility"
SET_TEXT = "set_text"
TOGGLE_FILTER = "toggle_filter"
TRIGGER_HOTKEY = "trigger_hotkey"
SET_SOURCE_TRANSFORM = "set_source_transform"
MOVE_SOURCE_TO_DETECTION = "move_source_to_detection"
START_VIRTUAL_CAMERA = "start_virtual_camera"
STOP_VIRTUAL_CAMERA = "stop_virtual_camera"
START_RECORDING = "start_recording"
STOP_RECORDING = "stop_recording"

# actions that name a scene or source, so an unknown name can be answered with
# the names OBS actually has
NAMED_TARGET_ACTIONS = {
    SET_SCENE,
    SET_SOURCE_VISIBILITY,
    SET_TEXT,
    TOGGLE_FILTER,
    SET_SOURCE_TRANSFORM,
    MOVE_SOURCE_TO_DETECTION,
}
# obs-websocket RequestStatus::ResourceNotFound
RESOURCE_NOT_FOUND = 600

ActionType = Literal[
    SET_SCENE,
    SET_SOURCE_VISIBILITY,
    SET_TEXT,
    TOGGLE_FILTER,
    TRIGGER_HOTKEY,
    SET_SOURCE_TRANSFORM,
    MOVE_SOURCE_TO_DETECTION,
    START_VIRTUAL_CAMERA,
    STOP_VIRTUAL_CAMERA,
    START_RECORDING,
    STOP_RECORDING,
]

# Fields each action cannot run without. Validated at compile time so a Workflow
# fails in the builder rather than halfway through a live stream.
REQUIRED_FIELDS_BY_ACTION: Dict[str, Tuple[str, ...]] = {
    SET_SCENE: ("scene_name",),
    SET_SOURCE_VISIBILITY: ("scene_name", "source_name", "enabled"),
    SET_TEXT: ("source_name", "text"),
    TOGGLE_FILTER: ("source_name", "filter_name", "enabled"),
    TRIGGER_HOTKEY: ("hotkey_name",),
    SET_SOURCE_TRANSFORM: (
        "scene_name",
        "source_name",
        "position_x",
        "position_y",
        "width",
        "height",
    ),
    MOVE_SOURCE_TO_DETECTION: ("scene_name", "source_name", "predictions", "image"),
    START_VIRTUAL_CAMERA: (),
    STOP_VIRTUAL_CAMERA: (),
    START_RECORDING: (),
    STOP_RECORDING: (),
}

LONG_DESCRIPTION = """
Perform a single action in a running OBS Studio instance - switch scene, show or hide a source,
update an on-screen text source, toggle a filter, press a hotkey, or control the virtual camera
and recording.

## How This Block Works

This block receives a connection from an OBS Connection block and issues one obs-websocket request
per execution. The block:

1. Reads the OBS connection descriptor produced upstream
2. Sends the configured action to OBS, reconnecting once if OBS restarted since the last request
3. Reports whether the request succeeded through `error_status` and `message`

The block does not decide *when* to act. Place flow-control blocks upstream to shape the trigger:
a **Delta Filter** so the action fires only when the value it depends on changes, a **Rate Limiter**
to cap how often a branch runs, or a **Continue If** to gate on a condition. This keeps the trigger
logic visible in the Workflow rather than buried in block configuration.

Supported actions:

| Action | Required fields | Effect |
|---|---|---|
| `set_scene` | `scene_name` | Switches the active program scene |
| `set_source_visibility` | `scene_name`, `source_name`, `enabled` | Shows or hides a source in a scene |
| `set_text` | `source_name`, `text` | Replaces the contents of a text source |
| `toggle_filter` | `source_name`, `filter_name`, `enabled` | Enables or disables a filter on a source |
| `trigger_hotkey` | `hotkey_name` | Fires an OBS hotkey by name, reaching actions with no dedicated request |
| `set_source_transform` | `scene_name`, `source_name`, `position_x`, `position_y`, `width`, `height` | Places and sizes a source at explicit canvas coordinates |
| `move_source_to_detection` | `scene_name`, `source_name`, `predictions`, `image` | Moves and sizes a source to the highest-confidence detection's bounding box, hiding it when nothing is detected (`hide_when_empty`) |
| `start_virtual_camera` / `stop_virtual_camera` | none | Controls the OBS virtual camera |
| `start_recording` / `stop_recording` | none | Controls recording |

## Common Use Cases

- **Scene switching on detection**: change scene when a class appears on camera
- **Live overlays**: write a running object count into an OBS text source
- **Privacy filters**: enable a blur filter when a face, badge or document is detected
- **Highlight capture**: start recording when activity begins and stop when it ends
- **Virtual camera control**: bring the OBS virtual camera up so a video call picks up the composed scene

## Connecting to Other Blocks

- **After an OBS Connection block**, which supplies the required `connection` input
- **After a Delta Filter block**, so the action fires on change rather than on every frame
- **After an Expression or Property Definition block**, to compute the scene name or text to display
- **After a Continue If block**, to gate the action on a condition

## Requirements

Requires a reachable OBS Studio instance (28 or later) with its websocket server enabled. The block
speaks the obs-websocket 5.x protocol directly, so no extra package is needed. Not available on
Roboflow Hosted Serverless or Dedicated Deployments, which cannot reach a local OBS instance.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OBS Action",
            "version": "v1",
            "short_description": "Switch scenes, toggle sources and filters, or drive recording in OBS Studio.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "video",
                "icon": "fal fa-reel",
                "blockPriority": 2,
                "popular": False,
            },
        }
    )
    type: Literal["roboflow_core/obs_action@v1"]
    connection: Selector(kind=[OBS_CONNECTION_KIND]) = Field(
        description="Connection produced by an OBS Connection block.",
        examples=["$steps.obs_connection.connection"],
    )
    action: ActionType = Field(
        description="The OBS operation to perform.",
        examples=[SET_SCENE],
        json_schema_extra={"always_visible": True},
    )
    scene_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Name of the OBS scene. Required by `set_scene`, and identifies the scene "
        "holding the source for `set_source_visibility`.",
        examples=["Detected", "$steps.scene_expression.output"],
        json_schema_extra={
            "relevant_for": {
                "action": {
                    "values": [
                        "set_scene",
                        "set_source_visibility",
                        "set_source_transform",
                        "move_source_to_detection",
                    ],
                    "required": True,
                },
            }
        },
    )
    source_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Name of the OBS source (called an input in OBS). Required by "
        "`set_source_visibility`, `set_text` and `toggle_filter`.",
        examples=["Overlay", "$inputs.source_name"],
        json_schema_extra={
            "relevant_for": {
                "action": {
                    "values": [
                        "set_source_visibility",
                        "set_text",
                        "toggle_filter",
                        "set_source_transform",
                        "move_source_to_detection",
                    ],
                    "required": True,
                },
            }
        },
    )
    filter_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Name of the filter on the source. Required by `toggle_filter`.",
        examples=["Blur"],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["toggle_filter"], "required": True},
            }
        },
    )
    text: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Text to write into the text source. Required by `set_text`.",
        examples=["$steps.count_expression.output"],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["set_text"], "required": True},
            }
        },
    )
    enabled: Optional[Union[bool, Selector(kind=[BOOLEAN_KIND])]] = Field(
        default=None,
        description="Target state for `set_source_visibility` and `toggle_filter`.",
        examples=[True, "$steps.detection_present.output"],
        json_schema_extra={
            "relevant_for": {
                "action": {
                    "values": ["set_source_visibility", "toggle_filter"],
                    "required": True,
                },
            }
        },
    )
    hotkey_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Name of the OBS hotkey to trigger, as listed by the obs-websocket "
        "`GetHotkeyList` request. Required by `trigger_hotkey`.",
        examples=["OBSBasic.StartStreaming"],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["trigger_hotkey"], "required": True},
            }
        },
    )
    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description="Predictions to track. Required by `move_source_to_detection`, which follows "
        "the highest-confidence detection - filter upstream to select the class to track.",
        examples=["$steps.detections_filter.predictions"],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["move_source_to_detection"], "required": True},
            }
        },
    )
    image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        description="The image the predictions were made on. Required by "
        "`move_source_to_detection` to map detection coordinates onto the OBS canvas.",
        examples=["$inputs.image"],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["move_source_to_detection"], "required": True},
            }
        },
    )
    position_x: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Left edge of the source in OBS canvas pixels. Required by `set_source_transform`.",
        examples=[640.0],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["set_source_transform"], "required": True},
            }
        },
    )
    position_y: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Top edge of the source in OBS canvas pixels. Required by `set_source_transform`.",
        examples=[360.0],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["set_source_transform"], "required": True},
            }
        },
    )
    width: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Width of the source in OBS canvas pixels. Required by `set_source_transform`.",
        examples=[512.0],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["set_source_transform"], "required": True},
            }
        },
    )
    height: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Height of the source in OBS canvas pixels. Required by `set_source_transform`.",
        examples=[512.0],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["set_source_transform"], "required": True},
            }
        },
    )
    fit: Literal["stretch", "fit", "fill"] = Field(
        default="fit",
        description="How the source fills the target rectangle: `stretch` matches it exactly "
        "(may distort), `fit` letterboxes inside it, `fill` covers it (may crop).",
        examples=["fit"],
        json_schema_extra={
            "relevant_for": {
                "action": {
                    "values": ["set_source_transform", "move_source_to_detection"],
                    "required": False,
                },
            }
        },
    )
    offset_x: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.0,
        description="For `move_source_to_detection`: shift the source horizontally by this "
        "multiple of the detection's width. -1.0 places it fully to the left of the detection, "
        "1.0 fully to the right, 0 centres it on the detection.",
        examples=[0.0, -1.0, 1.0],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["move_source_to_detection"], "required": False},
            }
        },
    )
    offset_y: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.0,
        description="For `move_source_to_detection`: shift the source vertically by this "
        "multiple of the detection's height.",
        examples=[0.0, -1.0],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["move_source_to_detection"], "required": False},
            }
        },
    )
    hide_when_empty: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="For `move_source_to_detection`: hide the source when there are no "
        "detections, and show it again when a detection returns.",
        examples=[True],
        json_schema_extra={
            "relevant_for": {
                "action": {"values": ["move_source_to_detection"], "required": False},
            }
        },
    )
    skip_if_unchanged: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Skip the OBS request when this block last applied the same value to the "
        "same target - a video workflow otherwise repeats `set_scene` every frame. Disable if "
        "something else changes OBS behind the block's back and it must re-apply each frame.",
        examples=[True],
        json_schema_extra={
            "relevant_for": {
                "action": {
                    "values": [
                        "set_scene",
                        "set_text",
                        "set_source_visibility",
                        "toggle_filter",
                    ],
                    "required": False,
                }
            }
        },
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Minimum number of seconds between two executions of this block. Leave at 0 "
        "when a Delta Filter or Rate Limiter upstream already shapes the trigger.",
        examples=[0, 2],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Send the request in the background. Faster, but `error_status` is then always "
        "`False` because the result is not awaited.",
        examples=[False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Disable the block without removing it from the Workflow.",
        examples=[False],
    )

    @model_validator(mode="after")
    def validate_required_fields_for_action(self) -> "BlockManifest":
        missing = [
            field
            for field in REQUIRED_FIELDS_BY_ACTION[self.action]
            if getattr(self, field) is None
        ]
        if missing:
            raise ValueError(
                f"OBS Action `{self.action}` requires the following field(s) to be set: "
                f"{', '.join(missing)}."
            )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [OBS_LAN_ACCESS_RESTRICTION, COOLDOWN_HTTP_SOFT_RESTRICTION]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.EXTERNAL_REQUEST]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return actual_restrictions_of(
            declared=[OBS_LAN_ACCESS_RESTRICTION, COOLDOWN_ACTUAL_RESTRICTION],
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


def _get_scene_item_id(client: Any, *, scene_name: str, source_name: str) -> int:
    response = client.call(
        "GetSceneItemId", data={"sceneName": scene_name, "sourceName": source_name}
    )
    item_id = response["sceneItemId"]

    return item_id


def _set_scene_item_enabled(
    client: Any, *, scene_name: str, item_id: int, enabled: bool
) -> None:
    client.call(
        "SetSceneItemEnabled",
        data={
            "sceneName": scene_name,
            "sceneItemId": item_id,
            "sceneItemEnabled": enabled,
        },
    )


def _perform_action(
    client: Any,
    action: str,
    scene_name: Optional[str],
    source_name: Optional[str],
    filter_name: Optional[str],
    text: Optional[str],
    enabled: Optional[bool],
    hotkey_name: Optional[str],
) -> str:
    if action == SET_SCENE:
        client.call("SetCurrentProgramScene", data={"sceneName": scene_name})
        return f"Switched OBS to scene '{scene_name}'"
    if action == SET_SOURCE_VISIBILITY:
        item_id = _get_scene_item_id(
            client, scene_name=scene_name, source_name=source_name
        )
        _set_scene_item_enabled(
            client, scene_name=scene_name, item_id=item_id, enabled=bool(enabled)
        )
        state = "visible" if enabled else "hidden"
        return f"Set source '{source_name}' in scene '{scene_name}' to {state}"
    if action == SET_TEXT:
        client.call(
            "SetInputSettings",
            data={
                "inputName": source_name,
                "inputSettings": {"text": str(text)},
                "overlay": True,
            },
        )
        return f"Updated text source '{source_name}'"
    if action == TOGGLE_FILTER:
        client.call(
            "SetSourceFilterEnabled",
            data={
                "sourceName": source_name,
                "filterName": filter_name,
                "filterEnabled": bool(enabled),
            },
        )
        state = "enabled" if enabled else "disabled"
        return f"{state.capitalize()} filter '{filter_name}' on source '{source_name}'"
    if action == TRIGGER_HOTKEY:
        client.call("TriggerHotkeyByName", data={"hotkeyName": hotkey_name})
        return f"Triggered OBS hotkey '{hotkey_name}'"
    if action == START_VIRTUAL_CAMERA:
        client.call("StartVirtualCam")
        return "Started OBS virtual camera"
    if action == STOP_VIRTUAL_CAMERA:
        client.call("StopVirtualCam")
        return "Stopped OBS virtual camera"
    if action == START_RECORDING:
        client.call("StartRecord")
        return "Started OBS recording"
    if action == STOP_RECORDING:
        client.call("StopRecord")
        return "Stopped OBS recording"
    raise ValueError(f"Unsupported OBS action: {action}")


class OBSActionBlockV1(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        disable_sinks: bool = False,
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._disable_sinks = disable_sinks
        self._last_action_fired: Optional[datetime] = None
        # Per-frame transforms would otherwise pay two lookup round trips per call.
        self._canvas_sizes: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._scene_item_ids: Dict[Tuple[str, int, str, str], int] = {}
        # last value applied per target, so an unchanged state costs no round trip
        self._last_applied: Dict[Tuple[Any, ...], Any] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor", "disable_sinks"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        connection: Dict[str, Any],
        action: str,
        scene_name: Optional[str],
        source_name: Optional[str],
        filter_name: Optional[str],
        text: Optional[str],
        enabled: Optional[bool],
        hotkey_name: Optional[str],
        cooldown_seconds: int,
        fire_and_forget: bool,
        disable_sink: bool,
        predictions: Optional[sv.Detections] = None,
        image: Optional[WorkflowImageData] = None,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        fit: str = "fit",
        hide_when_empty: bool = True,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        skip_if_unchanged: bool = True,
    ) -> BlockResult:
        if self._disable_sinks or disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": disabled_sink_message(
                    disabled_by_execution_policy=self._disable_sinks
                ),
            }
        if self._is_in_cooldown(cooldown_seconds=cooldown_seconds):
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        provided = {
            "scene_name": scene_name,
            "source_name": source_name,
            "filter_name": filter_name,
            "text": text,
            "enabled": enabled,
            "hotkey_name": hotkey_name,
            "predictions": predictions,
            "image": image,
            "position_x": position_x,
            "position_y": position_y,
            "width": width,
            "height": height,
        }
        # A selector can resolve to None at run time (e.g. a router that emitted nothing);
        # that means "no target this frame", not an error - hold the current state.
        missing = [
            field
            for field in REQUIRED_FIELDS_BY_ACTION[action]
            if provided.get(field) is None
        ]
        if missing and "predictions" not in missing:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": f"`{action}` skipped: no value for {', '.join(missing)}",
            }
        dedup_key, dedup_value = self._dedup_key(
            connection=connection,
            action=action,
            scene_name=scene_name,
            source_name=source_name,
            filter_name=filter_name,
            text=text,
            enabled=enabled,
        )
        if (
            skip_if_unchanged
            and dedup_key is not None
            and dedup_key in self._last_applied
            and self._last_applied[dedup_key] == dedup_value
        ):
            return {
                "error_status": False,
                "throttling_status": False,
                "message": f"`{action}` unchanged; skipped",
            }
        if action in (SET_SOURCE_TRANSFORM, MOVE_SOURCE_TO_DETECTION):
            operation = self._build_transform_operation(
                connection=connection,
                action=action,
                scene_name=scene_name,
                source_name=source_name,
                predictions=predictions,
                image=image,
                position_x=position_x,
                position_y=position_y,
                width=width,
                height=height,
                fit=fit,
                hide_when_empty=hide_when_empty,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        else:
            operation = partial(
                _perform_action,
                action=action,
                scene_name=scene_name,
                source_name=source_name,
                filter_name=filter_name,
                text=text,
                enabled=enabled,
                hotkey_name=hotkey_name,
            )
        action_handler = partial(
            call_with_reconnect,
            host=connection["host"],
            port=connection["port"],
            operation=operation,
            # older descriptors carried credentials; the registry covers the rest
            password=connection.get("password"),
            timeout=connection.get("timeout"),
        )

        def background_handler() -> None:
            # a failure on the thread pool would otherwise vanish without a trace
            try:
                action_handler()
            except Exception as error:  # noqa: BLE001 - logged, never raised
                self._last_applied.pop(dedup_key, None)
                logger.warning("Background OBS action `%s` failed: %s", action, error)

        self._last_action_fired = datetime.now()
        if dedup_key is not None:
            # optimistic for background sends; the handler drops it again on failure
            self._last_applied[dedup_key] = dedup_value
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(background_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "OBS action sent in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(background_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "OBS action sent in the background task",
            }
        try:
            message = action_handler()
            return {
                "error_status": False,
                "throttling_status": False,
                "message": message,
            }
        except Exception as error:  # noqa: BLE001 - surfaced through error_status
            self._last_applied.pop(dedup_key, None)
            return {
                "error_status": True,
                "throttling_status": False,
                "message": (
                    f"OBS action `{action}` failed: {error}"
                    + self._available_targets_hint(
                        connection=connection,
                        action=action,
                        scene_name=scene_name,
                        error=error,
                    )
                ),
            }

    BOUNDS_TYPE_BY_FIT = {
        "stretch": "OBS_BOUNDS_STRETCH",
        "fit": "OBS_BOUNDS_SCALE_INNER",
        "fill": "OBS_BOUNDS_SCALE_OUTER",
    }

    def _scene_item_id(
        self, client: Any, key: Tuple[str, int], scene_name: str, source_name: str
    ) -> int:
        item_key = (*key, scene_name, source_name)
        if item_key not in self._scene_item_ids:
            self._scene_item_ids[item_key] = _get_scene_item_id(
                client, scene_name=scene_name, source_name=source_name
            )
        return self._scene_item_ids[item_key]

    def _canvas_size(self, client: Any, key: Tuple[str, int]) -> Tuple[int, int]:
        if key not in self._canvas_sizes:
            settings = client.call("GetVideoSettings")
            self._canvas_sizes[key] = (settings["baseWidth"], settings["baseHeight"])
        return self._canvas_sizes[key]

    def _build_transform_operation(
        self,
        connection: Dict[str, Any],
        action: str,
        scene_name: str,
        source_name: str,
        predictions: Optional[sv.Detections],
        image: Optional[WorkflowImageData],
        position_x: Optional[float],
        position_y: Optional[float],
        width: Optional[float],
        height: Optional[float],
        fit: str,
        hide_when_empty: bool,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> Any:
        key = (connection["host"], connection["port"])
        bounds_type = self.BOUNDS_TYPE_BY_FIT[fit]

        def apply_transform(
            client: Any, rect: Tuple[float, float, float, float]
        ) -> None:
            item_id = self._scene_item_id(client, key, scene_name, source_name)
            x, y, target_width, target_height = rect
            try:
                client.call(
                    "SetSceneItemTransform",
                    data={
                        "sceneName": scene_name,
                        "sceneItemId": item_id,
                        "sceneItemTransform": {
                            "positionX": float(x),
                            "positionY": float(y),
                            "alignment": 5,  # top-left, so position is the rectangle's corner
                            "boundsType": bounds_type,
                            "boundsAlignment": 0,
                            "boundsWidth": max(1.0, float(target_width)),
                            "boundsHeight": max(1.0, float(target_height)),
                        },
                    },
                )
                _set_scene_item_enabled(
                    client, scene_name=scene_name, item_id=item_id, enabled=True
                )
            except Exception:
                # The cached item id may be stale (source recreated); refetch next call.
                self._scene_item_ids.pop((*key, scene_name, source_name), None)
                raise

        if action == SET_SOURCE_TRANSFORM:

            def operation(client: Any) -> str:
                apply_transform(client, (position_x, position_y, width, height))
                return (
                    f"Placed source '{source_name}' at ({position_x:.0f}, {position_y:.0f}) "
                    f"size {width:.0f}x{height:.0f}"
                )

            return operation

        def operation(client: Any) -> str:
            if detections_count(predictions) == 0:
                if not hide_when_empty:
                    return f"No detections; source '{source_name}' left unchanged"
                item_id = self._scene_item_id(client, key, scene_name, source_name)
                try:
                    _set_scene_item_enabled(
                        client, scene_name=scene_name, item_id=item_id, enabled=False
                    )
                except Exception:
                    self._scene_item_ids.pop((*key, scene_name, source_name), None)
                    raise
                return f"No detections; hid source '{source_name}'"
            canvas_width, canvas_height = self._canvas_size(client, key)
            image_height, image_width = image.numpy_image.shape[:2]
            boxes, confidences = detections_boxes_and_confidences(predictions)
            best = int(np.argmax(confidences))
            x_min, y_min, x_max, y_max = boxes[best]
            scale_x = canvas_width / image_width
            scale_y = canvas_height / image_height
            width = (x_max - x_min) * scale_x
            height = (y_max - y_min) * scale_y
            rect = (
                x_min * scale_x + offset_x * width,
                y_min * scale_y + offset_y * height,
                width,
                height,
            )
            apply_transform(client, rect)
            return (
                f"Moved source '{source_name}' to detection at canvas "
                f"({rect[0]:.0f}, {rect[1]:.0f}) size {rect[2]:.0f}x{rect[3]:.0f}"
            )

        return operation

    DEDUP_ACTIONS = (SET_SCENE, SET_TEXT, SET_SOURCE_VISIBILITY, TOGGLE_FILTER)

    @staticmethod
    def _dedup_key(
        connection: Dict[str, Any],
        action: str,
        scene_name: Optional[str],
        source_name: Optional[str],
        filter_name: Optional[str],
        text: Optional[str],
        enabled: Optional[bool],
    ) -> Tuple[Optional[Tuple[Any, ...]], Any]:
        """(target key, value) for state-setting actions; (None, None) for the rest."""
        base = (connection["host"], connection["port"], action)
        if action == SET_SCENE:
            return base, scene_name
        if action == SET_TEXT:
            return (*base, source_name), text
        if action == SET_SOURCE_VISIBILITY:
            return (*base, scene_name, source_name), bool(enabled)
        if action == TOGGLE_FILTER:
            return (*base, source_name, filter_name), bool(enabled)
        return None, None

    def _available_targets_hint(
        self,
        connection: Dict[str, Any],
        action: str,
        scene_name: Optional[str],
        error: Exception,
    ) -> str:
        """Append what OBS actually has when it rejected a name, so a typo is a one-shot fix."""
        if action not in NAMED_TARGET_ACTIONS:
            return ""

        if not isinstance(error, OBSRequestError) or error.code != RESOURCE_NOT_FOUND:
            return ""

        def describe(client: Any) -> str:
            # never let a diagnostic failure look like a dead socket (which would
            # trigger a reconnect) - a hint is best effort
            try:
                return _describe(client)
            except Exception:  # noqa: BLE001
                return ""

        def _describe(client: Any) -> str:
            scenes = [s["sceneName"] for s in client.call("GetSceneList")["scenes"]]
            parts = [f"Available scenes: {', '.join(scenes[:20])}"]
            if action != SET_SCENE and scene_name and scene_name in scenes:
                items = [
                    i["sourceName"]
                    for i in client.call(
                        "GetSceneItemList", data={"sceneName": scene_name}
                    )["sceneItems"]
                ]
                parts.append(f"sources in '{scene_name}': {', '.join(items[:20])}")
            elif action in (SET_TEXT, TOGGLE_FILTER):
                inputs = [i["inputName"] for i in client.call("GetInputList")["inputs"]]
                parts.append(f"available sources: {', '.join(inputs[:20])}")
            return ". " + "; ".join(parts)

        try:
            return call_with_reconnect(
                host=connection["host"],
                port=connection["port"],
                operation=describe,
            )
        except Exception:  # noqa: BLE001 - the hint must never mask the real error
            return ""

    def _is_in_cooldown(self, cooldown_seconds: int) -> bool:
        if cooldown_seconds <= 0 or self._last_action_fired is None:
            return False
        elapsed = (datetime.now() - self._last_action_fired).total_seconds()
        return elapsed < cooldown_seconds
