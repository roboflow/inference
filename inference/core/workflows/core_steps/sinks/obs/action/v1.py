from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, model_validator

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.sinks.noop import disabled_sink_message
from inference.core.workflows.core_steps.sinks.obs.client import call_with_reconnect
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
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
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

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

Requires a reachable OBS Studio instance with its websocket server enabled, and the `obsws-python`
package installed in the environment running `inference`. Not available on Roboflow Hosted
Serverless or Dedicated Deployments, which cannot reach a local OBS instance.
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
                "icon": "fa-brands fa-obs-studio",
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
        return [
            RuntimeRestriction(
                severity=Severity.HARD,
                note=(
                    "Block requires network access to an OBS Studio instance, which normally "
                    "runs on the same machine as the Workflow. Hosted Serverless and Roboflow "
                    "Dedicated Deployments cannot reach a local OBS websocket server."
                ),
                applies_to_runtimes=[
                    Runtime.HOSTED_SERVERLESS,
                    Runtime.DEDICATED_DEPLOYMENT,
                ],
            )
        ]


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
        client.set_current_program_scene(scene_name)
        return f"Switched OBS to scene '{scene_name}'"
    if action == SET_SOURCE_VISIBILITY:
        item_id = client.get_scene_item_id(scene_name, source_name).scene_item_id
        client.set_scene_item_enabled(scene_name, item_id, bool(enabled))
        state = "visible" if enabled else "hidden"
        return f"Set source '{source_name}' in scene '{scene_name}' to {state}"
    if action == SET_TEXT:
        client.set_input_settings(source_name, {"text": str(text)}, True)
        return f"Updated text source '{source_name}'"
    if action == TOGGLE_FILTER:
        client.set_source_filter_enabled(source_name, filter_name, bool(enabled))
        state = "enabled" if enabled else "disabled"
        return f"{state.capitalize()} filter '{filter_name}' on source '{source_name}'"
    if action == TRIGGER_HOTKEY:
        client.trigger_hot_key_by_name(hotkey_name)
        return f"Triggered OBS hotkey '{hotkey_name}'"
    if action == START_VIRTUAL_CAMERA:
        client.start_virtual_cam()
        return "Started OBS virtual camera"
    if action == STOP_VIRTUAL_CAMERA:
        client.stop_virtual_cam()
        return "Stopped OBS virtual camera"
    if action == START_RECORDING:
        client.start_record()
        return "Started OBS recording"
    if action == STOP_RECORDING:
        client.stop_record()
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
            password=connection.get("password"),
            timeout=connection.get("timeout", 3),
            operation=operation,
        )
        self._last_action_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(action_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "OBS action sent in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(action_handler)
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
            return {
                "error_status": True,
                "throttling_status": False,
                "message": f"OBS action `{action}` failed: {error}",
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
            self._scene_item_ids[item_key] = client.get_scene_item_id(
                scene_name, source_name
            ).scene_item_id
        return self._scene_item_ids[item_key]

    def _canvas_size(self, client: Any, key: Tuple[str, int]) -> Tuple[int, int]:
        if key not in self._canvas_sizes:
            settings = client.get_video_settings()
            self._canvas_sizes[key] = (settings.base_width, settings.base_height)
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
    ) -> Any:
        key = (connection["host"], connection["port"])
        bounds_type = self.BOUNDS_TYPE_BY_FIT[fit]

        def apply_transform(
            client: Any, rect: Tuple[float, float, float, float]
        ) -> None:
            item_id = self._scene_item_id(client, key, scene_name, source_name)
            x, y, target_width, target_height = rect
            try:
                client.set_scene_item_transform(
                    scene_name,
                    item_id,
                    {
                        "positionX": float(x),
                        "positionY": float(y),
                        "alignment": 5,  # top-left, so position is the rectangle's corner
                        "boundsType": bounds_type,
                        "boundsAlignment": 0,
                        "boundsWidth": max(1.0, float(target_width)),
                        "boundsHeight": max(1.0, float(target_height)),
                    },
                )
                client.set_scene_item_enabled(scene_name, item_id, True)
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
            if predictions is None or len(predictions) == 0:
                if not hide_when_empty:
                    return f"No detections; source '{source_name}' left unchanged"
                item_id = self._scene_item_id(client, key, scene_name, source_name)
                try:
                    client.set_scene_item_enabled(scene_name, item_id, False)
                except Exception:
                    self._scene_item_ids.pop((*key, scene_name, source_name), None)
                    raise
                return f"No detections; hid source '{source_name}'"
            canvas_width, canvas_height = self._canvas_size(client, key)
            image_height, image_width = image.numpy_image.shape[:2]
            best = int(np.argmax(predictions.confidence))
            x_min, y_min, x_max, y_max = predictions.xyxy[best]
            scale_x = canvas_width / image_width
            scale_y = canvas_height / image_height
            rect = (
                x_min * scale_x,
                y_min * scale_y,
                (x_max - x_min) * scale_x,
                (y_max - y_min) * scale_y,
            )
            apply_transform(client, rect)
            return (
                f"Moved source '{source_name}' to detection at canvas "
                f"({rect[0]:.0f}, {rect[1]:.0f}) size {rect[2]:.0f}x{rect[3]:.0f}"
            )

        return operation

    def _is_in_cooldown(self, cooldown_seconds: int) -> bool:
        if cooldown_seconds <= 0 or self._last_action_fired is None:
            return False
        elapsed = (datetime.now() - self._last_action_fired).total_seconds()
        return elapsed < cooldown_seconds
