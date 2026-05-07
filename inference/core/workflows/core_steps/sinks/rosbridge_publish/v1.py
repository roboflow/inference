"""ROS topic publisher sink — v1.

Publishes Workflow outputs to ROS topics over the rosbridge JSON-over-WebSocket
protocol. No ROS dependency: the host process only needs ``roslibpy`` (an
optional extra) and a reachable ``rosbridge_server`` instance.

Built-in serializers cover the common Workflow output kinds — object
detection, classification, semantic / instance segmentation, keypoints,
images, and scalars — so users can wire a model block straight to this sink
without writing UQL. The escape hatch (``message_type=custom``) wraps an
arbitrary ``json_payload`` dict in ``std_msgs/String`` for shapes we don't
ship a serializer for.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.core_steps.common.rosbridge.connection import (
    RosHandle,
    get_registry,
    normalize_message_type,
)
from inference.core.workflows.core_steps.common.rosbridge.serializers import (
    SUPPORTED_MESSAGE_TYPES,
    OutboundMessage,
    SerializerContext,
    serialize,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Publish Workflow outputs to a ROS topic via rosbridge."

LONG_DESCRIPTION = """
The **Rosbridge Publish Sink** block publishes Workflow outputs to a ROS topic
over the [rosbridge JSON protocol](https://github.com/RobotWebTools/rosbridge_suite).
It does **not** require ROS itself to be installed in the inference container —
only the optional ``roslibpy`` Python client (``pip install 'inference[ros]'``)
and a reachable ``rosbridge_server`` instance on the robot side.

## Built-in serializers

Pick the matching ``message_type`` for the value you bind to ``payload``:

| ``message_type``                    | Accepts                                  |
|-------------------------------------|------------------------------------------|
| ``vision_msgs/Detection2DArray``    | object detection / instance seg boxes    |
| ``semantic_segmentation``           | semantic seg (mask + latched LabelInfo)  |
| ``instance_segmentation``           | instance seg (instance + class masks + boxes + LabelInfo) |
| ``vision_msgs/Classification``      | classification predictions               |
| ``keypoints``                       | keypoint detection (JSON + MarkerArray)  |
| ``sensor_msgs/CompressedImage``     | a Workflow image (jpeg-encoded)          |
| ``std_msgs/String``                 | strings / language-model output          |
| ``std_msgs/Int32``                  | integer scalar                           |
| ``std_msgs/Float64``                | float scalar                             |
| ``std_msgs/Bool``                   | boolean scalar                           |
| ``custom``                          | arbitrary dict — wrapped in ``std_msgs/String`` JSON |

Segmentation and keypoints publish to several **companion topics** under the
configured ``topic`` (``<topic>/instances``, ``<topic>/classes``,
``<topic>/detections``, ``<topic>/label_info``, ``<topic>/markers``) so
downstream consumers (RViz, Foxglove, custom nodes) can subscribe to whichever
shape they need.

## Connection sharing

All blocks (and the rosbridge image source) targeting the same
``(host, port, ssl)`` triple share a single WebSocket. Topics are advertised
at most once per process per ``(topic, message_type)`` pair.

## Cooldown / fire-and-forget

Same semantics as the Webhook sink — see ``cooldown_seconds`` and
``fire_and_forget``.
"""


PAYLOAD_KINDS = [
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    IMAGE_KIND,
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    BOOLEAN_KIND,
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Rosbridge Publish Sink",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-robot",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/rosbridge_publish_sink@v1"]

    host: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Hostname or IP of the rosbridge_server.",
        examples=["robot.local", "$inputs.rosbridge_host"],
    )
    port: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=9090,
        description="rosbridge_server WebSocket port.",
        examples=[9090],
    )
    ssl: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=False,
        description="Use wss:// instead of ws://.",
        examples=[False],
    )
    topic: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="ROS topic to publish on (with or without leading slash).",
        examples=["/inference/detections"],
    )
    message_type: Literal[
        "vision_msgs/Detection2DArray",
        "semantic_segmentation",
        "instance_segmentation",
        "vision_msgs/Classification",
        "keypoints",
        "sensor_msgs/CompressedImage",
        "std_msgs/String",
        "std_msgs/Int32",
        "std_msgs/Float64",
        "std_msgs/Bool",
        "custom",
    ] = Field(
        description=(
            "Built-in serializer to use. Pick the entry matching the kind of "
            "value bound to `payload`. Use `custom` to pass `json_payload` "
            "through unchanged in `std_msgs/String`."
        ),
    )
    payload: Optional[Selector(kind=PAYLOAD_KINDS)] = Field(
        default=None,
        description="Workflow output to publish. Required for all built-in serializers.",
        examples=["$steps.model.predictions"],
    )
    json_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Custom payload dict — only used when `message_type=custom`. "
            "Same UQL semantics as the Webhook sink."
        ),
        examples=[{"detections": "$steps.model.predictions"}],
    )
    json_payload_operations: Dict[str, List[AllOperationsType]] = Field(
        default_factory=dict,
        description="UQL operations applied to `json_payload` values.",
    )
    frame_id: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="inference",
        description="Value placed into ROS message Header.frame_id fields.",
        examples=["camera_optical_frame"],
    )
    ros_version: Literal[1, 2] = Field(
        default=2,
        description="ROS version of the bridge — affects type-string and time-field shapes.",
    )
    latch: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=False,
        description=(
            "Publish with latch / transient_local durability. Always implied "
            "for the LabelInfo companion topic on segmentation outputs."
        ),
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description=(
            "Minimum seconds between publishes. 0 = no cooldown. Cooldown "
            "throttling only applies in InferencePipeline streaming context."
        ),
        json_schema_extra={"always_visible": True},
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description=(
            "Run the publish in a background thread/task. Set to False for "
            "synchronous error reporting."
        ),
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Skip publishing entirely (e.g. via a runtime flag).",
    )

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=True, reason=None)

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RosbridgePublishSinkBlockV1(WorkflowBlock):
    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._last_notification_fired: Optional[datetime] = None
        self._handle: Optional[RosHandle] = None
        self._handle_key: Optional[Tuple[str, int, bool]] = None
        self._advertised: Dict[Tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        host: str,
        port: int,
        ssl: bool,
        topic: str,
        message_type: str,
        payload: Any,
        json_payload: Dict[str, Any],
        json_payload_operations: Dict[str, List[AllOperationsType]],
        frame_id: str,
        ros_version: int,
        latch: bool,
        cooldown_seconds: int,
        fire_and_forget: bool,
        disable_sink: bool,
    ) -> BlockResult:
        if disable_sink:
            return _result(False, False, "Sink disabled by `disable_sink`.")

        if message_type not in SUPPORTED_MESSAGE_TYPES:
            return _result(
                True,
                False,
                f"Unsupported message_type: {message_type!r}",
            )

        cooldown = self._check_cooldown(cooldown_seconds)
        if cooldown is not None:
            return cooldown

        if message_type == "custom":
            value = _execute_operations_on_parameters(
                json_payload, json_payload_operations
            )
        else:
            if payload is None:
                return _result(
                    True, False, "`payload` is required for built-in serializers."
                )
            value = payload

        ctx = SerializerContext(frame_id=frame_id, ros_version=ros_version)
        try:
            envelopes = serialize(message_type, value, ctx)
        except Exception as error:
            logging.exception("rosbridge serialize failed")
            return _result(True, False, f"Serialization failed: {error}")

        topic_root = _normalize_topic(topic)
        publish = _make_publisher(
            block=self,
            host=host,
            port=int(port),
            ssl=bool(ssl),
            topic_root=topic_root,
            envelopes=envelopes,
            ros_version=int(ros_version),
            latch=bool(latch),
        )

        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks is not None:
            self._background_tasks.add_task(publish)
            return _result(False, False, "Publish dispatched to background task.")
        if fire_and_forget and self._thread_pool_executor is not None:
            self._thread_pool_executor.submit(publish)
            return _result(False, False, "Publish dispatched to thread pool.")
        error_status, message = publish()
        return _result(error_status, False, message)

    def _check_cooldown(self, cooldown_seconds: int) -> Optional[BlockResult]:
        if cooldown_seconds <= 0 or self._last_notification_fired is None:
            return None
        elapsed = (datetime.now() - self._last_notification_fired).total_seconds()
        if elapsed < cooldown_seconds:
            return _result(False, True, "Sink cooldown applies.")
        return None

    def _ensure_connection(self, host: str, port: int, ssl: bool) -> Any:
        key = (host, port, ssl)
        with self._lock:
            if self._handle is None or self._handle_key != key:
                if self._handle is not None:
                    self._handle.release()
                    self._advertised.clear()
                self._handle = get_registry().acquire(host=host, port=port, ssl=ssl)
                self._handle_key = key
            return self._handle.ros

    def _ensure_topic(
        self,
        ros: Any,
        topic_name: str,
        message_type: str,
        latch: bool,
    ) -> Any:
        try:
            import roslibpy  # type: ignore
        except ImportError as e:
            raise ImportError(
                "roslibpy is required for the rosbridge publish sink. "
                "Install with: pip install 'inference[ros]'"
            ) from e
        key = (topic_name, message_type)
        with self._lock:
            handle = self._advertised.get(key)
            if handle is None:
                handle = roslibpy.Topic(
                    ros,
                    name=topic_name,
                    message_type=message_type,
                    latch=latch,
                )
                handle.advertise()
                self._advertised[key] = handle
            return handle

    def _publish_envelopes(
        self,
        host: str,
        port: int,
        ssl: bool,
        topic_root: str,
        envelopes: List[OutboundMessage],
        ros_version: int,
        latch: bool,
    ) -> Tuple[bool, str]:
        try:
            import roslibpy  # type: ignore
        except ImportError as e:
            return True, str(e)
        try:
            ros = self._ensure_connection(host, port, ssl)
            for env in envelopes:
                topic_name = topic_root + env.topic_suffix
                msg_type = normalize_message_type(env.message_type, ros_version)
                topic_handle = self._ensure_topic(
                    ros=ros,
                    topic_name=topic_name,
                    message_type=msg_type,
                    latch=latch or env.latch,
                )
                topic_handle.publish(roslibpy.Message(env.payload))
            return False, f"Published {len(envelopes)} message(s) to {topic_root}."
        except Exception as error:
            logging.exception("rosbridge publish failed")
            return True, f"Publish failed: {error}"

    def __del__(self):
        try:
            with self._lock:
                if self._handle is not None:
                    for handle in self._advertised.values():
                        try:
                            handle.unadvertise()
                        except Exception:
                            pass
                    self._advertised.clear()
                    self._handle.release()
                    self._handle = None
        except Exception:
            pass


def _make_publisher(
    block: RosbridgePublishSinkBlockV1,
    host: str,
    port: int,
    ssl: bool,
    topic_root: str,
    envelopes: List[OutboundMessage],
    ros_version: int,
    latch: bool,
):
    def _do_publish() -> Tuple[bool, str]:
        return block._publish_envelopes(
            host=host,
            port=port,
            ssl=ssl,
            topic_root=topic_root,
            envelopes=envelopes,
            ros_version=ros_version,
            latch=latch,
        )

    return _do_publish


def _execute_operations_on_parameters(
    parameters: Dict[str, Any],
    operations: Dict[str, List[AllOperationsType]],
) -> Dict[str, Any]:
    parameters = copy(parameters)
    for parameter_name, ops in operations.items():
        if not ops or parameter_name not in parameters:
            continue
        parameters[parameter_name] = build_operations_chain(operations=ops)(
            parameters[parameter_name], global_parameters={}
        )
    return parameters


def _normalize_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        raise ValueError("topic must be non-empty")
    if not topic.startswith("/"):
        topic = "/" + topic
    return topic.rstrip("/") or "/"


def _result(error_status: bool, throttled: bool, message: str) -> BlockResult:
    return {
        "error_status": error_status,
        "throttling_status": throttled,
        "message": message,
    }
