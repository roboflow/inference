import logging
import math
import numbers
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import requests
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from roboflow_workflows.core_steps.common.keypoints import real_keypoints_count
from roboflow_workflows.core_steps.common.serializers import mask_to_polygon
from roboflow_workflows.core_steps.common.tensor_native import (
    KeyPointPrediction,
    instance_mask_to_numpy,
    split_key_point_prediction,
)
from roboflow_workflows.core_steps.common.workload_presets import (
    COOLDOWN_ACTUAL_RESTRICTION,
)
from roboflow_workflows.core_steps.sinks.noop import disabled_sink_message
from roboflow_workflows.core_steps.sinks.roboflow.vision_events.clip_attachments import (
    DEFAULT_CLIP_UPLOAD_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_LABEL,
    MAX_CLIP_UPLOAD_TIMEOUT_SECONDS,
    STATUS_DISABLED,
    STATUS_THROTTLED,
    STATUS_UNSUPPORTED_SINK_MODE,
    ClipAttachments,
    SinkClipUploads,
    plan_attachments,
    redact_signed_urls,
    skipped_attachments,
)
from roboflow_workflows.core_steps.sinks.roboflow.vision_events.v1 import (
    _api_key_safe_raise_for_status,
    _build_cloud_request_headers,
)
from roboflow_workflows.environment import API_BASE_URL
from roboflow_workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from roboflow_workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_SOLUTION_KIND,
    STRING_KIND,
    VIDEO_CLIP_ANCHOR_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.background_tasks import BackgroundTaskScheduler
from roboflow_workflows.prototypes.block import (
    COOLDOWN_HTTP_SOFT_RESTRICTION,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
)
from roboflow_workflows.prototypes.platform_client import (
    OFFLINE_PLATFORM_CLIENT,
    RoboflowPlatformClient,
    ensure_roboflow_api_access,
    resolve_platform_blocks_client,
)
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

logger = logging.getLogger(__name__)

# Tensor-native prediction union the block accepts. Detection predictions arrive
# as `inference_models` dataclasses (or the keypoint `(KeyPoints, Detections)`
# tuple) and classification predictions as the native classification dataclasses,
# rather than the `sv.Detections` / dict the numpy sibling consumed.
TensorNativePrediction = Union[
    Detections,
    InstanceDetections,
    KeyPointPrediction,
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
]

VALID_EVENT_TYPES = [
    "quality_check",
    "inventory_count",
    "safety_alert",
    "custom",
    "operator_feedback",
]

QUALITY_CHECK_RELEVANT = {
    "event_type": {"values": ["quality_check"], "required": True},
}
INVENTORY_COUNT_RELEVANT = {
    "event_type": {"values": ["inventory_count"], "required": True},
}
SAFETY_ALERT_RELEVANT = {
    "event_type": {"values": ["safety_alert"], "required": True},
}
CUSTOM_RELEVANT = {
    "event_type": {"values": ["custom"], "required": True},
}
OPERATOR_FEEDBACK_RELEVANT = {
    "event_type": {"values": ["operator_feedback"], "required": True},
}
ALL_DATA_SCHEMAS_RELEVANT = {
    "event_type": {
        "values": [
            "quality_check",
            "inventory_count",
            "safety_alert",
            "custom",
        ],
    },
}

# The cloud's bound on an event value (+/-$100M per event in USD). The local
# event store applies the same bound and rejects the whole event past it.
MAX_ABS_VALUE_BASE_UNITS = 10_000_000_000

# Parses a string event value with the manifest's own lax ``int`` rule, so a
# workflow input such as "1999" that passed input validation is sent, not
# dropped (the block receives the raw input, not the validated one).
_WHOLE_NUMBER_STRING = TypeAdapter(int)

SHORT_DESCRIPTION = "Send vision events to the Roboflow Vision Events API."

LONG_DESCRIPTION = """
Send images, model predictions, and event metadata to the Roboflow Vision Events API for
monitoring, quality control, safety alerting, and custom event tracking.

## How This Block Works

This block uploads workflow images and model predictions to the Roboflow Vision Events API,
creating structured events that can be queried, filtered, and visualized in the Roboflow
dashboard.

1. Optionally uploads an input image and/or output image (visualization) to the Vision Events
   image storage via the public API
2. Converts model predictions (object detection, classification, instance segmentation, or
   keypoint detection) into the Vision Events annotation format and attaches them to the
   input image
3. Creates a vision event with the specified event type, use case, event data,
   and custom metadata
4. Enforces a built-in rate limit (`cooldown_seconds`, default 1 second) so
   high-frequency video workflows do not flood the API with an event per frame
5. Supports fire-and-forget mode for non-blocking execution

## Rate Limiting

Video workflows can run many times per second, which by default would send an event
(and its images) for every frame. To prevent this, the block enforces a cooldown
between consecutive events: at most one event per second is sent by default. Events
triggered during the cooldown period are dropped and the `throttling_status` output
is set to `True`.

Adjust `cooldown_seconds` to your needs, or set it to `0` to disable rate limiting
entirely (e.g. for intentionally bursty use cases). The cooldown timer lives in the
block instance, so it throttles long-lived executions such as video processing with
InferencePipeline. Workflows served over HTTP (e.g. `/workflows/run`) create fresh
block instances per request, so the cooldown does not throttle across separate HTTP
calls.

## Deployment Modes

By default this block sends events to the **Roboflow Vision Events API** (cloud /
Serverless API), uploading images and posting the event over the public API.

For edge deployments, enable **Write to Local Event Store** to send events to a
local Event Ingestion Service instead. In this mode images are embedded directly
in the request (no upload step) and the event is posted to `<event store URL>/v2/events`.
The event store URL defaults to `http://localhost:8001` and can be overridden. No
Roboflow API key is required in this mode; if the local service requires
authentication, set the `EVENT_INGESTION_API_KEY` environment variable on the
inference server.

## Event Types

- **quality_check**: Manufacturing/inspection QA with pass/fail result and optional confidence
- **inventory_count**: Inventory tracking with location, item count, and item type
- **safety_alert**: Safety violations with alert type, severity (low/medium/high), and description
- **custom**: User-defined events with a free-form value string
- **operator_feedback**: Operator review/correction of previous events (correct/incorrect/inconclusive)

## Requirements

The default (cloud) mode requires a valid Roboflow API key with `vision-events:write`
scope, configured in your environment or workflow configuration. No Roboflow API key is
needed when **Write to Local Event Store** is enabled (see Deployment Modes above).

## Video Clips

Wire one or more **Video Clip Recorder** anchors into `video_clips` to attach short
video clips to the event (cloud mode only). The clip window and the caption come
from each recorder (`pre_roll_seconds`, `post_roll_seconds`, `label`); `video_label`
is used only when a recorder's label is empty. Up to 4 clips are attached in the
order given; extra anchors are reported as `rejected:too_many_videos`.

The event is created right away and references each clip as a pending video. The
clip is uploaded in the background as soon as its post-roll has been recorded, so
the `videos` output reports each clip's immediate `video_status` (`pending`,
`disabled`, `throttled`, `unavailable:*`, `rejected:*` or `prepare_failed`), never
a finished upload. Each clip's outcome is logged as a `vision_event_clip_completed`
record; enable `confirm_ingestion` to also wait for the platform to confirm it.
In fire-and-forget mode `video_id` is empty because the upload slot is requested
in the background. Disabled or throttled events capture nothing, and the local
event store mode does not support clips (`unavailable:unsupported_sink_mode`).

Keep `fire_and_forget` enabled when attaching clips to a video stream: with it
disabled, each clip's upload slot request (about 0.3 s each in validation) and the
event POST run on the pipeline thread, which stops processing frames meanwhile.

The completion record carries the event and video ids, the label, a
`terminal_status` (`put_accepted`, `upload_failed`, `expired`, `clip_failed`,
`cancelled` or `shutdown_deadline`), the clip report (bounds, frame counts and
missing frames by cause), timings and, with `confirm_ingestion`, the ingestion
result (`confirmed`, `pending` or `failed`). `put_accepted` means the upload was
accepted, not that the platform finished ingesting it. Transient upload errors are
retried until `clip_upload_timeout_seconds` after the upload slot was requested.

## Common Use Cases

- **Quality Control**: Automatically log inspection results with images and detection overlays
- **Safety Monitoring**: Send safety alerts when violations are detected in video streams
- **Production Analytics**: Track inventory counts and production metrics with visual evidence
- **Active Monitoring**: Fire-and-forget event logging from real-time video processing workflows
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Vision Events",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-eye",
                "blockPriority": 1,
                "popular": False,
                # Not unconditionally required: the local event store mode needs no
                # Roboflow key. A True value here walls off the whole block config in
                # the inference-embedded (edge) editor, which would defeat that mode.
                # The cloud path still raises at runtime if no key is available.
                "requires_rf_key": False,
            },
        }
    )
    type: Literal["roboflow_core/roboflow_vision_events@v1"]
    input_image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        title="Input Image",
        description="The original input image. Uploaded to the Vision Events API and "
        "used as the base image for detection annotations.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        json_schema_extra={"always_visible": True},
    )
    output_image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        title="Output Image",
        description="An optional output/visualized image (e.g., from a visualization "
        "block). Displayed as the primary image in the Vision Events dashboard.",
        examples=["$steps.visualization.image"],
        json_schema_extra={"always_visible": True},
    )
    predictions: Optional[
        Selector(
            kind=[
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        title="Predictions",
        description="Optional model predictions to include as detection annotations on "
        "the input image. Supports object detection, instance segmentation, keypoint "
        "detection, and classification predictions.",
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={"always_visible": True},
    )
    event_type: Union[
        Literal[
            "quality_check",
            "inventory_count",
            "safety_alert",
            "custom",
            "operator_feedback",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        title="Event Type",
        description="The type of vision event to create.",
        examples=["quality_check", "custom", "$inputs.event_type"],
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "quality_check": {
                    "name": "Quality Check",
                    "description": "Manufacturing/inspection QA with pass/fail result and optional confidence",
                },
                "inventory_count": {
                    "name": "Inventory Count",
                    "description": "Inventory tracking with location, item count, and item type",
                },
                "safety_alert": {
                    "name": "Safety Alert",
                    "description": "Safety violations with alert type, severity, and description",
                },
                "custom": {
                    "name": "Custom",
                    "description": "User-defined events with a free-form value string",
                },
                "operator_feedback": {
                    "name": "Operator Feedback",
                    "description": "Operator review/correction of previous events",
                },
            },
        },
    )
    solution: Union[str, Selector(kind=[ROBOFLOW_SOLUTION_KIND, STRING_KIND])] = Field(
        title="Use Case",
        description="The use case to associate the event with. Events are "
        "namespaced by use case within a workspace.",
        examples=["my-use-case", "$inputs.use_case"],
    )
    # --- External ID (shared across schemas) ---
    external_id: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="External ID",
        description="External identifier for correlation with other systems (max 1000 chars).",
        examples=["batch-2025-001", "$inputs.external_id"],
        json_schema_extra={
            "relevant_for": ALL_DATA_SCHEMAS_RELEVANT,
        },
    )

    # --- Event value (shared across schemas) ---
    value_base_units: Optional[
        Union[Selector(kind=[INTEGER_KIND, FLOAT_KIND]), int]
    ] = Field(
        default=None,
        title="Event Value",
        description="What this event is worth, in base units of the use case's "
        "currency, e.g. cents for USD (1999 = $19.99). The currency always comes "
        "from the use case. Leave empty to use the use case's default value, if it "
        "has one. Negative values record a cost. A typed value or workflow input "
        "must be a whole number; a value from another step is rounded to the "
        "nearest base unit. Values outside -10,000,000,000 to 10,000,000,000 are "
        "ignored.",
        examples=[1999, "$steps.expression.output"],
        json_schema_extra={"relevant_for": ALL_DATA_SCHEMAS_RELEVANT},
    )

    # --- Quality Check fields ---
    qc_result: Optional[
        Union[Selector(kind=[STRING_KIND]), Literal["pass", "fail"]]
    ] = Field(
        default=None,
        title="Result",
        description="Quality check result: pass or fail.",
        examples=["pass", "fail", "$steps.qc_logic.result"],
        json_schema_extra={
            "relevant_for": QUALITY_CHECK_RELEVANT,
            "always_visible": True,
        },
    )

    # --- Inventory Count fields ---
    location: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="Location",
        description="Location identifier for inventory count.",
        examples=["warehouse-A", "$inputs.location"],
        json_schema_extra={"relevant_for": INVENTORY_COUNT_RELEVANT},
    )
    item_count: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=None,
        title="Item Count",
        description="Number of items counted.",
        examples=[42, "$steps.counter.count"],
        json_schema_extra={
            "relevant_for": INVENTORY_COUNT_RELEVANT,
            "always_visible": True,
        },
    )
    item_type: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="Item Type",
        description="Type of item being counted.",
        examples=["widget", "$inputs.item_type"],
        json_schema_extra={"relevant_for": INVENTORY_COUNT_RELEVANT},
    )

    # --- Safety Alert fields ---
    alert_type: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="Alert Type",
        description="Alert type identifier (e.g. no_hardhat, spill_detected).",
        examples=["no_hardhat", "$steps.classifier.top_class"],
        json_schema_extra={
            "relevant_for": SAFETY_ALERT_RELEVANT,
            "always_visible": True,
        },
    )
    severity: Optional[
        Union[Selector(kind=[STRING_KIND]), Literal["low", "medium", "high"]]
    ] = Field(
        default=None,
        title="Severity",
        description="Severity level for the safety alert.",
        examples=["high", "$inputs.severity"],
        json_schema_extra={"relevant_for": SAFETY_ALERT_RELEVANT},
    )
    alert_description: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="Description",
        description="Description of the safety alert.",
        examples=["Worker detected without hardhat in zone B"],
        json_schema_extra={"relevant_for": SAFETY_ALERT_RELEVANT},
    )

    # --- Custom Event fields ---
    custom_value: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="Value",
        description="Arbitrary value for custom events.",
        examples=["anomaly detected at 14:32"],
        json_schema_extra={
            "relevant_for": CUSTOM_RELEVANT,
            "always_visible": True,
        },
    )

    # --- Operator Feedback fields ---
    related_event_id: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="Related Event ID",
        description="The event ID of the event being reviewed.",
        examples=["evt_abc123", "$inputs.related_event_id"],
        json_schema_extra={
            "relevant_for": OPERATOR_FEEDBACK_RELEVANT,
            "always_visible": True,
        },
    )
    feedback: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Literal["correct", "incorrect", "inconclusive"],
        ]
    ] = Field(
        default=None,
        title="Feedback",
        description="Operator feedback on the related event.",
        examples=["correct", "incorrect", "$inputs.feedback"],
        json_schema_extra={"relevant_for": OPERATOR_FEEDBACK_RELEVANT},
    )

    custom_metadata: Dict[str, Union[str, int, float, bool, Selector()]] = Field(
        default_factory=dict,
        title="Custom Metadata",
        description="Flat key-value metadata to attach to the event. Keys must match "
        "pattern [a-zA-Z0-9_ -]+ (max 100 chars). String values max 1000 chars.",
        examples=[{"camera_id": "cam_01", "location": "$inputs.location"}],
        json_schema_extra={"additional_section": True},
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        title="Fire and Forget",
        description="If True, the event is sent asynchronously and the workflow "
        "continues without waiting. If False, the block waits for the API response.",
        examples=[True, "$inputs.fire_and_forget"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        title="Disable Sink",
        description="If True, the block is disabled and no events are sent.",
        examples=[False, "$inputs.disable_vision_events"],
    )
    cooldown_seconds: Union[
        NonNegativeInt, NonNegativeFloat, Selector(kind=[INTEGER_KIND, FLOAT_KIND])
    ] = Field(
        default=1,
        title="Cooldown",
        description="Minimum number of seconds between consecutive events sent by "
        "this block. Events triggered during the cooldown period are dropped and "
        "the `throttling_status` output is set to True. Defaults to 1 second (at "
        "most 1 event per second) so high-frequency video workflows do not flood "
        "the Vision Events API with an event per frame. Set to 0 to disable rate "
        "limiting for intentionally bursty use cases.",
        examples=[1, 0.5, "$inputs.cooldown_seconds"],
        json_schema_extra={"always_visible": True},
    )
    write_to_event_store: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        title="Write to Local Event Store",
        description="If True, send the event to a local Event Ingestion Service "
        "(edge deployment) instead of the Roboflow Vision Events API (cloud). "
        "Images are embedded in the request and the event is posted to "
        "`<Event Store URL>/v2/events`. No Roboflow API key is required in this mode.",
        examples=[False, True, "$inputs.write_to_event_store"],
    )
    event_store_url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="http://localhost:8001",
        title="Event Store URL",
        description="Base URL of the local Event Ingestion Service. Only used when "
        "`Write to Local Event Store` is enabled.",
        examples=["http://localhost:8001", "$inputs.event_store_url"],
        json_schema_extra={
            "relevant_for": {
                "write_to_event_store": {
                    "values": [True],
                    "required": False,
                },
            },
        },
    )

    video_clips: Optional[
        Union[
            Selector(kind=[VIDEO_CLIP_ANCHOR_KIND]),
            List[Selector(kind=[VIDEO_CLIP_ANCHOR_KIND])],
        ]
    ] = Field(
        default=None,
        title="Video Clips",
        description="One Video Clip Recorder anchor, or a list of up to 4, whose "
        "clips are attached to the event in the given order. Window and caption "
        "come from each recorder. Cloud mode only.",
        examples=["$steps.video_clip_recorder.anchor"],
        json_schema_extra={
            "relevant_for": {
                "write_to_event_store": {"values": [False], "required": False},
            },
        },
    )
    video_label: str = Field(
        default=DEFAULT_VIDEO_LABEL,
        title="Video Label",
        description="Caption used for a clip whose recorder has an empty label.",
        examples=["Evidence"],
        min_length=1,
        max_length=256,
        json_schema_extra={"additional_section": True},
    )
    confirm_ingestion: bool = Field(
        default=False,
        title="Confirm Ingestion",
        description="If True, after a clip is uploaded the block polls the platform "
        "for up to 5 minutes and reports in the clip's completion record whether "
        "the video was ingested.",
        examples=[False, True],
        json_schema_extra={"additional_section": True},
    )
    clip_upload_timeout_seconds: float = Field(
        default=DEFAULT_CLIP_UPLOAD_TIMEOUT_SECONDS,
        title="Clip Upload Timeout",
        description="Seconds, counted from the upload slot request, within which a "
        "clip must be uploaded; the upload URL expires after 1 hour.",
        examples=[3300.0],
        gt=0,
        le=MAX_CLIP_UPLOAD_TIMEOUT_SECONDS,
        json_schema_extra={"additional_section": True},
    )

    @field_validator("value_base_units", mode="before")
    @classmethod
    def reject_boolean_value_base_units(cls, value: Any) -> Any:
        # Lax int validation would turn true/false into 1/0 and record a value (or
        # replace the use case default with 0) instead of rejecting the mistake.
        if isinstance(value, bool):
            raise ValueError("value_base_units must be a number, not true or false")
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="event_id", kind=[STRING_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
            OutputDefinition(name="videos", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [COOLDOWN_HTTP_SOFT_RESTRICTION]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.EXTERNAL_REQUEST, WorkOperation.IMAGE_ENCODING]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare the cooldown-timer state-loss caveat for the target deployment.

        Args:
            ignore_environment_restrictions: If True, return every declaration
                with its condition intact (the portable view). If False,
                evaluate configuration predicates against this host and drop
                entries that definitively do not apply here.

        Returns:
            The step's restrictions. In the host view the discovery is
            incomplete when a configuration predicate cannot be evaluated.
        """
        return actual_restrictions_of(
            declared=[COOLDOWN_ACTUAL_RESTRICTION],
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        # Posts an event to the platform or to a local event store; `solution` is
        # a Vision Events use case identifier, not a project - no Roboflow model,
        # no Roboflow project, no third-party model.
        return []


class RoboflowVisionEventsBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTaskScheduler],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        disable_sinks: bool = False,
        platform_client: RoboflowPlatformClient = OFFLINE_PLATFORM_CLIENT,
    ):
        self._api_key = api_key
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._disable_sinks = disable_sinks
        self._platform_client = resolve_platform_blocks_client(platform_client)
        self._last_event_fired: Optional[datetime] = None
        self._clip_uploads = SinkClipUploads()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "api_key",
            "background_tasks",
            "thread_pool_executor",
            "disable_sinks",
            "platform_client",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def prepare_close(self, deadline_seconds: Optional[float] = None) -> None:
        """Stop taking new clips and record the host's close deadline.

        Args:
            deadline_seconds (Optional[float]): Seconds the host allows for the
                whole close; None uses the default deadline.
        """
        self._clip_uploads.prepare_close(deadline_seconds)

    def close(self) -> None:
        """Finish clip uploads within the close deadline, then stop the uploader."""
        self._clip_uploads.close()

    def run(
        self,
        input_image: Optional[WorkflowImageData],
        output_image: Optional[WorkflowImageData],
        predictions: Optional[TensorNativePrediction],
        event_type: str,
        solution: str,
        custom_metadata: Dict[str, Any],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: Union[int, float] = 1,
        write_to_event_store: bool = False,
        event_store_url: str = "http://localhost:8001",
        external_id: Optional[str] = None,
        qc_result: Optional[str] = None,
        location: Optional[str] = None,
        item_count: Optional[int] = None,
        item_type: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        alert_description: Optional[str] = None,
        custom_value: Optional[str] = None,
        related_event_id: Optional[str] = None,
        feedback: Optional[str] = None,
        # Appended last on purpose: inserting it earlier would silently
        # rebind any positional caller's arguments.
        value_base_units: Optional[Union[int, float]] = None,
        video_clips: Optional[Any] = None,
        video_label: str = DEFAULT_VIDEO_LABEL,
        confirm_ingestion: bool = False,
        clip_upload_timeout_seconds: float = DEFAULT_CLIP_UPLOAD_TIMEOUT_SECONDS,
    ) -> BlockResult:
        if self._disable_sinks or disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "event_id": "",
                "message": disabled_sink_message(
                    disabled_by_execution_policy=self._disable_sinks
                ),
                "videos": skipped_attachments(
                    video_clips, video_label=video_label, video_status=STATUS_DISABLED
                ),
            }

        # Selector-resolved values bypass manifest validation; a negative
        # cooldown behaves as 0 (rate limiting disabled).
        cooldown_seconds = max(cooldown_seconds, 0)
        seconds_since_last_event = cooldown_seconds
        if self._last_event_fired is not None:
            seconds_since_last_event = (
                datetime.now() - self._last_event_fired
            ).total_seconds()
        if seconds_since_last_event < cooldown_seconds:
            logger.info("Activated `roboflow_core/roboflow_vision_events@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "event_id": "",
                "message": "Sink cooldown applies",
                "videos": skipped_attachments(
                    video_clips, video_label=video_label, video_status=STATUS_THROTTLED
                ),
            }

        event_data = _build_event_data(
            event_type=event_type,
            external_id=external_id,
            qc_result=qc_result,
            location=location,
            item_count=item_count,
            item_type=item_type,
            alert_type=alert_type,
            severity=severity,
            alert_description=alert_description,
            custom_value=custom_value,
            related_event_id=related_event_id,
            feedback=feedback,
        )
        # Operator feedback never carries a value. The cloud and the local event store
        # drop one too, but omitting it here holds for every destination and version.
        value_base_units = (
            None
            if event_type == "operator_feedback"
            else _normalize_value_base_units(value_base_units)
        )

        clip_attachments: Optional[ClipAttachments] = None
        videos: List[Dict[str, Any]] = []
        if write_to_event_store:
            videos = skipped_attachments(
                video_clips,
                video_label=video_label,
                video_status=STATUS_UNSUPPORTED_SINK_MODE,
            )
            task = partial(
                _execute_local_event,
                event_store_url=event_store_url,
                input_image=input_image,
                output_image=output_image,
                prediction=predictions,
                event_type=event_type,
                solution=solution,
                event_data=event_data,
                custom_metadata=custom_metadata,
                value_base_units=value_base_units,
            )
        else:
            if self._api_key is None:
                raise ValueError(
                    "VisionEvents block cannot run without Roboflow API key. "
                    "If you do not know how to get API key - visit "
                    "https://docs.roboflow.com/api-reference/authentication"
                    "#retrieve-an-api-key to learn how to retrieve one."
                )
            # Captures are taken here, on the caller's thread, even in
            # fire-and-forget mode: the anchor refers to this frame.
            clip_attachments = plan_attachments(video_clips, video_label=video_label)
            clip_task_parameters: Dict[str, Any] = {}
            if clip_attachments is not None:
                clip_attachments.bind_uploader(
                    self._clip_uploads.get,
                    clip_upload_timeout_seconds=clip_upload_timeout_seconds,
                    confirm_ingestion=confirm_ingestion,
                )
                videos = clip_attachments.outputs()
                clip_task_parameters["clip_attachments"] = clip_attachments
                # Released at close if the event task never runs.
                self._clip_uploads.track(clip_attachments)
            task = partial(
                _execute_vision_event,
                api_base_url=API_BASE_URL,
                api_key=self._api_key,
                platform_client=self._platform_client,
                input_image=input_image,
                output_image=output_image,
                prediction=predictions,
                event_type=event_type,
                solution=solution,
                event_data=event_data,
                custom_metadata=custom_metadata,
                value_base_units=value_base_units,
                **clip_task_parameters,
            )

        self._last_event_fired = datetime.now()

        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(task)
            return {
                "error_status": False,
                "throttling_status": False,
                "event_id": "",
                "message": "Vision event sent in background task",
                "videos": videos,
            }
        elif fire_and_forget and self._thread_pool_executor:
            task_future = self._thread_pool_executor.submit(task)
            if clip_attachments is not None:
                task_future.add_done_callback(
                    partial(
                        self._clip_uploads.abandon_if_cancelled,
                        clip_attachments=clip_attachments,
                    )
                )
            return {
                "error_status": False,
                "throttling_status": False,
                "event_id": "",
                "message": "Vision event sent in background task",
                "videos": videos,
            }
        else:
            error_status, message, event_id = task()
            if clip_attachments is not None:
                videos = clip_attachments.outputs()
            return {
                "error_status": error_status,
                "throttling_status": False,
                "event_id": event_id,
                "message": message,
                "videos": videos,
            }


def _build_event_data(
    event_type: str,
    external_id: Optional[str] = None,
    qc_result: Optional[str] = None,
    location: Optional[str] = None,
    item_count: Optional[int] = None,
    item_type: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    alert_description: Optional[str] = None,
    custom_value: Optional[str] = None,
    related_event_id: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Build schema-specific eventData dict with camelCase keys, stripping None values."""
    if event_type == "quality_check":
        data = {"result": qc_result, "externalId": external_id}
    elif event_type == "inventory_count":
        data = {
            "location": location,
            "itemCount": item_count,
            "itemType": item_type,
            "externalId": external_id,
        }
    elif event_type == "safety_alert":
        data = {
            "alertType": alert_type,
            "severity": severity,
            "description": alert_description,
            "externalId": external_id,
        }
    elif event_type == "custom":
        data = {"value": custom_value, "externalId": external_id}
    elif event_type == "operator_feedback":
        data = {
            "relatedEventId": related_event_id,
            "feedback": feedback,
        }
    else:
        logger.warning("Unknown event_type: %s", event_type)
        data = {}
    return {k: v for k, v in data.items() if v is not None}


def _normalize_value_base_units(value_base_units: Any) -> Optional[int]:
    """Coerce the event value to a whole number of base units, or None to omit it.

    Literals and workflow inputs are validated against the manifest, so they are
    whole numbers, but a workflow input arrives raw and may still be a string
    such as "1999"; strings are parsed with the manifest's own rule. Step
    outputs bypass that validation, so any real number is accepted (numpy
    scalars too, which the JSON encoder would reject) and a fractional amount is
    rounded to the nearest base unit. Anything else, or an amount past
    ``MAX_ABS_VALUE_BASE_UNITS``, is dropped with a warning instead of failing
    the event: cloud ingestion drops such a value and keeps the event, while the
    local event store would reject the whole event.
    """
    if value_base_units is None:
        return None

    normalized: Optional[int] = None
    if isinstance(value_base_units, str):
        try:
            normalized = _WHOLE_NUMBER_STRING.validate_python(value_base_units)
        except ValidationError:
            normalized = None
    elif not isinstance(value_base_units, bool):
        if isinstance(value_base_units, numbers.Integral):
            normalized = int(value_base_units)
        elif isinstance(value_base_units, numbers.Real) and math.isfinite(
            value_base_units
        ):
            normalized = int(round(value_base_units))
    if normalized is not None and abs(normalized) <= MAX_ABS_VALUE_BASE_UNITS:
        return normalized

    logger.warning(
        "Ignoring value_base_units %r: expected a finite number of base units "
        "between -%d and %d.",
        value_base_units,
        MAX_ABS_VALUE_BASE_UNITS,
        MAX_ABS_VALUE_BASE_UNITS,
    )
    return None


def _convert_predictions_to_annotations(
    prediction: Optional[TensorNativePrediction],
) -> Dict[str, List[dict]]:
    """Convert predictions into vision events annotation lists.

    Returns a dict keyed by annotation type (objectDetections, classifications,
    instanceSegmentations, keypoints), containing only the non-empty annotation
    lists. Shared by the cloud and local event store code paths.
    """
    object_detections: List[dict] = []
    classifications: List[dict] = []
    instance_segmentations: List[dict] = []
    keypoints_detections: List[dict] = []

    if prediction is not None:
        if isinstance(
            prediction,
            (ClassificationPrediction, MultiLabelClassificationPrediction),
        ):
            classifications = _convert_classification_to_vision_events_format(
                prediction
            )
        else:
            # Detection-style prediction (object detection, instance segmentation,
            # or the keypoint `(KeyPoints, Detections)` tuple).
            (
                object_detections,
                instance_segmentations,
                keypoints_detections,
            ) = _convert_native_detections_to_vision_events_format(prediction)

    annotations: Dict[str, List[dict]] = {}
    if object_detections:
        annotations["objectDetections"] = object_detections
    if classifications:
        annotations["classifications"] = classifications
    if instance_segmentations:
        annotations["instanceSegmentations"] = instance_segmentations
    if keypoints_detections:
        annotations["keypoints"] = keypoints_detections
    return annotations


def _execute_vision_event(
    api_base_url: str,
    api_key: str,
    input_image: Optional[WorkflowImageData],
    output_image: Optional[WorkflowImageData],
    prediction: Optional[TensorNativePrediction],
    event_type: str,
    solution: str,
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
    platform_client: Optional[RoboflowPlatformClient] = None,
    value_base_units: Optional[int] = None,
    clip_attachments: Optional[ClipAttachments] = None,
) -> Tuple[bool, str, str]:
    payload: Dict[str, Any] = {}
    try:
        # Step 1: Convert predictions to vision events annotation format
        annotations = _convert_predictions_to_annotations(prediction)

        # Step 2: Upload images and build a single image entry
        # sourceId = output/display image, inputSourceId = original input image
        image_entry: Dict[str, Any] = {}

        if output_image is not None:
            output_source_id, _ = _upload_image(
                api_base_url, api_key, output_image, platform_client=platform_client
            )
            image_entry["sourceId"] = output_source_id

        if input_image is not None:
            input_source_id, _ = _upload_image(
                api_base_url,
                api_key,
                input_image,
                jpeg_quality=95,
                platform_client=platform_client,
            )
            image_entry["inputSourceId"] = input_source_id

        if image_entry:
            image_entry["label"] = "workflow"
            image_entry.update(annotations)

        images_payload: List[dict] = [image_entry] if image_entry else []

        # Step 3: Build and send event
        payload = _build_event_payload(
            event_type=event_type,
            solution=solution,
            images=images_payload,
            event_data=event_data,
            custom_metadata=custom_metadata,
            value_base_units=value_base_units,
        )
        if clip_attachments is not None:
            # One upload slot per clip; the event references them in order.
            clip_attachments.prepare_uploads(
                event_id=payload["eventId"],
                api_base_url=api_base_url,
                platform_client=resolve_platform_blocks_client(platform_client),
                request_headers=_build_cloud_request_headers(
                    api_key=api_key,
                    platform_client=platform_client,
                    explicit_headers={"Content-Type": "application/json"},
                ),
                raise_for_status=_api_key_safe_raise_for_status,
            )
            videos_field = clip_attachments.videos_field()
            if videos_field:
                payload["videos"] = videos_field

        error_status, message = _send_event(
            api_base_url, api_key, payload, platform_client=platform_client
        )
        if clip_attachments is not None:
            message = redact_signed_urls(message)
            if error_status:
                clip_attachments.abandon(event_id=payload["eventId"], reason=message)
            else:
                clip_attachments.hand_off(event_id=payload["eventId"])
        # The eventId is generated client-side and sent in the payload, so it is the
        # canonical id of the created event; surface it on success.
        event_id = "" if error_status else payload.get("eventId", "")
        return error_status, message, event_id
    except Exception as error:
        if clip_attachments is not None:
            clip_attachments.abandon(
                event_id=payload.get("eventId", ""),
                reason=f"{type(error).__name__}: {error}",
            )
        logger.warning("Failed to create vision event: %s", error)
        return (
            True,
            f"Error creating vision event: {type(error).__name__}: {error}",
            "",
        )


def _execute_local_event(
    event_store_url: str,
    input_image: Optional[WorkflowImageData],
    output_image: Optional[WorkflowImageData],
    prediction: Optional[TensorNativePrediction],
    event_type: str,
    solution: str,
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
    value_base_units: Optional[int] = None,
) -> Tuple[bool, str, str]:
    """Send an event to a local Event Ingestion Service (v2 API).

    Unlike the cloud path, images are embedded directly in the request as base64
    rather than uploaded first. The use case (`solution`) is forwarded so events are
    namespaced consistently with the cloud path; the service requires it when cloud
    upload is enabled.
    """
    try:
        # Convert predictions to vision events annotation format
        annotations = _convert_predictions_to_annotations(prediction)

        # Build a single image entry with base64-embedded images
        # base64Image = output/display image, inputBase64Image = original input image
        image_entry: Dict[str, Any] = {}
        if output_image is not None:
            image_entry["base64Image"] = output_image.base64_image
        if input_image is not None:
            image_entry["inputBase64Image"] = input_image.base64_image
        if image_entry:
            image_entry["label"] = "workflow"
            image_entry.update(annotations)

        images_payload: List[dict] = [image_entry] if image_entry else []

        payload: Dict[str, Any] = {
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "solution": solution,
            "event_schema": event_type,
            "event_data": event_data,
            "images": images_payload,
        }
        if custom_metadata:
            payload["custom_metadata"] = custom_metadata
        if value_base_units is not None:
            payload["value_base_units"] = value_base_units
        if images_payload:
            payload["displayImagePosition"] = 0

        url = f"{event_store_url.rstrip('/')}/v2/events"
        return _send_local_event(url, payload)
    except Exception as error:
        logger.warning("Failed to write event to local event store: %s", error)
        return (
            True,
            f"Error writing event to event store: {type(error).__name__}: {error}",
            "",
        )


def _send_local_event(
    url: str,
    payload: dict,
) -> Tuple[bool, str, str]:
    """Send an event to the local Event Ingestion Service.

    Authenticates with the ``EVENT_INGESTION_API_KEY`` environment variable when set.
    Mirrors the Event Writer sink's response handling for the shared ``/v2/events``
    endpoint: the service returns 201 with the server-assigned event ``id`` on success,
    and 529 when the device is at capacity and applying backpressure while it waits for
    cloud uploads to drain.

    Returns:
        Tuple of (error_status, message, event_id)
    """
    try:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("EVENT_INGESTION_API_KEY")
        if api_key:
            headers["X-API-Key"] = api_key
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 201:
            event_id = str(response.json().get("id", ""))
            return False, "Event written to local event store successfully", event_id

        if response.status_code == 529:
            detail = _extract_detail(response)
            return (
                True,
                "Event Ingestion Service at capacity (529). The device is "
                f"experiencing backpressure. Detail: {detail}",
                "",
            )

        detail = _extract_detail(response)
        logger.warning(
            "Event Ingestion Service error (%s): %s", response.status_code, detail
        )
        return (
            True,
            f"Failed to write event to event store. HTTP {response.status_code}: {detail}",
            "",
        )
    except requests.exceptions.Timeout:
        return True, "Request to event store timed out after 30s", ""
    except Exception as e:
        logger.warning("Failed to write event to local event store: %s", e)
        return (
            True,
            f"Failed to write event to event store. Error: {type(e).__name__}: {e}",
            "",
        )


def _extract_detail(response: requests.Response) -> str:
    """Extract the ``detail`` field from a JSON error response, falling back to text."""
    try:
        return str(response.json().get("detail", response.text))
    except Exception:
        return response.text


def _detect_prediction_type(
    prediction: Union[Detections, InstanceDetections, KeyPointPrediction],
) -> str:
    """Determine detection type from the tensor-native prediction shape.

    Keypoint predictions arrive as a `(KeyPoints, Detections)` tuple, instance
    segmentation as `inference_models.InstanceDetections`, and plain object
    detection as `inference_models.Detections`.
    """
    if isinstance(prediction, tuple):
        return "keypoint_detection"
    if isinstance(prediction, InstanceDetections):
        return "instance_segmentation"
    return "object_detection"


def _convert_native_detections_to_vision_events_format(
    prediction: Union[Detections, InstanceDetections, KeyPointPrediction],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Convert tensor-native detections to vision events format.

    Returns:
        Tuple of (object_detections, instance_segmentations, keypoints)
        using center-based bounding boxes in absolute pixel coordinates.
    """
    detection_type = _detect_prediction_type(prediction)

    # Keypoint predictions arrive as a `(KeyPoints, Detections)` tuple; the vision
    # events payload is driven by the bbox component, whose `bboxes_metadata`
    # carries the per-detection keypoint arrays.
    _key_points, detections = split_key_point_prediction(prediction)

    object_detections: List[dict] = []
    instance_segmentations: List[dict] = []
    keypoints_list: List[dict] = []

    # class_name of detection i is resolved through the per-image class_id -> name
    # map (numpy stored it per-detection in `sv.Detections.data["class_name"]`).
    class_names_map = (detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}

    for index, (
        xyxy,
        _mask,
        class_id,
        confidence,
        _tracker_id,
        data,
        _meta,
    ) in enumerate(detections):
        xyxy = xyxy.detach().to("cpu").numpy().astype(float).tolist()
        x1, y1, x2, y2 = xyxy
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        cx = x1 + w / 2
        cy = y1 + h / 2
        class_id_int = int(class_id)
        class_name = str(class_names_map.get(class_id_int, "unknown"))
        conf = float(confidence) if confidence is not None else 0.0
        # Clamp confidence to [0, 1]
        conf = max(0.0, min(1.0, conf))

        base = {
            "class": class_name,
            "x": cx,
            "y": cy,
            "width": w,
            "height": h,
            "confidence": conf,
        }

        if detection_type == "keypoint_detection":
            kp_entry = dict(base)
            kp_xy = data.get(KEYPOINTS_XY_KEY_IN_SV_DETECTIONS)
            kp_class_id = data.get(KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS)
            kp_class_name = data.get(KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS)
            if kp_xy is not None and len(kp_xy) > 0:
                # Skip trailing padding slots (see common/keypoints.py).
                kp_count = real_keypoints_count(kp_class_name, total=len(kp_xy))
                kp_entry["keypoints"] = []
                for i, (kx, ky) in enumerate(kp_xy[:kp_count]):
                    kp_id = (
                        int(kp_class_id[i])
                        if kp_class_id is not None and i < len(kp_class_id)
                        else i
                    )
                    kp_entry["keypoints"].append(
                        {"id": kp_id, "x": float(kx), "y": float(ky)}
                    )
            keypoints_list.append(kp_entry)

        elif detection_type == "instance_segmentation":
            seg_entry = dict(base)
            polygon = data.get(POLYGON_KEY_IN_SV_DETECTIONS)
            if polygon is None and detections.mask is not None:
                polygon = mask_to_polygon(
                    mask=instance_mask_to_numpy(detections, index)
                )
            if polygon is not None and len(polygon) >= 3:
                if isinstance(polygon, np.ndarray):
                    polygon = polygon.astype(float).tolist()
                seg_entry["points"] = [[float(pt[0]), float(pt[1])] for pt in polygon]
                instance_segmentations.append(seg_entry)
            else:
                # Fall back to object detection if polygon is invalid
                object_detections.append(base)

        else:
            object_detections.append(base)

    return object_detections, instance_segmentations, keypoints_list


def _convert_classification_to_vision_events_format(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> List[dict]:
    """Convert tensor-native classification predictions to vision events classification format.

    Handles both single-label (`ClassificationPrediction`, top-1 class) and
    multi-label (`MultiLabelClassificationPrediction`, all above-threshold classes)
    predictions. Class names are resolved through the per-image `class_names` map
    carried in the prediction's image metadata; the `confidence` tensor is the full
    distribution, so each emitted class is paired with its own per-class confidence.
    """
    classifications: List[dict] = []

    if isinstance(prediction, ClassificationPrediction):
        images_metadata = prediction.images_metadata or [{}]
        image_metadata = images_metadata[0] if images_metadata else {}
        class_names_map = image_metadata.get(CLASS_NAMES_KEY) or {}
        class_id = int(prediction.class_id[0])
        class_name = class_names_map.get(class_id, f"class_{class_id}")
        conf = float(prediction.confidence[0, class_id])
        classifications.append({"class": str(class_name), "confidence": conf})
        return classifications

    if isinstance(prediction, MultiLabelClassificationPrediction):
        image_metadata = prediction.image_metadata or {}
        class_names_map = image_metadata.get(CLASS_NAMES_KEY) or {}
        for class_id_scalar in prediction.class_ids.tolist():
            class_id = int(class_id_scalar)
            class_name = class_names_map.get(class_id, f"class_{class_id}")
            conf = float(prediction.confidence[class_id])
            classifications.append({"class": str(class_name), "confidence": conf})
        return classifications

    return classifications


def _upload_image(
    api_base_url: str,
    api_key: str,
    image_data: WorkflowImageData,
    jpeg_quality: int = 85,
    platform_client: Optional[RoboflowPlatformClient] = None,
) -> Tuple[str, str]:
    """Upload an image to the Vision Events API.

    Returns:
        Tuple of (sourceId, url)
    """
    platform_client = resolve_platform_blocks_client(platform_client)
    ensure_roboflow_api_access(platform_client)
    image_bytes = encode_image_to_jpeg_bytes(
        image_data.numpy_image, jpeg_quality=jpeg_quality
    )
    response = requests.post(
        platform_client.wrap_url(f"{api_base_url}/vision-events/upload"),
        headers=_build_cloud_request_headers(
            api_key=api_key, platform_client=platform_client
        ),
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    _api_key_safe_raise_for_status(response=response)
    result = response.json()
    return result["sourceId"], result.get("url", "")


def _build_event_payload(
    event_type: str,
    solution: str,
    images: List[dict],
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
    value_base_units: Optional[int] = None,
) -> dict:
    """Build the full event payload for the Vision Events API."""
    event_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    payload: Dict[str, Any] = {
        "eventId": event_id,
        "eventType": event_type,
        "useCaseId": solution,
        "timestamp": timestamp,
        "images": images,
    }

    if event_data:
        payload["eventData"] = event_data
    if custom_metadata:
        payload["customMetadata"] = custom_metadata
    if value_base_units is not None:
        payload["valueBaseUnits"] = value_base_units
    # Output image is at position 0 (first in images array), always display it
    if len(images) > 0:
        payload["displayImagePosition"] = 0

    return payload


def _send_event(
    api_base_url: str,
    api_key: str,
    payload: dict,
    platform_client: Optional[RoboflowPlatformClient] = None,
) -> Tuple[bool, str]:
    """Send a vision event to the API.

    Returns:
        Tuple of (error_status, message)
    """
    platform_client = resolve_platform_blocks_client(platform_client)
    try:
        ensure_roboflow_api_access(platform_client)
        response = requests.post(
            platform_client.wrap_url(f"{api_base_url}/vision-events"),
            headers=_build_cloud_request_headers(
                api_key=api_key,
                platform_client=platform_client,
                explicit_headers={"Content-Type": "application/json"},
            ),
            json=payload,
            timeout=30,
        )
        _api_key_safe_raise_for_status(response=response)
        return False, "Vision event sent successfully"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text if e.response is not None else "no response"
        logger.warning("Vision Events API error (%s): %s", status_code, body)
        return (
            True,
            f"Failed to send vision event. Status: {status_code}. Details: {body}",
        )
    except Exception as e:
        logger.warning("Failed to send vision event: %s", e)
        return True, f"Failed to send vision event. Error: {type(e).__name__}: {e}"
