import io
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from inference.core.env import ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE
from inference.core.logger import logger
from inference.core.utils.file_system import dump_bytes_atomic
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.sinks.local_file.v1 import (
    path_is_within_specified_directory,
)
from inference.core.workflows.core_steps.sinks.noop import disabled_sink_message
from inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1 import (
    ALL_DATA_SCHEMAS_RELEVANT,
    CUSTOM_RELEVANT,
    INVENTORY_COUNT_RELEVANT,
    OPERATOR_FEEDBACK_RELEVANT,
    QUALITY_CHECK_RELEVANT,
    SAFETY_ALERT_RELEVANT,
    _build_event_data,
    _convert_predictions_to_annotations,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_SOLUTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    COOLDOWN_HTTP_SOFT_RESTRICTION,
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

BUNDLE_FORMAT_VERSION = 1

# Companion API enforces a 25 MiB raw bundle limit at ingest time.  Bundles
# larger than this will always be rejected, so we catch the condition locally
# and return an error instead of writing a bundle the consumer will reject.
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MiB

# Consumer schema caps each annotation list at 1 000 items.  Payloads with
# more entries trigger a consumer-side fallback that silently drops *all*
# images, so we enforce the cap before attaching annotations to the payload.
MAX_ANNOTATIONS_PER_LIST = 1000

SHORT_DESCRIPTION = (
    "Write vision events as self-contained tarball bundles to a local directory."
)

LONG_DESCRIPTION = """
Serialize vision events as self-contained tarball bundles on the local filesystem instead
of sending them to the Roboflow Vision Events API. Designed for air-gapped and OT-network
deployments where the inference server has no route to the cloud: a customer file-mover
service transports the bundles out of the network, and an uploader later POSTs each bundle
to Roboflow's `POST /vision-events/bundle` endpoint without unpacking it.

## Bundle Format (version 1)

One tarball per event, written to `target_directory`:

```
event_<UTC timestamp>_<eventId>.tar.gz
├── payload.json               # the event payload (versioned contract, camelCase)
└── images/<file_id>.jpg       # image members, file_id is a uuid4
```

`payload.json` has the same shape as the `POST /vision-events` request body, with these
differences:

- `bundleFormatVersion` identifies the bundle contract version (currently `1`)
- `images[].file` / `images[].inputFile` reference tar member paths instead of
  `sourceId` / `inputSourceId` (the cloud resolves them to source ids at ingest time)
- `useCaseId` is only present when the optional **Use Case** field is set on this block;
  air-gapped deployments typically leave it unset so no cloud identifiers are stored in
  the OT network, and the uploader supplies it via the `useCaseId` query parameter instead

Media members live under type-named directories (`images/` today); future media types will
use sibling directories, so consumers should ignore unknown top-level directories.

## Atomic Writes

Bundles are written to a dot-prefixed temporary file in the target directory, fsynced, and
atomically renamed to their final `event_*.tar.gz` name (the directory is fsynced after the
rename). A file-mover service that matches `event_*.tar.gz` (or skips dotfiles) can never
pick up a partially written bundle.

## Rate Limiting

Video workflows can run many times per second, which by default would write a bundle for
every frame. The block enforces a cooldown between consecutive events: at most one event
per second is written by default. Events triggered during the cooldown period are dropped
and the `throttling_status` output is set to `True`. Adjust `cooldown_seconds` to your
needs, or set it to `0` to disable rate limiting entirely.

## Requirements

**Local Filesystem Access**: This block requires write access to the local filesystem and
is intended for self-hosted `inference`. Filesystem access can be controlled via
environment variables:

- Set `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False` to disable the block
  (it will raise an error)
- Set `WORKFLOW_BLOCKS_WRITE_DIRECTORY` to an absolute path to restrict writes to a
  specific directory and its subdirectories only

No Roboflow API key is required: the block never talks to the network.

## Event Types

- **quality_check**: Manufacturing/inspection QA with pass/fail result and optional confidence
- **inventory_count**: Inventory tracking with location, item count, and item type
- **safety_alert**: Safety violations with alert type, severity (low/medium/high), and description
- **custom**: User-defined events with a free-form value string
- **operator_feedback**: Operator review/correction of previous events (correct/incorrect/inconclusive)
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Write Vision Event Bundle",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-box-archive",
                "blockPriority": 2,
                "popular": False,
                # The block writes to local disk only and never needs cloud
                # credentials — that is the point of it (air-gapped deployments).
                "requires_rf_key": False,
                "local_only": True,
            },
        }
    )
    type: Literal["roboflow_core/vision_event_bundle@v1"]
    target_directory: Union[Selector(kind=[STRING_KIND]), str] = Field(
        title="Target Directory",
        description="Directory path where event bundles will be written. Created "
        "automatically if it does not exist. If WORKFLOW_BLOCKS_WRITE_DIRECTORY is "
        "set, this path must be a subdirectory of the allowed directory.",
        examples=["/data/vision-event-bundles"],
        json_schema_extra={"always_visible": True},
    )
    input_image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        title="Input Image",
        description="The original input image. Stored in the bundle and used as the "
        "base image for detection annotations.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        json_schema_extra={"always_visible": True},
    )
    output_image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        title="Output Image",
        description="An optional output/visualized image (e.g., from a visualization "
        "block). Displayed as the primary image once the event is ingested.",
        examples=["$steps.visualization.image"],
        json_schema_extra={"always_visible": True},
    )
    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
                CLASSIFICATION_PREDICTION_KIND,
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
    solution: Optional[
        Union[str, Selector(kind=[ROBOFLOW_SOLUTION_KIND, STRING_KIND])]
    ] = Field(
        default=None,
        title="Use Case",
        description="Optional use case to bake into the bundle as `useCaseId`. Leave "
        "unset in air-gapped deployments so no cloud identifiers are stored in the "
        "bundle — the uploader supplies the use case at upload time instead. A "
        "`useCaseId` query parameter passed to the ingest endpoint always overrides "
        "the bundled value.",
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
        description="If True, the bundle is written asynchronously and the workflow "
        "continues without waiting. If False, the block waits for the write to finish.",
        examples=[True, "$inputs.fire_and_forget"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        title="Disable Sink",
        description="If True, the block is disabled and no bundles are written.",
        examples=[False, "$inputs.disable_vision_event_bundles"],
    )
    cooldown_seconds: Union[
        NonNegativeInt, NonNegativeFloat, Selector(kind=[INTEGER_KIND, FLOAT_KIND])
    ] = Field(
        default=1,
        title="Cooldown",
        description="Minimum number of seconds between consecutive event bundles "
        "written by this block. Events triggered during the cooldown period are "
        "dropped and the `throttling_status` output is set to True. Defaults to 1 "
        "second so high-frequency video workflows do not write a bundle per frame. "
        "Set to 0 to disable rate limiting for intentionally bursty use cases.",
        examples=[1, 0.5, "$inputs.cooldown_seconds"],
        json_schema_extra={"always_visible": True},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="event_id", kind=[STRING_KIND]),
            OutputDefinition(name="bundle_path", kind=[STRING_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        restrictions = [
            COOLDOWN_HTTP_SOFT_RESTRICTION,
            RuntimeRestriction(
                severity=Severity.SOFT,
                note=(
                    "Bundles are persisted on the deployment's volume but are "
                    "not retrievable through the Roboflow API; this block is "
                    "intended for self-hosted deployments with a file-mover "
                    "process."
                ),
                applies_to_runtimes=[Runtime.DEDICATED_DEPLOYMENT],
            ),
        ]
        if not ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.HARD,
                    note=(
                        "Block raises RuntimeError when ALLOW_WORKFLOW_BLOCKS_"
                        "ACCESSING_LOCAL_STORAGE is False."
                    ),
                    applies_to_runtimes=[
                        Runtime.HOSTED_SERVERLESS,
                        Runtime.DEDICATED_DEPLOYMENT,
                    ],
                )
            )
        else:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.SOFT,
                    note=(
                        "Container disk is ephemeral, so bundles are lost when "
                        "the worker scales down; if there's more than one replica "
                        "consuming workflow requests the result will be non "
                        "deterministic."
                    ),
                    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                )
            )
        return restrictions


class VisionEventBundleSinkBlockV1(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        allow_access_to_file_system: bool,
        allowed_write_directory: Optional[str],
        disable_sinks: bool = False,
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._allow_access_to_file_system = allow_access_to_file_system
        self._allowed_write_directory = allowed_write_directory
        self._disable_sinks = disable_sinks
        self._last_event_fired: Optional[datetime] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return [
            "background_tasks",
            "thread_pool_executor",
            "allow_access_to_file_system",
            "allowed_write_directory",
            "disable_sinks",
        ]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        target_directory: str,
        input_image: Optional[WorkflowImageData],
        output_image: Optional[WorkflowImageData],
        predictions: Optional[Union[sv.Detections, dict]],
        event_type: str,
        custom_metadata: Dict[str, Any],
        fire_and_forget: bool,
        disable_sink: bool,
        solution: Optional[str] = None,
        cooldown_seconds: Union[int, float] = 1,
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
    ) -> BlockResult:
        if self._disable_sinks or disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "event_id": "",
                "bundle_path": "",
                "message": disabled_sink_message(
                    disabled_by_execution_policy=self._disable_sinks
                ),
            }
        if not self._allow_access_to_file_system:
            raise RuntimeError(
                "`roboflow_core/vision_event_bundle@v1` block cannot run in this "
                "environment - local file system usage is forbidden - use "
                "self-hosted `inference` or Roboflow Dedicated Deployment."
            )

        # Selector-resolved values bypass manifest validation; a negative
        # cooldown behaves as 0 (rate limiting disabled).
        cooldown_seconds = max(cooldown_seconds, 0)
        seconds_since_last_event = cooldown_seconds
        if self._last_event_fired is not None:
            seconds_since_last_event = (
                datetime.now() - self._last_event_fired
            ).total_seconds()
        if seconds_since_last_event < cooldown_seconds:
            logger.info("Activated `roboflow_core/vision_event_bundle@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "event_id": "",
                "bundle_path": "",
                "message": "Sink cooldown applies",
            }

        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)
        target_path = _generate_bundle_path(
            target_directory=target_directory,
            timestamp=timestamp,
            event_id=event_id,
        )
        # Misconfiguration should fail loudly here, not vanish into a
        # fire-and-forget background task.
        self._verify_write_access_to_directory(target_directory=target_directory)
        self._verify_write_access_to_directory(target_directory=target_path)
        os.makedirs(target_directory, exist_ok=True)
        if not os.access(target_directory, os.W_OK):
            raise ValueError(
                f"`roboflow_core/vision_event_bundle@v1` block cannot write to "
                f"`{target_directory}` - the directory is not writable."
            )

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
        task = partial(
            _write_event_bundle,
            target_directory=target_directory,
            target_path=target_path,
            event_id=event_id,
            timestamp=timestamp,
            input_image=input_image,
            output_image=output_image,
            prediction=predictions,
            event_type=event_type,
            solution=solution,
            event_data=event_data,
            custom_metadata=custom_metadata,
        )

        self._last_event_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(task)
            return {
                "error_status": False,
                "throttling_status": False,
                "event_id": "",
                "bundle_path": "",
                "message": "Vision event bundle written in background task",
            }
        elif fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(task)
            return {
                "error_status": False,
                "throttling_status": False,
                "event_id": "",
                "bundle_path": "",
                "message": "Vision event bundle written in background task",
            }
        else:
            error_status, message, event_id, bundle_path = task()
            return {
                "error_status": error_status,
                "throttling_status": False,
                "event_id": event_id,
                "bundle_path": bundle_path,
                "message": message,
            }

    def _verify_write_access_to_directory(self, target_directory: str) -> None:
        if self._allowed_write_directory is None:
            return None
        if not path_is_within_specified_directory(
            path=target_directory,
            specified_directory=self._allowed_write_directory,
        ):
            raise ValueError(
                f"Requested `roboflow_core/vision_event_bundle@v1` block to write "
                f"bundles into `{target_directory}` which is not a sub-directory of "
                f"the allowed write location. Expected sub-directory of "
                f"{self._allowed_write_directory}"
            )


def _generate_bundle_path(
    target_directory: str,
    timestamp: datetime,
    event_id: str,
) -> str:
    file_name = f"event_{timestamp.strftime('%Y%m%dT%H%M%S_%f')}_{event_id}.tar.gz"
    return os.path.abspath(os.path.join(target_directory, file_name))


def _write_event_bundle(
    target_directory: str,
    target_path: str,
    event_id: str,
    timestamp: datetime,
    input_image: Optional[WorkflowImageData],
    output_image: Optional[WorkflowImageData],
    prediction: Optional[Union[sv.Detections, dict]],
    event_type: str,
    solution: Optional[str],
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
) -> Tuple[bool, str, str, str]:
    try:
        annotations = _cap_annotation_lists(
            _convert_predictions_to_annotations(prediction)
        )

        image_members: Dict[str, bytes] = {}
        image_entry: Dict[str, Any] = {}
        if output_image is not None:
            member_name = f"images/{uuid4()}.jpg"
            image_members[member_name] = encode_image_to_jpeg_bytes(
                output_image.numpy_image, jpeg_quality=85
            )
            image_entry["file"] = member_name
        if input_image is not None:
            member_name = f"images/{uuid4()}.jpg"
            image_members[member_name] = encode_image_to_jpeg_bytes(
                input_image.numpy_image, jpeg_quality=95
            )
            image_entry["inputFile"] = member_name
        if image_entry:
            image_entry["label"] = "workflow"
            image_entry.update(annotations)

        payload = _build_bundle_payload(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            solution=solution,
            images=[image_entry] if image_entry else [],
            event_data=event_data,
            custom_metadata=custom_metadata,
        )

        tar_bytes = _build_tar_bytes(
            payload=payload,
            image_members=image_members,
            mtime=int(timestamp.timestamp()),
        )
        if len(tar_bytes) > MAX_BUNDLE_SIZE_BYTES:
            raise ValueError(
                f"Bundle size {len(tar_bytes):,} bytes exceeds the companion API "
                f"limit of {MAX_BUNDLE_SIZE_BYTES:,} bytes (25 MiB). Reduce the "
                "number or size of images in this event."
            )
        dump_bytes_atomic(target_path, tar_bytes)
        _fsync_directory(target_directory)
        return False, "Vision event bundle written successfully", event_id, target_path
    except Exception as error:
        logger.warning("Failed to write vision event bundle: %s", error)
        return (
            True,
            f"Error writing vision event bundle: {type(error).__name__}: {error}",
            "",
            "",
        )


def _build_bundle_payload(
    event_id: str,
    timestamp: datetime,
    event_type: str,
    solution: Optional[str],
    images: List[dict],
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
) -> dict:
    payload: Dict[str, Any] = {
        "bundleFormatVersion": BUNDLE_FORMAT_VERSION,
        "eventId": event_id,
        "eventType": event_type,
        "timestamp": timestamp.isoformat(),
        "images": images,
    }
    if solution:
        payload["useCaseId"] = solution
    if event_data:
        payload["eventData"] = event_data
    if custom_metadata:
        payload["customMetadata"] = custom_metadata
    # Output image is at position 0 (first in images array), always display it
    if len(images) > 0:
        payload["displayImagePosition"] = 0
    return payload


def _cap_annotation_lists(annotations: Dict[str, Any]) -> Dict[str, Any]:
    annotation_keys = (
        "objectDetections",
        "classifications",
        "instanceSegmentations",
        "keypoints",
    )
    capped = {}
    for key, value in annotations.items():
        if key in annotation_keys and isinstance(value, list):
            if len(value) > MAX_ANNOTATIONS_PER_LIST:
                logger.warning(
                    "vision_event_bundle: annotation list '%s' has %d items; "
                    "truncating to %d to match consumer schema limit.",
                    key,
                    len(value),
                    MAX_ANNOTATIONS_PER_LIST,
                )
            capped[key] = value[:MAX_ANNOTATIONS_PER_LIST]
        else:
            capped[key] = value
    return capped


def _build_tar_bytes(
    payload: dict,
    image_members: Dict[str, bytes],
    mtime: int,
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload_bytes = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
        payload_info = tarfile.TarInfo(name="payload.json")
        payload_info.size = len(payload_bytes)
        payload_info.mtime = mtime
        tar.addfile(payload_info, io.BytesIO(payload_bytes))
        for member_name, member_bytes in image_members.items():
            member_info = tarfile.TarInfo(name=member_name)
            member_info.size = len(member_bytes)
            member_info.mtime = mtime
            tar.addfile(member_info, io.BytesIO(member_bytes))
    return buffer.getvalue()


def _fsync_directory(path: str) -> None:
    # Directory fsync makes the atomic rename durable across power loss.
    # Windows has no directory fsync; skip there.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
