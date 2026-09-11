import errno
import io
import json
import os
import re
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
)

from inference.core.env import ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE
from inference.core.logger import logger
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
    is_workflow_selector,
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

DEFAULT_FILE_NAME_PREFIX = "event_"
BUNDLE_FILE_NAME_SUFFIX = ".tar.gz"

# A custom file name replaces the generated one wholesale, so it must not be
# able to redirect the write (path separators, `..`) nor collide with the
# dot-prefixed temporary files that file movers are told to skip.  Matched with
# `fullmatch` - `$` alone would also accept a trailing newline.
BUNDLE_FILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# `AtomicPath` writes through a `.tmp_<random>_<file name>` sibling, so the
# name must stay clear of the 255-byte path-segment limit with that padding.
MAX_FILE_NAME_LENGTH = 200

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
event_<UTC timestamp>_<eventId>.tar.gz    # or your own `file_name`
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
atomically renamed to their final `*.tar.gz` name (the directory is fsynced after the
rename). A file-mover service that matches `*.tar.gz` (or skips dotfiles) can never pick
up a partially written bundle.

## Output Routing

By default the file name is `event_<UTC timestamp>_<eventId>.tar.gz`. Set **File Name** to
replace it outright - typically wired from another step that composes the name - so no part
of Roboflow's naming is imposed on the output. `.tar.gz` is appended when absent.

Because `target_directory` is per-block too, several bundle blocks in one workflow can own
their output completely: a person-detection block writing `person_*.tar.gz` into
`/media/person` alongside an episode block writing the default `event_*.tar.gz` into
`/media/event`, letting a downstream file mover route purely by folder and name. The upload
endpoint ignores the file name entirely, so it is free to carry deployment-specific meaning.

The generated name embeds a uuid4 and is unique by construction. A custom name is not: if
the target file already exists the block raises rather than overwriting it, so a name that
repeats across events will fail the workflow run. Compose something unique per event.

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
    file_name: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        title="File Name",
        description="Optional file name for the bundle, replacing the generated "
        "`event_<UTC timestamp>_<eventId>.tar.gz` name entirely. Usually wired "
        "from another step that composes the name, so nothing of Roboflow's "
        "naming is imposed. `.tar.gz` is appended when absent. May contain "
        "letters, digits, `.`, `_` and `-`, must start with a letter or digit, "
        "and must be at most 200 characters. The block errors if the "
        "resulting file already exists, so a "
        "custom name must be unique per event. Leave unset for the default "
        "name, which is unique by construction.",
        examples=["$steps.compose_name.output", "person_batch7.tar.gz"],
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

    @field_validator("file_name")
    @classmethod
    def ensure_file_name_is_safe(cls, value: Any) -> Any:
        if isinstance(value, str) and not is_workflow_selector(value):
            validate_bundle_file_name(value)
        return value

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
        # Appended last on purpose: inserting it earlier would silently
        # rebind any positional caller's event-data arguments.
        file_name: Optional[str] = None,
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

        # Selector-resolved values bypass manifest validation, so an unsafe
        # name would otherwise reach the path unchecked.
        if file_name is not None:
            validate_bundle_file_name(file_name)
        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)
        target_path = _generate_bundle_path(
            target_directory=target_directory,
            timestamp=timestamp,
            event_id=event_id,
            file_name=file_name,
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
        # A generated name carries a uuid4 and cannot collide; a custom one
        # repeats as often as its author repeats it.  Checked here so the
        # ordinary repeated-name case fails loudly and synchronously, before a
        # fire-and-forget dispatch can swallow it.  The publish below is what
        # actually guarantees no bundle is ever overwritten.
        if os.path.exists(target_path):
            raise ValueError(
                f"`roboflow_core/vision_event_bundle@v1` block cannot write "
                f"`{target_path}` - the file already exists. A custom "
                f"`file_name` must be unique for every event."
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


def validate_bundle_file_name(file_name: str) -> None:
    if not file_name:
        raise ValueError(
            "`file_name` must not be empty - leave it unset for the default "
            "generated name."
        )
    if len(file_name) > MAX_FILE_NAME_LENGTH:
        raise ValueError(
            f"`file_name` must be at most {MAX_FILE_NAME_LENGTH} characters "
            f"long, got {len(file_name)}."
        )
    if not BUNDLE_FILE_NAME_PATTERN.fullmatch(file_name):
        raise ValueError(
            f"`file_name` must start with a letter or digit and may only "
            f"contain letters, digits, `.`, `_` and `-`, got `{file_name}`."
        )


# Only these mean "this filesystem cannot do hard links".  Every other OSError
# is a real write failure and must not silently reach the replacing fallback.
_LINK_UNSUPPORTED_ERRNOS = frozenset(
    {errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV, errno.EMLINK}
)


def _publish_bundle(target_path: str, content: bytes) -> None:
    """Write `content` and publish it at `target_path` without replacing.

    `os.link` fails atomically when the destination already exists, so two
    concurrent events that pass the foreground collision check cannot both
    believe they won - and unlike a reservation marker, an interrupted process
    leaves only the dot-prefixed temporary file that file movers already skip.

    On filesystems with no hard-link support (FAT-family removable media, the
    usual way bundles leave an air-gapped network) this degrades to a checked
    replacing rename: repeated names are still caught, but two genuinely
    concurrent events with the same name can race.
    """
    directory = os.path.dirname(os.path.abspath(target_path))
    temp_file = tempfile.NamedTemporaryFile(
        dir=directory,
        prefix=".tmp_",
        suffix=f"_{os.path.basename(target_path)}",
        delete=False,
    )
    temp_path = temp_file.name
    try:
        temp_file.write(content)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        try:
            os.link(temp_path, target_path)
        except FileExistsError:
            raise ValueError(
                f"`roboflow_core/vision_event_bundle@v1` block cannot write "
                f"`{target_path}` - the file already exists. A custom "
                f"`file_name` must be unique for every event."
            )
        except OSError as error:
            if error.errno not in _LINK_UNSUPPORTED_ERRNOS:
                raise
            # FAT-family removable media - the way bundles physically leave an
            # air-gapped network - cannot do hard links.  Falling back keeps
            # those targets working, at the cost of the atomic guarantee: the
            # re-check below narrows the window but cannot close it.
            if os.path.exists(target_path):
                raise ValueError(
                    f"`roboflow_core/vision_event_bundle@v1` block cannot write "
                    f"`{target_path}` - the file already exists. A custom "
                    f"`file_name` must be unique for every event."
                )
            logger.warning(
                "Hard links unsupported under %s - publishing vision event "
                "bundle with a replacing rename, which cannot guarantee that "
                "a concurrent event with the same file name is not "
                "overwritten.",
                directory,
            )
            os.replace(temp_path, target_path)
    finally:
        # Cleanup must never turn a published bundle into a reported failure,
        # nor mask the original write error.
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(
                "Failed to remove vision event bundle temporary file %s: %s",
                temp_path,
                error,
            )


def _generate_bundle_path(
    target_directory: str,
    timestamp: datetime,
    event_id: str,
    file_name: Optional[str] = None,
) -> str:
    if file_name:
        if not file_name.endswith(BUNDLE_FILE_NAME_SUFFIX):
            file_name = f"{file_name}{BUNDLE_FILE_NAME_SUFFIX}"
    else:
        file_name = (
            f"{DEFAULT_FILE_NAME_PREFIX}{timestamp.strftime('%Y%m%dT%H%M%S_%f')}"
            f"_{event_id}{BUNDLE_FILE_NAME_SUFFIX}"
        )
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
        _publish_bundle(target_path=target_path, content=tar_bytes)
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
