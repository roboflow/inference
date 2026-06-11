import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import requests
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

from inference.core.env import API_BASE_URL
from inference.core.logger import logger
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.utils.url_utils import wrap_url
from inference.core.workflows.core_steps.common.serializers import mask_to_polygon
from inference.core.workflows.core_steps.common.tensor_native import (
    KeyPointPrediction,
    instance_mask_to_numpy,
    split_key_point_prediction,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    ROBOFLOW_SOLUTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

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
4. Supports fire-and-forget mode for non-blocking execution

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

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="event_id", kind=[STRING_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowVisionEventsBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._api_key = api_key
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

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
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "event_id": "",
                "message": "Sink was disabled by parameter `disable_sink`",
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

        if write_to_event_store:
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
            )
        else:
            if self._api_key is None:
                raise ValueError(
                    "VisionEvents block cannot run without Roboflow API key. "
                    "If you do not know how to get API key - visit "
                    "https://docs.roboflow.com/api-reference/authentication"
                    "#retrieve-an-api-key to learn how to retrieve one."
                )
            task = partial(
                _execute_vision_event,
                api_base_url=API_BASE_URL,
                api_key=self._api_key,
                input_image=input_image,
                output_image=output_image,
                prediction=predictions,
                event_type=event_type,
                solution=solution,
                event_data=event_data,
                custom_metadata=custom_metadata,
            )

        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(task)
            return {
                "error_status": False,
                "event_id": "",
                "message": "Vision event sent in background task",
            }
        elif fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(task)
            return {
                "error_status": False,
                "event_id": "",
                "message": "Vision event sent in background task",
            }
        else:
            error_status, message, event_id = task()
            return {
                "error_status": error_status,
                "event_id": event_id,
                "message": message,
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
) -> Tuple[bool, str, str]:
    try:
        # Step 1: Convert predictions to vision events annotation format
        annotations = _convert_predictions_to_annotations(prediction)

        # Step 2: Upload images and build a single image entry
        # sourceId = output/display image, inputSourceId = original input image
        image_entry: Dict[str, Any] = {}

        if output_image is not None:
            output_source_id, _ = _upload_image(api_base_url, api_key, output_image)
            image_entry["sourceId"] = output_source_id

        if input_image is not None:
            input_source_id, _ = _upload_image(
                api_base_url, api_key, input_image, jpeg_quality=95
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
        )

        error_status, message = _send_event(api_base_url, api_key, payload)
        # The eventId is generated client-side and sent in the payload, so it is the
        # canonical id of the created event; surface it on success.
        event_id = "" if error_status else payload.get("eventId", "")
        return error_status, message, event_id
    except Exception as error:
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

    for index, (xyxy, _mask, class_id, confidence, _tracker_id, data, _meta) in enumerate(
        detections
    ):
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
            if kp_xy is not None and len(kp_xy) > 0:
                kp_entry["keypoints"] = []
                for i, (kx, ky) in enumerate(kp_xy):
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
) -> Tuple[str, str]:
    """Upload an image to the Vision Events API.

    Returns:
        Tuple of (sourceId, url)
    """
    image_bytes = encode_image_to_jpeg_bytes(
        image_data.numpy_image, jpeg_quality=jpeg_quality
    )
    response = requests.post(
        wrap_url(f"{api_base_url}/vision-events/upload"),
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    return result["sourceId"], result.get("url", "")


def _build_event_payload(
    event_type: str,
    solution: str,
    images: List[dict],
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
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
    # Output image is at position 0 (first in images array), always display it
    if len(images) > 0:
        payload["displayImagePosition"] = 0

    return payload


def _send_event(
    api_base_url: str,
    api_key: str,
    payload: dict,
) -> Tuple[bool, str]:
    """Send a vision event to the API.

    Returns:
        Tuple of (error_status, message)
    """
    try:
        response = requests.post(
            wrap_url(f"{api_base_url}/vision-events"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
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
