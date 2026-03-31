import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import requests
import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.env import API_BASE_URL
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.serializers import mask_to_polygon
from inference.core.workflows.execution_engine.constants import (
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
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
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

logger = logging.getLogger(__name__)

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

## Event Types

- **quality_check**: Manufacturing/inspection QA with pass/fail result and optional confidence
- **inventory_count**: Inventory tracking with location, item count, and item type
- **safety_alert**: Safety violations with alert type, severity (low/medium/high), and description
- **custom**: User-defined events with a free-form value string
- **operator_feedback**: Operator review/correction of previous events (correct/incorrect/inconclusive)

## Requirements

**API Key Required**: This block requires a valid Roboflow API key with `vision-events:write`
scope. The API key must be configured in your environment or workflow configuration.

**Enterprise Plan**: Vision Events requires an Enterprise plan.

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
                "requires_rf_key": True,
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
    )
    output_image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        title="Output Image",
        description="An optional output/visualized image (e.g., from a visualization "
        "block). Displayed as the primary image in the Vision Events dashboard.",
        examples=["$steps.visualization.image"],
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
        json_schema_extra={"always_visible": True},
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

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["input_image", "output_image", "predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
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
        input_image: Optional[Batch[WorkflowImageData]],
        output_image: Optional[Batch[WorkflowImageData]],
        predictions: Optional[Batch[Union[sv.Detections, dict]]],
        event_type: str,
        solution: str,
        custom_metadata: Dict[str, Any],
        fire_and_forget: bool,
        disable_sink: bool,
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
        if self._api_key is None:
            raise ValueError(
                "VisionEvents block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication"
                "#retrieve-an-api-key to learn how to retrieve one."
            )
        # Determine batch size from whichever image input is provided
        batch_size = 1
        if input_image is not None:
            batch_size = len(input_image)
        elif output_image is not None:
            batch_size = len(output_image)
        elif predictions is not None:
            batch_size = len(predictions)

        if disable_sink:
            return [
                {
                    "error_status": False,
                    "message": "Sink was disabled by parameter `disable_sink`",
                }
                for _ in range(batch_size)
            ]

        input_images = [None] * batch_size if input_image is None else input_image
        output_images = [None] * batch_size if output_image is None else output_image
        predictions_list = [None] * batch_size if predictions is None else predictions

        result = []
        for img_in, img_out, pred in zip(input_images, output_images, predictions_list):
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
                _execute_vision_event,
                api_base_url=API_BASE_URL,
                api_key=self._api_key,
                input_image=img_in,
                output_image=img_out,
                prediction=pred,
                event_type=event_type,
                solution=solution,
                event_data=event_data,
                custom_metadata=custom_metadata,
            )

            if fire_and_forget and self._background_tasks:
                self._background_tasks.add_task(task)
                result.append(
                    {
                        "error_status": False,
                        "message": "Vision event sent in background task",
                    }
                )
            elif fire_and_forget and self._thread_pool_executor:
                self._thread_pool_executor.submit(task)
                result.append(
                    {
                        "error_status": False,
                        "message": "Vision event sent in background task",
                    }
                )
            else:
                error_status, message = task()
                result.append({"error_status": error_status, "message": message})

        return result


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
        data = {}
    return {k: v for k, v in data.items() if v is not None}


def _execute_vision_event(
    api_base_url: str,
    api_key: str,
    input_image: Optional[WorkflowImageData],
    output_image: Optional[WorkflowImageData],
    prediction: Optional[Union[sv.Detections, dict]],
    event_type: str,
    solution: str,
    event_data: Dict[str, Any],
    custom_metadata: Dict[str, Any],
) -> Tuple[bool, str]:
    try:
        # Step 1: Convert predictions to vision events format
        object_detections: List[dict] = []
        classifications: List[dict] = []
        instance_segmentations: List[dict] = []
        keypoints_detections: List[dict] = []

        if prediction is not None:
            if isinstance(prediction, sv.Detections) and len(prediction) > 0:
                (
                    object_detections,
                    classifications,
                    instance_segmentations,
                    keypoints_detections,
                ) = _convert_sv_detections_to_vision_events_format(prediction)
            elif isinstance(prediction, dict):
                classifications = _convert_classification_to_vision_events_format(
                    prediction
                )

        # Step 2: Upload images and build images array
        images_payload: List[dict] = []
        # Track which image entry should carry annotations. Prefer input_image,
        # fall back to output_image so predictions are never silently dropped.
        annotation_target: Optional[Dict[str, Any]] = None

        if output_image is not None:
            output_source_id, _ = _upload_image(api_base_url, api_key, output_image)
            output_entry: Dict[str, Any] = {
                "label": "output",
                "sourceId": output_source_id,
            }
            images_payload.append(output_entry)
            annotation_target = output_entry

        if input_image is not None:
            input_source_id, _ = _upload_image(api_base_url, api_key, input_image)
            input_image_entry: Dict[str, Any] = {
                "label": "input",
                "sourceId": input_source_id,
                "inputSourceId": input_source_id,
            }
            images_payload.append(input_image_entry)
            annotation_target = input_image_entry

        if annotation_target is not None:
            if object_detections:
                annotation_target["objectDetections"] = object_detections
            if classifications:
                annotation_target["classifications"] = classifications
            if instance_segmentations:
                annotation_target["instanceSegmentations"] = instance_segmentations
            if keypoints_detections:
                annotation_target["keypoints"] = keypoints_detections

        # Step 3: Build and send event
        payload = _build_event_payload(
            event_type=event_type,
            solution=solution,
            images=images_payload,
            event_data=event_data,
            custom_metadata=custom_metadata,
        )

        return _send_event(api_base_url, api_key, payload)
    except Exception as error:
        logger.warning(f"Failed to create vision event: {error}")
        return True, f"Error creating vision event: {type(error).__name__}: {error}"


def _detect_prediction_type(detections: sv.Detections) -> str:
    """Determine detection type from sv.Detections data."""
    if len(detections) == 0:
        return "object_detection"

    data = detections.data
    # Keypoints are most specific
    if KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in data:
        return "keypoint_detection"
    # Instance segmentation has masks or polygons
    if detections.mask is not None or POLYGON_KEY_IN_SV_DETECTIONS in data:
        return "instance_segmentation"
    # Check prediction_type field
    if PREDICTION_TYPE_KEY in data and len(data[PREDICTION_TYPE_KEY]) > 0:
        ptype = str(data[PREDICTION_TYPE_KEY][0])
        if "segmentation" in ptype:
            return "instance_segmentation"
        if "keypoint" in ptype:
            return "keypoint_detection"
    return "object_detection"


def _convert_sv_detections_to_vision_events_format(
    detections: sv.Detections,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Convert sv.Detections to vision events format.

    Returns:
        Tuple of (object_detections, classifications, instance_segmentations, keypoints)
        using center-based bounding boxes in absolute pixel coordinates.
    """
    detection_type = _detect_prediction_type(detections)

    object_detections: List[dict] = []
    classifications: List[dict] = []
    instance_segmentations: List[dict] = []
    keypoints_list: List[dict] = []

    for xyxy, mask, confidence, _class_id, _tracker_id, data in detections:
        if isinstance(xyxy, np.ndarray):
            # Explicitly cast each coordinate to float to match previous behavior
            x1 = float(xyxy[0])
            y1 = float(xyxy[1])
            x2 = float(xyxy[2])
            y2 = float(xyxy[3])
        else:
            x1, y1, x2, y2 = xyxy

        w = abs(x2 - x1)
        h = abs(y2 - y1)
        cx = x1 + w / 2
        cy = y1 + h / 2
        class_name = str(data.get("class_name", "unknown"))
        conf = float(confidence) if confidence is not None else 0.0
        # Clamp confidence to [0, 1]
        if conf < 0.0:
            conf = 0.0
        elif conf > 1.0:
            conf = 1.0

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
            if polygon is None and mask is not None:
                polygon = mask_to_polygon(mask=mask)
            if polygon is not None and len(polygon) >= 3:
                seg_entry["points"] = [[float(pt[0]), float(pt[1])] for pt in polygon]
                instance_segmentations.append(seg_entry)
            else:
                # Fall back to object detection if polygon is invalid
                object_detections.append(base)

        else:
            object_detections.append(base)

    return object_detections, classifications, instance_segmentations, keypoints_list


def _convert_classification_to_vision_events_format(
    prediction: dict,
) -> List[dict]:
    """Convert classification dict to vision events classification format.

    Handles both multi-class (list) and multi-label (dict) prediction formats.
    """
    classifications: List[dict] = []

    # Check for "predicted_classes" (standard classification output)
    predicted_classes = prediction.get("predicted_classes")
    if isinstance(predicted_classes, list):
        for class_name in predicted_classes:
            conf = 0.0
            # Try to get confidence from predictions dict
            preds = prediction.get("predictions", {})
            if isinstance(preds, dict) and class_name in preds:
                conf = float(preds[class_name].get("confidence", 0.0))
            classifications.append({"class": str(class_name), "confidence": conf})
        if classifications:
            return classifications

    preds = prediction.get("predictions", [])

    if isinstance(preds, list):
        # Multi-class: [{"class_name": "A", "confidence": 0.9}, ...]
        for p in preds:
            class_name = p.get("class_name", p.get("class", "unknown"))
            conf = float(p.get("confidence", 0.0))
            classifications.append({"class": str(class_name), "confidence": conf})
    elif isinstance(preds, dict):
        # Multi-label: {"A": {"confidence": 0.9, "class_id": 0}, ...}
        for class_name, details in preds.items():
            if isinstance(details, dict):
                conf = float(details.get("confidence", 0.0))
            else:
                conf = float(details) if isinstance(details, (int, float)) else 0.0
            classifications.append({"class": str(class_name), "confidence": conf})

    # Fallback: check "top" field for single classification
    if not classifications and "top" in prediction:
        conf = float(prediction.get("confidence", 0.0))
        classifications.append({"class": str(prediction["top"]), "confidence": conf})

    return classifications


def _upload_image(
    api_base_url: str,
    api_key: str,
    image_data: WorkflowImageData,
) -> Tuple[str, str]:
    """Upload an image to the Vision Events API.

    Returns:
        Tuple of (sourceId, url)
    """
    image_bytes = encode_image_to_jpeg_bytes(image_data.numpy_image, jpeg_quality=95)
    response = requests.post(
        f"{api_base_url}/vision-events/upload",
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
    from datetime import datetime, timezone

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
            f"{api_base_url}/vision-events",
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
        logger.warning(f"Vision Events API error ({status_code}): {body}")
        return (
            True,
            f"Failed to send vision event. Status: {status_code}. Details: {body}",
        )
    except Exception as e:
        logger.warning(f"Failed to send vision event: {e}")
        return True, f"Failed to send vision event. Error: {type(e).__name__}: {e}"
