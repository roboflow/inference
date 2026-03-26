import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import requests
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
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
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Event Writer** block sends structured events to the Event Ingestion Service
using the v2 API.

## Supported Event Schemas

* **Quality Check** — pass/fail inspection results
* **Inventory Count** — item counts at a location
* **Safety Alert** — safety incidents with severity levels
* **Custom** — free-form events with an arbitrary value field

## Images

Each event includes one image entry. You must provide an **output image** (the
primary display image, typically a visualization). You can optionally attach an
**input image** (the original frame before any annotation).

## Annotations

You can optionally pass object detection, classification, instance segmentation,
or keypoint predictions from upstream model blocks. These are stored as
structured annotations on the image within the event.

## Execution Modes

* **Fire-and-forget** (`fire_and_forget=True`, default) — the HTTP request is
  dispatched in the background so the workflow continues immediately. The
  `event_id` output will be empty.
* **Synchronous** (`fire_and_forget=False`) — the block waits for the response
  and returns the created `event_id`.

## Rate Limiting

Use the **Rate Limiter** workflow block upstream of this block to control how
often events are sent.

## Authentication

If the Event Ingestion Service requires an API key, set the
`EVENT_INGESTION_API_KEY` environment variable on the inference server.
Requests are sent unauthenticated when the variable is not set.
"""

QUALITY_CHECK_RELEVANT = {
    "event_schema": {"values": ["quality_check"], "required": True},
}
INVENTORY_COUNT_RELEVANT = {
    "event_schema": {"values": ["inventory_count"], "required": True},
}
SAFETY_ALERT_RELEVANT = {
    "event_schema": {"values": ["safety_alert"], "required": True},
}
CUSTOM_RELEVANT = {
    "event_schema": {"values": ["custom"], "required": True},
}
ALL_DATA_SCHEMAS_RELEVANT = {
    "event_schema": {
        "values": ["quality_check", "inventory_count", "safety_alert", "custom"],
    },
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Event Writer",
            "version": "v1",
            "short_description": "Write structured events to the Event Ingestion Service.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-calendar-check",
                "blockPriority": 5,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal["roboflow_enterprise/event_writer_sink@v1"]

    # --- Connection ---
    event_ingestion_url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="http://localhost:8001",
        description="Base URL of the Event Ingestion Service.",
        examples=["http://localhost:8001", "$inputs.event_ingestion_url"],
    )

    # --- Schema selector ---
    event_schema: Literal["quality_check", "inventory_count", "safety_alert", "custom"] = Field(
        description="The event schema to use.",
        json_schema_extra={"always_visible": True},
    )

    # --- Images ---
    output_image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The output/visualization image. Sent as the primary display image.",
        json_schema_extra={"always_visible": True},
    )
    input_image: Optional[Selector(kind=[IMAGE_KIND])] = Field(
        default=None,
        description="The original input image (optional). Sent as the source image.",
    )
    image_label: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Label for the image entry.",
        examples=["defect-analysis", "$inputs.image_label"],
    )

    # --- Custom metadata ---
    custom_metadata: Dict[
        str,
        Union[
            Selector(),
            str,
            float,
            bool,
            int,
        ],
    ] = Field(
        default_factory=dict,
        description="Flat key-value metadata (max 100 keys, values must be str/int/float/bool).",
        examples=[{"line": "A1", "shift": "morning"}],
        json_schema_extra={"always_visible": True},
    )

    # --- Quality Check fields ---
    qc_result: Optional[Union[Selector(kind=[STRING_KIND]), Literal["pass", "fail"]]] = Field(
        default=None,
        description="Quality check result: pass or fail.",
        examples=["pass", "fail", "$steps.qc_logic.result"],
        json_schema_extra={
            "relevant_for": QUALITY_CHECK_RELEVANT,
            "always_visible": True,
        },
    )

    # --- External ID (shared across schemas) ---
    external_id: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="External identifier for correlation with other systems (max 1000 chars).",
        examples=["batch-2025-001", "$inputs.external_id"],
        json_schema_extra={
            "relevant_for": ALL_DATA_SCHEMAS_RELEVANT,
        },
    )

    # --- Inventory Count fields ---
    location: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Location identifier for inventory count.",
        examples=["warehouse-A", "$inputs.location"],
        json_schema_extra={"relevant_for": INVENTORY_COUNT_RELEVANT},
    )
    item_count: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=None,
        description="Number of items counted.",
        examples=[42, "$steps.counter.count"],
        json_schema_extra={"relevant_for": INVENTORY_COUNT_RELEVANT},
    )
    item_type: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Type of item being counted.",
        examples=["widget", "$inputs.item_type"],
        json_schema_extra={"relevant_for": INVENTORY_COUNT_RELEVANT},
    )

    # --- Safety Alert fields ---
    alert_type: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Alert type identifier (alphanumeric, underscores, hyphens).",
        examples=["no_hardhat", "$steps.classifier.top_class"],
        json_schema_extra={"relevant_for": SAFETY_ALERT_RELEVANT},
    )
    severity: Optional[
        Union[Selector(kind=[STRING_KIND]), Literal["low", "medium", "high"]]
    ] = Field(
        default=None,
        description="Severity level for the safety alert.",
        examples=["high", "$inputs.severity"],
        json_schema_extra={"relevant_for": SAFETY_ALERT_RELEVANT},
    )
    alert_description: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Description of the safety alert (max 10000 chars).",
        examples=["Worker detected without hardhat in zone B"],
        json_schema_extra={"relevant_for": SAFETY_ALERT_RELEVANT},
    )

    # --- Custom Event fields ---
    custom_value: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Arbitrary value for custom events (max 10000 chars).",
        examples=["anomaly detected at 14:32"],
        json_schema_extra={"relevant_for": CUSTOM_RELEVANT},
    )

    # --- Annotation pass-through ---
    object_detections: Optional[
        Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND])
    ] = Field(
        default=None,
        description="Object detection predictions to attach to the image.",
        json_schema_extra={"additional_section": True},
    )
    classifications: Optional[
        Selector(kind=[CLASSIFICATION_PREDICTION_KIND])
    ] = Field(
        default=None,
        description="Classification predictions to attach to the image.",
        json_schema_extra={"additional_section": True},
    )
    instance_segmentations: Optional[
        Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND])
    ] = Field(
        default=None,
        description="Instance segmentation predictions to attach to the image.",
        json_schema_extra={"additional_section": True},
    )
    keypoint_detections: Optional[
        Selector(kind=[KEYPOINT_DETECTION_PREDICTION_KIND])
    ] = Field(
        default=None,
        description="Keypoint detection predictions to attach to the image.",
        json_schema_extra={"additional_section": True},
    )

    # --- Execution control ---
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If True, send the event asynchronously (no event_id returned). "
        "If False, wait for the response and return the event_id.",
        examples=[True, False, "$inputs.fire_and_forget"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, skip sending the event entirely.",
        examples=[False, "$inputs.disable_event_writer"],
    )
    request_timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="HTTP request timeout in seconds.",
        examples=[5, 10],
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


class EventWriterSinkBlockV1(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        event_ingestion_url: str,
        event_schema: str,
        output_image: WorkflowImageData,
        fire_and_forget: bool,
        disable_sink: bool,
        request_timeout: int,
        input_image: Optional[WorkflowImageData] = None,
        image_label: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        qc_result: Optional[str] = None,
        external_id: Optional[str] = None,
        location: Optional[str] = None,
        item_count: Optional[int] = None,
        item_type: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        alert_description: Optional[str] = None,
        custom_value: Optional[str] = None,
        object_detections: Optional[Any] = None,
        classifications: Optional[Any] = None,
        instance_segmentations: Optional[Any] = None,
        keypoint_detections: Optional[Any] = None,
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "event_id": "",
                "message": "Sink was disabled by parameter `disable_sink`",
            }

        url = event_ingestion_url.rstrip("/")

        event_data = _build_event_data(
            event_schema=event_schema,
            qc_result=qc_result,
            external_id=external_id,
            location=location,
            item_count=item_count,
            item_type=item_type,
            alert_type=alert_type,
            severity=severity,
            alert_description=alert_description,
            custom_value=custom_value,
        )

        image_entry = _build_image_entry(
            output_image=output_image,
            input_image=input_image,
            image_label=image_label,
            object_detections=object_detections,
            classifications=classifications,
            instance_segmentations=instance_segmentations,
            keypoint_detections=keypoint_detections,
        )

        payload: Dict[str, Any] = {
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "event_schema": event_schema,
            "event_data": event_data,
            "images": [image_entry],
            "displayImagePosition": 0,
        }
        if custom_metadata is not None:
            payload["custom_metadata"] = custom_metadata

        request_handler = partial(
            _execute_event_request,
            url=f"{url}/v2/events",
            payload=payload,
            api_key=os.environ.get("EVENT_INGESTION_API_KEY"),
            timeout=request_timeout,
        )

        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(request_handler)
            return {
                "error_status": False,
                "event_id": "",
                "message": "Event sent in background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(request_handler)
            return {
                "error_status": False,
                "event_id": "",
                "message": "Event sent in background task",
            }

        error_status, message, event_id = request_handler()
        return {
            "error_status": error_status,
            "event_id": event_id,
            "message": message,
        }


def _build_event_data(
    event_schema: str,
    qc_result: Optional[str] = None,
    external_id: Optional[str] = None,
    location: Optional[str] = None,
    item_count: Optional[int] = None,
    item_type: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    alert_description: Optional[str] = None,
    custom_value: Optional[str] = None,
) -> Dict[str, Any]:
    """Build schema-specific event_data dict with camelCase keys, stripping None values."""
    if event_schema == "quality_check":
        data = {"result": qc_result, "externalId": external_id}
    elif event_schema == "inventory_count":
        data = {
            "location": location,
            "itemCount": item_count,
            "itemType": item_type,
            "externalId": external_id,
        }
    elif event_schema == "safety_alert":
        data = {
            "alertType": alert_type,
            "severity": severity,
            "description": alert_description,
            "externalId": external_id,
        }
    elif event_schema == "custom":
        data = {"externalId": external_id, "value": custom_value}
    else:
        data = {}
    # Strip None values — event_data schemas use extra="forbid"
    return {k: v for k, v in data.items() if v is not None}


def _detections_to_v2_object_detections(detections: Any) -> List[Dict[str, Any]]:
    """Convert Supervision Detections to v2 objectDetections format.

    Uses center-based absolute pixel coordinates to match the main Roboflow
    annotation format: x,y = center of box, width,height = full dimensions.
    """
    if detections is None:
        return []
    results = []
    try:
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_names = detections.data.get("class_name", [])
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            results.append({
                "class": class_names[i] if i < len(class_names) else "unknown",
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": w,
                "height": h,
                "confidence": float(confidence[i]) if confidence is not None else 0.0,
            })
    except Exception as e:
        logging.warning(f"Failed to convert object detections: {e}")
    return results


def _detections_to_v2_instance_segmentations(detections: Any) -> List[Dict[str, Any]]:
    """Convert Supervision Detections with masks to v2 instanceSegmentations format."""
    if detections is None:
        return []
    results = []
    try:
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_names = detections.data.get("class_name", [])
        masks = detections.mask
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            entry = {
                "class": class_names[i] if i < len(class_names) else "unknown",
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": w,
                "height": h,
                "confidence": float(confidence[i]) if confidence is not None else 0.0,
                "points": [],
            }
            if masks is not None and i < len(masks):
                mask = masks[i].astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    entry["points"] = [
                        [float(pt[0][0]), float(pt[0][1])] for pt in largest
                    ]
            if len(entry["points"]) >= 3:
                results.append(entry)
    except Exception as e:
        logging.warning(f"Failed to convert instance segmentations: {e}")
    return results


def _classifications_to_v2(predictions: Any) -> List[Dict[str, Any]]:
    """Convert classification predictions to v2 classifications format."""
    if predictions is None:
        return []
    results = []
    try:
        if isinstance(predictions, dict):
            class_names = predictions.get("class_name", [])
            confidences = predictions.get("confidence", [])
            for i in range(len(class_names)):
                results.append({
                    "class": class_names[i],
                    "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
                })
        elif hasattr(predictions, "data"):
            class_names = predictions.data.get("class_name", [])
            confidences = predictions.confidence
            for i in range(len(class_names)):
                results.append({
                    "class": class_names[i],
                    "confidence": float(confidences[i]) if confidences is not None and i < len(confidences) else 0.0,
                })
    except Exception as e:
        logging.warning(f"Failed to convert classifications: {e}")
    return results


def _keypoints_to_v2(detections: Any) -> List[Dict[str, Any]]:
    """Convert keypoint detection predictions to v2 keypoints format."""
    if detections is None:
        return []
    results = []
    try:
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_names = detections.data.get("class_name", [])
        keypoints_xy = detections.data.get("keypoints_xy")
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            entry = {
                "class": class_names[i] if i < len(class_names) else "unknown",
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": w,
                "height": h,
                "confidence": float(confidence[i]) if confidence is not None else 0.0,
                "keypoints": [],
            }
            if keypoints_xy is not None and i < len(keypoints_xy):
                kps = keypoints_xy[i]
                for j, kp in enumerate(kps):
                    entry["keypoints"].append({
                        "id": j,
                        "x": float(kp[0]),
                        "y": float(kp[1]),
                    })
            if entry["keypoints"]:
                results.append(entry)
    except Exception as e:
        logging.warning(f"Failed to convert keypoint detections: {e}")
    return results


def _build_image_entry(
    output_image: WorkflowImageData,
    input_image: Optional[WorkflowImageData] = None,
    image_label: Optional[str] = None,
    object_detections: Optional[Any] = None,
    classifications: Optional[Any] = None,
    instance_segmentations: Optional[Any] = None,
    keypoint_detections: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the single image object for the v2 API payload."""
    entry: Dict[str, Any] = {
        "base64Image": output_image.base64_image,
    }
    if input_image is not None:
        entry["inputBase64Image"] = input_image.base64_image
    if image_label is not None:
        entry["label"] = image_label

    od = _detections_to_v2_object_detections(object_detections)
    if od:
        entry["objectDetections"] = od

    cl = _classifications_to_v2(classifications)
    if cl:
        entry["classifications"] = cl

    seg = _detections_to_v2_instance_segmentations(instance_segmentations)
    if seg:
        entry["instanceSegmentations"] = seg

    kp = _keypoints_to_v2(keypoint_detections)
    if kp:
        entry["keypoints"] = kp

    return entry


def _execute_event_request(
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    timeout: int,
) -> Tuple[bool, str, str]:
    """POST the event to the ingestion service.

    Returns:
        (error_status, message, event_id)
    """
    try:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)

        if response.status_code == 201:
            data = response.json()
            event_id = str(data.get("id", ""))
            return False, "Event created successfully", event_id

        if response.status_code == 529:
            detail = _extract_detail(response)
            return (
                True,
                f"Event Ingestion Service at capacity (529). "
                f"The device is experiencing backpressure. Detail: {detail}",
                "",
            )

        detail = _extract_detail(response)
        return True, f"HTTP {response.status_code}: {detail}", ""
    except requests.exceptions.Timeout:
        return True, f"Request timed out after {timeout}s", ""
    except Exception as e:
        logging.warning(f"Event writer request failed: {e}")
        return True, f"Request failed: {e}", ""


def _extract_detail(response: requests.Response) -> str:
    """Try to extract 'detail' from a JSON error response."""
    try:
        return str(response.json().get("detail", response.text))
    except Exception:
        return response.text
