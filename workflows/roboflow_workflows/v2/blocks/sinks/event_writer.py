"""Event writer sink block: send one structured event to the Event Ingestion Service.

Port of V1 ``roboflow_enterprise/event_writer_sink@v1``::

    disabled?  ─▶ error_status=False, event_id="", V1's disabled message
    schema fields ─▶ event_data (camelCase, None dropped)
    images + predictions ─▶ one image entry (base64 JPEG, annotations)
    payload ─▶ POST {event_ingestion_url}/v2/events
               fire_and_forget and sink_effects ─▶ background, event_id=""
               else sent now ─▶ 201: event_id from the response body

The payload is built in the calling thread, as in V1, so the background
effect holds only JSON data, never workflow images or predictions.

Annotation formats (absolute pixels; x, y is the box center)::

    objectDetections       {class, x, y, width, height, confidence}
    instanceSegmentations  same + points: largest outer mask contour
                           (rows with fewer than 3 points are left out)
    keypoints              same + keypoints: [{id, x, y}] of the slots with
                           confidence > 0; id is the skeleton slot (rows
                           without such slots are left out)
    classifications        {class, confidence}: the top class, or the
                           predicted classes of a multi-label prediction

Event data, boxes, masks, JPEG quality 95, response handling and disabled
outputs match V1 on equal inputs. Differences from V1, each checked
against V1's code:

- ``fire_and_forget`` uses the host's ``sink_effects`` (V1 used FastAPI
  ``BackgroundTasks`` or a thread pool).
- The API key is the ``event_ingestion_api_key`` resource, a str or
  ``Secret``; messages are scrubbed of it. V1 read ``EVENT_INGESTION_API_KEY``
  from the environment on every call.
- Classifications are sent. V1 read a ``class_name`` key that no V1
  classifier output has, so it never sent any. A class id without a name
  fails the step.
- Annotations are sent for ``inference_models`` predictions. V1 in tensor
  mode received them without ``.data`` and sent no boxes, masks or keypoints.
- A keypoint ``id`` is its skeleton slot. V1 numbered the sent keypoints, so
  one below the threshold shifted the ids after it. Keypoints need boxes.
- Images are always encoded from pixels; V1 reused an input's original base64.
- ``request_timeout`` must be positive.
"""

import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.formatters.values import WallClock
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    BackgroundEffects,
    EffectOutcome,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.blocks.visualization.annotations.classification_label import (  # noqa: E501
    shown_classes,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id, row_class_names
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import decode_rle_masks
from roboflow_workflows.v2.data.secrets import (
    Revealed,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.types import InstancesRLEMasks

BLOCK_TYPE = "v2/event_writer_sink@v1"
CREATED_MESSAGE = "Event created successfully"
BACKGROUND_EVENT_MESSAGE = "Event sent in background task"
# V1 WorkflowImageData.base64_image quality.
JPEG_QUALITY = 95

EventSchema = Literal["quality_check", "inventory_count", "safety_alert", "custom"]
Scalar = Union[StrictStr, StrictInt, StrictFloat, StrictBool]
SCALAR_KINDS = (STRING_KIND, INTEGER_KIND, FLOAT_KIND, BOOLEAN_KIND)
OptionalText = Optional[StrictStr] | Ref(STRING_KIND)


class EventWriterSinkBlock(Block):
    """Send a quality check, inventory count, safety alert or custom event.

    Example: ``event_schema="quality_check"``, ``qc_result="fail"``,
    ``output_image="$steps.boxes.image"`` and
    ``object_detections="$steps.model.predictions"`` post one event whose image
    carries the drawn frame and its boxes.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Event Writer",
        "port_of": "roboflow_enterprise/event_writer_sink@v1",
        "section": "data_storage",
    }
    outputs = {
        "error_status": ERROR_STATUS_OUTPUT,
        "event_id": Output(
            STRING_KIND,
            description="ID of the created event; empty unless sent now and created.",
        ),
        "message": MESSAGE_OUTPUT,
    }

    class Params(BlockParams):
        event_ingestion_url: StrictStr | Ref(STRING_KIND) = Field(
            default="http://localhost:8001",
            description="Base URL of the Event Ingestion Service.",
            examples=["http://localhost:8001"],
        )
        event_schema: EventSchema = Field(
            description="Event type; selects which event fields are sent.",
            examples=["quality_check"],
        )
        output_image: Ref(IMAGE_KIND) = Field(
            description="Primary display image, typically a visualization.",
            examples=["$steps.boxes.image"],
        )
        input_image: Optional[Ref(IMAGE_KIND)] = Field(
            default=None,
            description="Original frame before annotation, sent as the source image.",
            examples=["$inputs.image"],
        )
        image_label: OptionalText = Field(
            default=None,
            description="Label of the image entry.",
            examples=["defect-analysis"],
        )
        custom_metadata: Dict[str, Scalar | Ref(*SCALAR_KINDS)] = Field(  # noqa: F821
            default_factory=dict,
            description="Flat metadata: string, number or boolean values by key.",
            examples=[{"line": "A1", "shift": "morning"}],
        )
        qc_result: Optional[Literal["pass", "fail"]] | Ref(STRING_KIND) = Field(
            default=None,
            description="quality_check: pass or fail.",
            examples=["fail"],
        )
        external_id: OptionalText = Field(
            default=None,
            description="Identifier for correlation with other systems (any schema).",
            examples=["batch-2025-001"],
        )
        location: OptionalText = Field(
            default=None,
            description="inventory_count: location identifier.",
            examples=["warehouse-A"],
        )
        item_count: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            description="inventory_count: number of items counted.",
            examples=[42],
        )
        item_type: OptionalText = Field(
            default=None,
            description="inventory_count: type of item counted.",
            examples=["widget"],
        )
        alert_type: OptionalText = Field(
            default=None,
            description="safety_alert: alert type identifier.",
            examples=["no_hardhat"],
        )
        severity: Optional[Literal["low", "medium", "high"]] | Ref(STRING_KIND) = Field(
            default=None,
            description="safety_alert: low, medium or high.",
            examples=["high"],
        )
        alert_description: OptionalText = Field(
            default=None,
            description="safety_alert: description.",
            examples=["Worker without hardhat in zone B"],
        )
        custom_value: OptionalText = Field(
            default=None,
            description="custom: free-form value.",
            examples=["anomaly detected at 14:32"],
        )
        object_detections: Optional[Ref(OBJECT_DETECTION_PREDICTION_KIND)] = Field(
            default=None,
            description="Boxes to attach to the image.",
            examples=["$steps.model.predictions"],
        )
        classifications: Optional[Ref(CLASSIFICATION_PREDICTION_KIND)] = Field(
            default=None,
            description="Classification to attach: top or predicted classes.",
            examples=["$steps.classifier.predictions"],
        )
        instance_segmentations: Optional[Ref(INSTANCE_SEGMENTATION_PREDICTION_KIND)] = (
            Field(
                default=None,
                description="Instance masks to attach, sent as contour polygons.",
                examples=["$steps.segmenter.predictions"],
            )
        )
        keypoint_detections: Optional[Ref(KEYPOINT_DETECTION_PREDICTION_KIND)] = Field(
            default=None,
            description="Keypoints with their boxes to attach.",
            examples=["$steps.pose.predictions"],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Do not wait for the response when the host provides background "
                "effects (event_id is then empty); otherwise send now."
            ),
            examples=[True],
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip sending; outputs say the sink was disabled.",
            examples=[False],
        )
        request_timeout: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            gt=0,
            description="Seconds to wait for the response.",
            examples=[5],
        )

    def __init__(
        self,
        *,
        sink_effects: Optional[BackgroundEffects] = None,
        event_ingestion_api_key: Optional[Union[str, Secret]] = None,
        disable_sinks: bool = False,
        wall_clock: WallClock = datetime.now,
    ) -> None:
        self._sink_effects = sink_effects
        self._api_key = to_secret(
            event_ingestion_api_key, name="event_ingestion_api_key"
        )
        self._disable_sinks = disable_sinks
        self._wall_clock = wall_clock

    def run(
        self,
        *,
        event_ingestion_url: str,
        event_schema: str,
        output_image: ImageData,
        input_image: Optional[ImageData],
        image_label: Optional[str],
        custom_metadata: Dict[str, Any],
        qc_result: Optional[str],
        external_id: Optional[str],
        location: Optional[str],
        item_count: Optional[int],
        item_type: Optional[str],
        alert_type: Optional[str],
        severity: Optional[str],
        alert_description: Optional[str],
        custom_value: Optional[str],
        object_detections: Any,
        classifications: Any,
        instance_segmentations: Any,
        keypoint_detections: Any,
        fire_and_forget: bool,
        disable_sink: bool,
        request_timeout: int,
    ) -> Dict[str, Any]:
        """Build and send one event unless disabled.

        Args:
            event_ingestion_url: Base URL of the service.
            event_schema: Event type.
            output_image: Primary display image.
            input_image: Source image, or ``None``.
            image_label: Image entry label, or ``None``.
            custom_metadata: Flat metadata.
            qc_result: quality_check result.
            external_id: Correlation identifier.
            location: inventory_count location.
            item_count: inventory_count count.
            item_type: inventory_count item type.
            alert_type: safety_alert type.
            severity: safety_alert severity.
            alert_description: safety_alert description.
            custom_value: custom value.
            object_detections: Boxes, or ``None``.
            classifications: Classification, or ``None``.
            instance_segmentations: Instance masks, or ``None``.
            keypoint_detections: Keypoints with boxes, or ``None``.
            fire_and_forget: Hand the request to background effects if any.
            disable_sink: Skip this event.
            request_timeout: Response timeout in seconds.

        Returns:
            ``error_status``, ``event_id`` and ``message``.

        Raises:
            ValueError: When a prediction row cannot be named or keypoints
                come without boxes.
            SecretUnavailableError: When the API key is an unavailable secret.
        """
        if self._disable_sinks or disable_sink:
            message = disabled_sink_message(by_policy=self._disable_sinks)
            return {"error_status": False, "event_id": "", "message": message}

        event_data = {
            "quality_check": {"result": qc_result, "externalId": external_id},
            "inventory_count": {
                "location": location,
                "itemCount": item_count,
                "itemType": item_type,
                "externalId": external_id,
            },
            "safety_alert": {
                "alertType": alert_type,
                "severity": severity,
                "description": alert_description,
                "externalId": external_id,
            },
            "custom": {"externalId": external_id, "value": custom_value},
        }[event_schema]
        payload = {
            "inference_timestamp": self._wall_clock(timezone.utc).isoformat(),
            "event_schema": event_schema,
            # The service's event_data schemas forbid extra (None) fields.
            "event_data": {k: v for k, v in event_data.items() if v is not None},
            "images": [
                image_entry(
                    output_image,
                    input_image=input_image,
                    image_label=image_label,
                    object_detections=object_detections,
                    classifications=classifications,
                    instance_segmentations=instance_segmentations,
                    keypoint_detections=keypoint_detections,
                )
            ],
            "displayImagePosition": 0,
            "custom_metadata": dict(custom_metadata),
        }
        revealed = reveal_secrets(self._api_key)
        url = f"{event_ingestion_url.rstrip('/')}/v2/events"
        event_ids: List[str] = []

        def effect() -> EffectOutcome:
            error_status, message, event_id = post_event(
                url, payload, revealed=revealed, timeout=request_timeout
            )
            event_ids.append(event_id)
            return error_status, message

        error_status, message = deliver(
            effect,
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=BLOCK_TYPE,
        )
        if (error_status, message) == (False, BACKGROUND_MESSAGE):
            return {
                "error_status": False,
                "event_id": "",
                "message": BACKGROUND_EVENT_MESSAGE,
            }

        # The effect ran in this thread (now or inline), or was refused unrun.
        event_id = event_ids[0] if event_ids else ""

        return {"error_status": error_status, "event_id": event_id, "message": message}


def post_event(
    url: str, payload: Dict[str, Any], *, revealed: Revealed, timeout: int
) -> Tuple[bool, str, str]:
    """POST one event and return V1's outcome.

    Args:
        url: ``<base>/v2/events``.
        payload: JSON body.
        revealed: The revealed API key (``value`` is ``None`` without one).
        timeout: Response timeout in seconds.

    Returns:
        ``(error_status, message, event_id)``; messages are scrubbed of the key.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    if revealed.value:
        headers["X-API-Key"] = revealed.value
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return True, f"Request timed out after {timeout}s", ""
    except Exception as error:
        return True, revealed.scrub(f"Request failed: {error}"), ""

    if response.status_code == 201:
        try:
            event_id = str(response.json().get("id", ""))
        except Exception as error:
            return True, revealed.scrub(f"Request failed: {error}"), ""
        return False, CREATED_MESSAGE, event_id

    detail = revealed.scrub(response_detail(response))
    if response.status_code == 529:
        message = (
            "Event Ingestion Service at capacity (529). The device is "
            f"experiencing backpressure. Detail: {detail}"
        )
        return True, message, ""

    outcome = (True, f"HTTP {response.status_code}: {detail}", "")

    return outcome


def response_detail(response: Any) -> str:
    """Return the ``detail`` of a JSON error body, else the body text.

    Args:
        response: A ``requests`` response.

    Returns:
        The detail text.
    """
    try:
        detail = str(response.json().get("detail", response.text))
    except Exception:
        detail = response.text

    return detail


def image_entry(
    output_image: ImageData,
    *,
    input_image: Optional[ImageData],
    image_label: Optional[str],
    object_detections: Any,
    classifications: Any,
    instance_segmentations: Any,
    keypoint_detections: Any,
) -> Dict[str, Any]:
    """Build the event's single image entry; empty annotation lists are left out.

    Args:
        output_image: Primary display image.
        input_image: Source image, or ``None``.
        image_label: Label, or ``None``.
        object_detections: ``Detections``, or ``None``.
        classifications: A classification prediction, or ``None``.
        instance_segmentations: ``InstanceDetections``, or ``None``.
        keypoint_detections: ``(KeyPoints, Detections | None)``, or ``None``.

    Returns:
        The entry.

    Raises:
        ValueError: When a row cannot be named or keypoints have no boxes.
    """
    entry: Dict[str, Any] = {"base64Image": base64_jpeg(output_image)}
    if input_image is not None:
        entry["inputBase64Image"] = base64_jpeg(input_image)
    if image_label is not None:
        entry["label"] = image_label

    for name, prediction, to_entries in (
        ("objectDetections", object_detections, box_entries),
        ("classifications", classifications, classification_entries),
        ("instanceSegmentations", instance_segmentations, segmentation_entries),
        ("keypoints", keypoint_detections, keypoint_entries),
    ):
        rows = [] if prediction is None else to_entries(prediction)
        if rows:
            entry[name] = rows

    return entry


def base64_jpeg(image: ImageData) -> str:
    """Encode an image as base64 JPEG, as V1's ``base64_image`` did.

    Args:
        image: RGB or gray image.

    Returns:
        ASCII base64 text.

    Raises:
        ValueError: When OpenCV cannot encode the image.
    """
    # Host boundary: the pixels leave their device here, once.
    hwc = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    if image.channels == 1:
        pixels = np.ascontiguousarray(hwc[:, :, 0])
    else:
        pixels = cv2.cvtColor(np.ascontiguousarray(hwc), cv2.COLOR_RGB2BGR)
    encoded, jpeg = cv2.imencode(
        ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not encoded:
        raise ValueError(f"OpenCV could not encode an image of shape {pixels.shape}")

    text = base64.b64encode(jpeg.tobytes()).decode("ascii")

    return text


def box_entries(boxes: Any) -> List[Dict[str, Any]]:
    """Return ``{class, x, y, width, height, confidence}`` per box row.

    Centers and sizes are computed in the boxes' own dtype, as V1 did.

    Args:
        boxes: ``Detections`` or ``InstanceDetections``.

    Returns:
        One entry per row; a row without a class name is ``"unknown"``.
    """
    rows = len(boxes)
    if rows == 0:
        return []

    xyxy = boxes.xyxy.detach()
    centers = ((xyxy[:, :2] + xyxy[:, 2:]) / 2).cpu().tolist()
    sizes = (xyxy[:, 2:] - xyxy[:, :2]).cpu().tolist()
    confidences = boxes.confidence.detach().cpu().tolist()
    names = row_class_names(
        boxes.class_id.cpu().tolist(),
        row_metadata=boxes.bboxes_metadata or [{}] * rows,
        names_by_id=class_names_by_id(boxes.image_metadata),
    )
    entries = [
        {
            "class": "unknown" if name is None else str(name),
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height),
            "confidence": float(confidence),
        }
        for name, (x, y), (width, height), confidence in zip(
            names, centers, sizes, confidences
        )
    ]

    return entries


def segmentation_entries(instances: Any) -> List[Dict[str, Any]]:
    """Return box entries with ``points``: each mask's largest outer contour.

    Args:
        instances: ``InstanceDetections`` with dense or RLE masks.

    Returns:
        Entries of rows whose contour has at least 3 points.
    """
    entries = box_entries(instances)
    if not entries:
        return []

    if isinstance(instances.mask, InstancesRLEMasks):
        masks = decode_rle_masks(instances.mask)
    else:
        # Host boundary: contours are traced by OpenCV.
        masks = instances.mask.detach().cpu().numpy()
    kept = []
    for entry, mask in zip(entries, masks):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            continue
        points = [[float(x), float(y)] for x, y in largest[:, 0, :].tolist()]
        kept.append({**entry, "points": points})

    return kept


def keypoint_entries(prediction: Tuple[Any, Any]) -> List[Dict[str, Any]]:
    """Return box entries with ``keypoints``: ``[{id, x, y}]`` per skeleton.

    A slot with confidence 0 is absent: models zero the slots below their
    keypoint threshold and those past a shorter skeleton. ``id`` is the slot
    index, the keypoint's place in its class skeleton.

    Args:
        prediction: ``(KeyPoints, Detections)``; one box per skeleton.

    Returns:
        Entries of rows with at least one present keypoint.

    Raises:
        ValueError: When the prediction has no boxes.
    """
    key_points, boxes = prediction
    if boxes is None:
        raise ValueError(
            f"{BLOCK_TYPE}: keypoint_detections need boxes; the event format "
            "places keypoints in a box"
        )

    entries = box_entries(boxes)
    xy = key_points.xy.detach().cpu().tolist()
    present = (key_points.confidence.detach() > 0).cpu().tolist()
    kept = []
    for entry, skeleton, shown in zip(entries, xy, present):
        points = [
            {"id": index, "x": float(x), "y": float(y)}
            for index, ((x, y), is_present) in enumerate(zip(skeleton, shown))
            if is_present
        ]
        if points:
            kept.append({**entry, "keypoints": points})

    return kept


def classification_entries(prediction: Any) -> List[Dict[str, Any]]:
    """Return ``{class, confidence}`` of the top or predicted classes.

    Args:
        prediction: ``ClassificationPrediction`` or
            ``MultiLabelClassificationPrediction`` of one image.

    Returns:
        One entry per shown class.

    Raises:
        ValueError: When a class id has no name.
    """
    entries = [
        {"class": name, "confidence": confidence}
        for _, name, confidence in shown_classes(prediction)
    ]

    return entries
