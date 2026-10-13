"""Unit tests of ``v2/event_writer_sink@v1``.

Events go over real HTTP to ``EventService`` on ``127.0.0.1:<free port>``;
nothing leaves this process's loopback interface. The API key is invented.
"""

import base64
import json
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Tuple

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.sinks.event_writer import (
    EventWriterSinkBlock,
    box_entries,
    keypoint_entries,
    segmentation_entries,
)
from roboflow_workflows.v2.blocks.sinks.runtime import FULL_MESSAGE, BackgroundEffects
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

API_KEY = "test-EVENT-api-key-41b7"
NAMES = {"class_names": {0: "cat", 1: "dog"}}


class EventService:
    """Loopback stand-in for the Event Ingestion Service.

    Args:
        status: Response status.
        body: Response body; a dict is sent as JSON.
        delay: Seconds to wait before answering.
    """

    def __init__(
        self, *, status: int = 201, body: Any = None, delay: float = 0
    ) -> None:
        self.received: List[Tuple[str, Dict[str, str], Any]] = []
        body = {"id": "evt-1"} if body is None else body
        raw = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                headers = {k.lower(): v for k, v in self.headers.items()}
                service.received.append(
                    (self.path, headers, json.loads(self.rfile.read(length)))
                )
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def __enter__(self) -> "EventService":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def service() -> Iterator[EventService]:
    with EventService() as running:
        yield running


def _clock(tz: Any) -> datetime:
    return datetime(2026, 10, 13, 8, 0, tzinfo=timezone.utc).astimezone(tz)


def _image(height: int = 24, width: int = 32) -> ImageData:
    pixels = torch.zeros((3, height, width), dtype=torch.uint8)
    pixels[0] = 200
    return ImageData.from_tensor(pixels)


def _boxes(**extra: Any) -> Detections:
    detections = Detections(
        xyxy=torch.tensor([[2.0, 4.0, 12.0, 10.0], [0.5, 0.5, 3.0, 2.0]]),
        class_id=torch.tensor([1, 0]),
        confidence=torch.tensor([0.9, 0.25]),
        image_metadata=NAMES,
        **extra,
    )
    return detections


def _run(block: EventWriterSinkBlock, url: str, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = dict.fromkeys(
        (
            "input_image",
            "image_label",
            "qc_result",
            "external_id",
            "location",
            "item_count",
            "item_type",
            "alert_type",
            "severity",
            "alert_description",
            "custom_value",
            "object_detections",
            "classifications",
            "instance_segmentations",
            "keypoint_detections",
        )
    )
    arguments.update(
        event_ingestion_url=url,
        event_schema="quality_check",
        output_image=_image(),
        custom_metadata={},
        fire_and_forget=False,
        disable_sink=False,
        request_timeout=5,
    )
    arguments.update(params)
    result = block.run(**arguments)

    return result


def _decoded_shape(text: str) -> Tuple[int, ...]:
    jpeg = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR).shape


def test_sync_quality_check_posts_v1_payload_and_returns_event_id(
    service: EventService,
) -> None:
    block = EventWriterSinkBlock(wall_clock=_clock)

    result = _run(
        block,
        service.url,
        qc_result="fail",
        external_id="batch-1",
        image_label="station-3",
        input_image=_image(48, 64),
        custom_metadata={"line": "A1", "shift": 2},
    )

    assert result == {
        "error_status": False,
        "event_id": "evt-1",
        "message": "Event created successfully",
    }
    path, headers, payload = service.received[0]
    assert path == "/v2/events"
    assert headers["content-type"] == "application/json"
    assert "x-api-key" not in headers
    image = payload.pop("images")[0]
    assert payload == {
        "inference_timestamp": "2026-10-13T08:00:00+00:00",
        "event_schema": "quality_check",
        "event_data": {"result": "fail", "externalId": "batch-1"},
        "displayImagePosition": 0,
        "custom_metadata": {"line": "A1", "shift": 2},
    }
    assert sorted(image) == ["base64Image", "inputBase64Image", "label"]
    assert image["label"] == "station-3"
    assert _decoded_shape(image["base64Image"]) == (24, 32, 3)
    assert _decoded_shape(image["inputBase64Image"]) == (48, 64, 3)


@pytest.mark.parametrize(
    "schema, params, event_data",
    [
        ("quality_check", {"location": "W-A"}, {}),
        (
            "inventory_count",
            {"location": "W-A", "item_count": 0, "item_type": "box"},
            {"location": "W-A", "itemCount": 0, "itemType": "box"},
        ),
        (
            "safety_alert",
            {"alert_type": "no_hardhat", "severity": "high", "alert_description": "B"},
            {"alertType": "no_hardhat", "severity": "high", "description": "B"},
        ),
        (
            "custom",
            {"custom_value": "x", "external_id": "e", "qc_result": "pass"},
            {"value": "x", "externalId": "e"},
        ),
    ],
)
def test_event_data_keeps_only_set_fields_of_its_schema(
    service: EventService, schema: str, params: Dict[str, Any], event_data: Dict
) -> None:
    # Fields of other schemas (location, qc_result above) are not sent.
    _run(EventWriterSinkBlock(), service.url, event_schema=schema, **params)

    assert service.received[0][2]["event_data"] == event_data


def test_api_key_resource_is_sent_as_header(service: EventService) -> None:
    block = EventWriterSinkBlock(event_ingestion_api_key=Secret(API_KEY, name="EIS"))

    _run(block, service.url)

    assert service.received[0][1]["x-api-key"] == API_KEY


def test_unavailable_api_key_fails_the_step_and_sends_nothing(
    service: EventService,
) -> None:
    block = EventWriterSinkBlock(event_ingestion_api_key=Secret.unavailable("EIS"))

    with pytest.raises(SecretUnavailableError):
        _run(block, service.url)

    assert service.received == []


@pytest.mark.parametrize(
    "status, body, message",
    [
        (
            529,
            {"detail": "queue full"},
            "Event Ingestion Service at capacity (529). The device is experiencing "
            "backpressure. Detail: queue full",
        ),
        (422, {"detail": "bad schema"}, "HTTP 422: bad schema"),
        (500, "plain failure", "HTTP 500: plain failure"),
    ],
)
def test_error_responses_give_v1_messages(status: int, body: Any, message: str) -> None:
    with EventService(status=status, body=body) as failing:
        result = _run(EventWriterSinkBlock(), failing.url)

    assert result == {"error_status": True, "event_id": "", "message": message}


def test_created_without_id_gives_empty_event_id() -> None:
    with EventService(body={}) as created:
        result = _run(EventWriterSinkBlock(), created.url)

    assert result == {
        "error_status": False,
        "event_id": "",
        "message": "Event created successfully",
    }


def test_timeout_is_reported() -> None:
    with EventService(delay=1.5) as slow:
        result = _run(EventWriterSinkBlock(), slow.url, request_timeout=1)

    assert result == {
        "error_status": True,
        "event_id": "",
        "message": "Request timed out after 1s",
    }


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_connection_error_message_is_scrubbed_of_the_api_key() -> None:
    block = EventWriterSinkBlock(event_ingestion_api_key=API_KEY)
    url = f"http://127.0.0.1:{_closed_port()}/{API_KEY}"

    result = _run(block, url)

    assert result["error_status"] is True
    assert result["message"].startswith("Request failed: ")
    assert API_KEY not in result["message"]
    assert "<secret event_ingestion_api_key>" in result["message"]


def test_fire_and_forget_sends_in_background_effects(service: EventService) -> None:
    with BackgroundEffects() as effects:
        block = EventWriterSinkBlock(sink_effects=effects)
        result = _run(block, service.url, fire_and_forget=True)
        assert effects.drain(timeout=5)

    assert result == {
        "error_status": False,
        "event_id": "",
        "message": "Event sent in background task",
    }
    assert len(service.received) == 1
    assert effects.stats().completed == 1


def test_admitted_event_has_no_id_even_when_the_worker_finishes_first(
    service: EventService,
) -> None:
    class WorkerFinishesFirst(BackgroundEffects):
        def deliver(self, effect: Any, *, label: str) -> Tuple[bool, str]:
            outcome = super().deliver(effect, label=label)
            # Force a legal interleaving: the real worker completes before
            # the calling step can inspect its admission result.
            assert self.drain(timeout=10)
            return outcome

    with WorkerFinishesFirst() as effects:
        result = _run(
            EventWriterSinkBlock(sink_effects=effects),
            service.url,
            fire_and_forget=True,
        )

    assert len(service.received) == 1
    assert effects.stats().completed == 1
    assert result == {
        "error_status": False,
        "event_id": "",
        "message": "Event sent in background task",
    }


@pytest.mark.parametrize("when_full", ["reject", "run_inline"])
def test_full_background_runner_preserves_reject_and_inline_event_ids(
    service: EventService, when_full: str
) -> None:
    release = threading.Event()

    def held_effect() -> Tuple[bool, str]:
        assert release.wait(timeout=10)
        return False, "holder completed"

    with BackgroundEffects(max_pending=1, when_full=when_full) as effects:
        try:
            effects.deliver(held_effect, label="holder")
            result = _run(
                EventWriterSinkBlock(sink_effects=effects),
                service.url,
                fire_and_forget=True,
            )
        finally:
            release.set()

    if when_full == "reject":
        assert result == {
            "error_status": True,
            "event_id": "",
            "message": FULL_MESSAGE,
        }
        assert service.received == []
        assert effects.stats().rejected == 1
    else:
        assert result == {
            "error_status": False,
            "event_id": "evt-1",
            "message": "Event created successfully",
        }
        assert len(service.received) == 1
        assert effects.stats().ran_inline == 1


def test_fire_and_forget_without_background_effects_sends_now(
    service: EventService,
) -> None:
    result = _run(EventWriterSinkBlock(), service.url, fire_and_forget=True)

    assert result["event_id"] == "evt-1"


def test_background_failure_is_counted_not_returned() -> None:
    with EventService(status=500, body="down") as failing:
        with BackgroundEffects() as effects:
            block = EventWriterSinkBlock(sink_effects=effects)
            result = _run(block, failing.url, fire_and_forget=True)
            effects.drain(timeout=5)

    assert result["error_status"] is False
    assert [failure.message for failure in effects.failures()] == ["HTTP 500: down"]


@pytest.mark.parametrize(
    "policy, parameter, message",
    [
        (True, False, "Sink was disabled by workflow execution policy"),
        (False, True, "Sink was disabled by parameter `disable_sink`"),
    ],
)
def test_disabled_sink_sends_nothing(
    service: EventService, policy: bool, parameter: bool, message: str
) -> None:
    block = EventWriterSinkBlock(disable_sinks=policy)

    result = _run(block, service.url, disable_sink=parameter)

    assert result == {"error_status": False, "event_id": "", "message": message}
    assert service.received == []


def test_box_entries_use_centers_and_row_or_map_names() -> None:
    boxes = _boxes(bboxes_metadata=[{}, {"class": "kitten"}])

    entries = box_entries(boxes)

    assert entries == [
        {
            "class": "dog",
            "x": 7.0,
            "y": 7.0,
            "width": 10.0,
            "height": 6.0,
            "confidence": pytest.approx(0.9),
        },
        {
            "class": "kitten",
            "x": 1.75,
            "y": 1.25,
            "width": 2.5,
            "height": 1.5,
            "confidence": 0.25,
        },
    ]


def test_box_without_any_name_is_unknown() -> None:
    boxes = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        class_id=torch.tensor([7]),
        confidence=torch.tensor([0.5]),
    )

    assert box_entries(boxes)[0]["class"] == "unknown"


def _instances(rle: bool) -> InstanceDetections:
    masks = torch.zeros((2, 12, 16), dtype=torch.bool)
    masks[0, 2:8, 3:11] = True
    masks[1, 5, 5] = True  # one pixel: a contour of fewer than 3 points
    mask: Any = masks
    if rle:
        mask = InstancesRLEMasks.from_coco_rle_masks(
            (12, 16), [torch_mask_to_coco_rle(m) for m in masks]
        )
    instances = InstanceDetections(
        xyxy=torch.tensor([[3.0, 2.0, 11.0, 8.0], [5.0, 5.0, 6.0, 6.0]]),
        class_id=torch.tensor([0, 1]),
        confidence=torch.tensor([0.5, 0.75]),
        mask=mask,
        image_metadata=NAMES,
    )
    return instances


@pytest.mark.parametrize("rle", [False, True])
def test_segmentation_points_are_the_outer_contour(rle: bool) -> None:
    entries = segmentation_entries(_instances(rle))

    assert len(entries) == 1
    assert entries[0]["class"] == "cat"
    assert entries[0]["points"] == [[3.0, 2.0], [3.0, 7.0], [10.0, 7.0], [10.0, 2.0]]


def test_keypoint_entries_list_every_keypoint_inside_its_box() -> None:
    points = KeyPoints(
        xy=torch.tensor([[[3.0, 4.0], [5.0, 6.0]], [[1.0, 1.0], [2.0, 2.0]]]),
        class_id=torch.tensor([1, 0]),
        confidence=torch.full((2, 2), 0.5),
    )

    entries = keypoint_entries((points, _boxes()))

    assert entries[0]["class"] == "dog"
    assert entries[0]["keypoints"] == [
        {"id": 0, "x": 3.0, "y": 4.0},
        {"id": 1, "x": 5.0, "y": 6.0},
    ]
    assert len(entries) == 2


def test_zero_confidence_keypoint_slots_are_left_out_and_ids_keep_their_slot() -> None:
    # Models zero the slots below their threshold and past a shorter skeleton.
    points = KeyPoints(
        xy=torch.tensor([[[3.0, 4.0], [0.0, 0.0], [7.0, 8.0]], [[0.0, 0.0]] * 3]),
        class_id=torch.tensor([1, 0]),
        confidence=torch.tensor([[0.9, 0.0, 0.6], [0.0, 0.0, 0.0]]),
    )

    entries = keypoint_entries((points, _boxes()))

    assert len(entries) == 1
    assert entries[0]["keypoints"] == [
        {"id": 0, "x": 3.0, "y": 4.0},
        {"id": 2, "x": 7.0, "y": 8.0},
    ]


def test_keypoints_without_boxes_fail() -> None:
    points = KeyPoints(
        xy=torch.zeros((1, 1, 2)),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="need boxes"):
        keypoint_entries((points, None))


def test_annotations_are_attached_and_empty_ones_left_out(
    service: EventService,
) -> None:
    single = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.2, 0.8]]),
        images_metadata=[NAMES],
    )
    empty = Detections(
        xyxy=torch.zeros((0, 4)),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,)),
    )

    _run(
        EventWriterSinkBlock(),
        service.url,
        object_detections=empty,
        classifications=single,
        instance_segmentations=_instances(rle=True),
    )

    image = service.received[0][2]["images"][0]
    assert sorted(image) == ["base64Image", "classifications", "instanceSegmentations"]
    assert image["classifications"] == [
        {"class": "dog", "confidence": pytest.approx(0.8)}
    ]


def test_multi_label_classification_sends_predicted_classes(
    service: EventService,
) -> None:
    multi = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 1]),
        confidence=torch.tensor([0.6, 0.7]),
        image_metadata=NAMES,
    )

    _run(EventWriterSinkBlock(), service.url, classifications=multi)

    classes = service.received[0][2]["images"][0]["classifications"]
    assert [entry["class"] for entry in classes] == ["dog", "cat"]


def test_gray_image_is_encoded(service: EventService) -> None:
    gray = ImageData.from_tensor(torch.full((1, 8, 8), 90, dtype=torch.uint8))

    _run(EventWriterSinkBlock(), service.url, output_image=gray)

    image = service.received[0][2]["images"][0]
    assert _decoded_shape(image["base64Image"]) == (8, 8, 3)
