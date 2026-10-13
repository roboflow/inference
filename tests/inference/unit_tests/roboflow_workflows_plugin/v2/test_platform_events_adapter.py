"""Event adapters against loopback HTTP and a temporary directory.

Every request goes to a 127.0.0.1 fixture server; every key is invented.
These tests check the adapters' wire and archive behavior, not the real
Roboflow API or Event Ingestion Service.
"""

import base64
import errno
import json
import logging
import os
import re
import socket
import stat
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Thread

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.blocks.roboflow_platform.vision_events import (
    RoboflowVisionEventsBlock,
    VisionEventBundleBlock,
    VisionEventRequest,
)
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret

from inference.core import roboflow_api
from inference.roboflow_workflows_plugin.v2 import platform_events_adapter as module
from inference.roboflow_workflows_plugin.v2.credential_boundary import (
    AUDITED_LOGGERS,
    HostOperation,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

KEY = "invented-R4 secret/+sentinel"
FIXED_TIME = datetime(2026, 10, 13, 10, 20, 30, tzinfo=timezone.utc)


def request(**changes):
    values = dict(
        input_image=None,
        output_image=None,
        predictions=None,
        event_type="inventory_count",
        solution="use-case",
        event_data={"itemCount": 0},
        custom_metadata={"camera": "north"},
    )
    values.update(changes)
    return VisionEventRequest(**values)


def image(red):
    """A red (red=True) or blue RGB image."""
    rgb = torch.zeros((3, 8, 10), dtype=torch.uint8)
    rgb[0 if red else 2] = 240
    return ImageData.from_tensor(rgb)


def is_red(jpeg):
    pixels = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return pixels[:, :, 2].mean() > 200 and pixels[:, :, 0].mean() < 50


def multipart_jpeg(body):
    return body[body.index(b"\xff\xd8") : body.rindex(b"\xff\xd9") + 2]


@pytest.fixture
def server():
    """Loopback server: ``routes[path] = (status, JSON value or raw bytes)``."""
    state = {
        "calls": [],
        "delay": 0,
        "routes": {
            "/vision-events/upload": None,
            "/vision-events": (201, {"accepted": True}),
            "/v2/events": (201, {"id": "local-event"}),
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["calls"].append((self.path, dict(self.headers), body))
            time.sleep(state["delay"])
            route = state["routes"][self.path]
            uploads = sum(path == self.path for path, _, _ in state["calls"])
            status, answer = route or (200, {"sourceId": f"image-{uploads}"})
            raw = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{http.server_port}"
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        thread.join(5)


def cloud(server, event_request, **adapter_options):
    adapter = module.RoboflowVisionEventsAdapter(
        api_base_url=server["url"], **adapter_options
    )
    return adapter.submit(
        event_request,
        api_key=Secret(KEY, name="api_key"),
        write_to_event_store=False,
        event_store_url="http://unused.invalid",
    )


def local(url, event_request, **adapter_options):
    adapter = module.RoboflowVisionEventsAdapter(
        api_base_url="http://unused.invalid", **adapter_options
    )
    return adapter.submit(
        event_request, api_key=None, write_to_event_store=True, event_store_url=url
    )


def assert_fixed_failure(caught, text):
    assert str(caught.value) == text
    assert KEY not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_cloud_uploads_display_then_original_and_returns_client_event_id(server):
    detections = Detections(
        xyxy=torch.tensor([[1.0, 2.0, 7.0, 6.0]]),
        class_id=torch.tensor([2]),
        confidence=torch.tensor([1.2]),
        image_metadata={"class_names": {2: "part"}},
    )

    event_id = cloud(
        server,
        request(
            output_image=image(True), input_image=image(False), predictions=detections
        ),
    )

    calls = server["calls"]
    assert [path for path, _, _ in calls] == [
        "/vision-events/upload",
        "/vision-events/upload",
        "/vision-events",
    ]
    assert all(headers["Authorization"] == f"Bearer {KEY}" for _, headers, _ in calls)
    assert all("X-Roboflow-Inference-Version" in headers for _, headers, _ in calls)
    assert is_red(multipart_jpeg(calls[0][2]))
    assert not is_red(multipart_jpeg(calls[1][2]))
    payload = json.loads(calls[2][2])
    assert payload.pop("eventId") == event_id
    assert datetime.fromisoformat(payload.pop("timestamp")).tzinfo is not None
    assert payload == {
        "eventType": "inventory_count",
        "useCaseId": "use-case",
        "eventData": {"itemCount": 0},
        "customMetadata": {"camera": "north"},
        "displayImagePosition": 0,
        "images": [
            {
                "sourceId": "image-1",
                "inputSourceId": "image-2",
                "label": "workflow",
                # V1 clamps confidence into [0, 1].
                "objectDetections": [
                    {
                        "class": "part",
                        "x": 4.0,
                        "y": 4.0,
                        "width": 6.0,
                        "height": 4.0,
                        "confidence": 1.0,
                    }
                ],
            }
        ],
    }


def test_cloud_extra_headers_apply_and_a_configured_authorization_wins(
    server, monkeypatch
):
    monkeypatch.setattr(
        roboflow_api,
        "ROBOFLOW_API_EXTRA_HEADERS",
        json.dumps({"Authorization": "Bearer gateway", "X-Gateway": "edge"}),
    )

    cloud(server, request(input_image=image(True)))

    for _, headers, _ in server["calls"]:
        assert headers["Authorization"] == "Bearer gateway"
        assert headers["X-Gateway"] == "edge"


def test_cloud_without_images_uploads_nothing(server):
    cloud(server, request(event_data={}, custom_metadata={}))

    (path, _, body), *rest = server["calls"]
    payload = json.loads(body)
    assert path == "/vision-events" and not rest
    assert payload["images"] == []
    assert {"eventData", "customMetadata", "displayImagePosition"}.isdisjoint(payload)


@pytest.mark.parametrize(
    "status,category",
    [
        (500, "request unsuccessful"),
        (401, "authorization failed"),
        (403, "access forbidden"),
        (404, "resource not found"),
    ],
)
def test_event_failure_after_upload_is_a_fixed_failure(server, status, category):
    server["routes"]["/vision-events"] = (status, {"detail": KEY})

    with pytest.raises(PlatformRequestError) as caught:
        cloud(server, request(input_image=image(True)))

    # The image stays uploaded: V1 has no rollback either.
    assert [path for path, _, _ in server["calls"]] == [
        "/vision-events/upload",
        "/vision-events",
    ]
    assert_fixed_failure(caught, f"vision event submission failed: {category}")


@pytest.mark.parametrize(
    "answer",
    [
        (200, {}),
        (200, {"sourceId": ""}),
        (200, {"sourceId": 2}),
        (200, []),
        (200, b"not json"),
    ],
)
def test_malformed_upload_response_stops_before_the_event(server, answer):
    server["routes"]["/vision-events/upload"] = answer

    with pytest.raises(PlatformRequestError) as caught:
        cloud(server, request(input_image=image(True)))

    assert len(server["calls"]) == 1
    assert_fixed_failure(
        caught, "vision event submission failed: malformed API response"
    )


def test_rejected_upload_stops_before_the_event(server):
    server["routes"]["/vision-events/upload"] = (401, {"detail": KEY})

    with pytest.raises(PlatformRequestError) as caught:
        cloud(server, request(output_image=image(True), input_image=image(False)))

    assert len(server["calls"]) == 1
    assert_fixed_failure(caught, "vision event submission failed: authorization failed")


def test_connection_refused_and_timeout_are_fixed_categories(server, monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_url = f"http://127.0.0.1:{probe.getsockname()[1]}"
    with pytest.raises(PlatformRequestError) as caught:
        local(closed_url, request())
    assert_fixed_failure(caught, "vision event submission failed: connection failed")

    server["delay"] = 1
    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 0.2)
    with pytest.raises(PlatformRequestError) as caught:
        cloud(server, request())
    assert_fixed_failure(caught, "vision event submission failed: request timed out")


def test_local_store_embeds_both_images_and_forwards_the_use_case(server):
    event_id = local(
        server["url"] + "/",
        request(output_image=image(True), input_image=image(False)),
        event_ingestion_api_key=Secret(KEY, name="event_ingestion_api_key"),
    )

    assert event_id == "local-event"
    ((path, headers, body),) = server["calls"]
    assert path == "/v2/events"
    assert headers["X-API-Key"] == KEY and "Authorization" not in headers
    payload = json.loads(body)
    assert datetime.fromisoformat(payload.pop("inference_timestamp")).tzinfo
    (entry,) = payload.pop("images")
    assert payload == {
        "solution": "use-case",
        "event_schema": "inventory_count",
        "event_data": {"itemCount": 0},
        "custom_metadata": {"camera": "north"},
        "displayImagePosition": 0,
    }
    assert list(entry) == ["base64Image", "inputBase64Image", "label"]
    assert is_red(base64.b64decode(entry["base64Image"]))
    assert not is_red(base64.b64decode(entry["inputBase64Image"]))


def test_local_store_without_key_images_or_metadata(server):
    local(server["url"], request(custom_metadata={}))

    ((_, headers, body),) = server["calls"]
    payload = json.loads(body)
    assert "X-API-Key" not in headers
    assert payload["images"] == []
    assert {"custom_metadata", "displayImagePosition"}.isdisjoint(payload)


@pytest.mark.parametrize(
    "answer,expected", [({}, ""), ({"id": 3}, "3"), ({"id": "evt-1"}, "evt-1")]
)
def test_local_store_created_event_ids_follow_v1(server, answer, expected):
    server["routes"]["/v2/events"] = (201, answer)

    assert local(server["url"], request()) == expected


@pytest.mark.parametrize(
    "answer,category",
    [
        ((529, {"detail": KEY}), "request unsuccessful"),
        ((200, {"id": "not-created"}), "request unsuccessful"),
        ((403, {"detail": KEY}), "access forbidden"),
        ((201, []), "malformed API response"),
        ((201, b"not json"), "malformed API response"),
    ],
)
def test_local_store_failures_are_fixed(server, answer, category):
    server["routes"]["/v2/events"] = answer

    with pytest.raises(PlatformRequestError) as caught:
        local(server["url"], request())

    assert_fixed_failure(caught, f"vision event submission failed: {category}")


def run_event_block(server, effects, *, fire_and_forget):
    block = RoboflowVisionEventsBlock(
        vision_events_port=module.RoboflowVisionEventsAdapter(
            api_base_url=server["url"]
        ),
        api_key=KEY,
        sink_effects=effects,
    )
    arguments = block.Params(
        event_type="quality_check", solution="use-case", qc_result="pass"
    ).model_dump()
    arguments.update(
        input_image=image(True), fire_and_forget=fire_and_forget, cooldown_seconds=0
    )
    return block.run(**arguments)


def test_block_synchronous_and_scheduled_outcomes_with_the_real_adapter(server):
    with BackgroundEffects() as effects:
        now = run_event_block(server, effects, fire_and_forget=False)
        later = run_event_block(server, effects, fire_and_forget=True)
        assert later["event_id"] == "" and later["error_status"] is False

    events = [
        json.loads(body)
        for path, _, body in server["calls"]
        if path == "/vision-events"
    ]
    assert now["event_id"] == events[0]["eventId"]
    assert now["message"] == "Vision event sent successfully"
    assert len(events) == 2 and events[1]["eventData"] == {"result": "pass"}
    assert effects.stats().completed == 1


def test_block_background_failure_after_upload_is_fixed_and_counted(server):
    server["routes"]["/vision-events"] = (500, {"detail": KEY})
    with BackgroundEffects() as effects:
        result = run_event_block(server, effects, fire_and_forget=True)

    assert result["error_status"] is False
    assert [failure.message for failure in effects.failures()] == [
        "PlatformRequestError: vision event submission failed: request unsuccessful"
    ]
    assert len(server["calls"]) == 2


def bundles(**options):
    return module.VisionEventBundleAdapter(allow_access_to_file_system=True, **options)


def write(adapter, directory, file_name=None, **changes):
    adapter.prepare_destination(target_directory=str(directory), file_name=file_name)
    return adapter.create_bundle(
        request(**changes), target_directory=str(directory), file_name=file_name
    )


def inspect(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        payload = json.load(archive.extractfile("payload.json"))
        files = {m.name: archive.extractfile(m).read() for m in members[1:]}
    return payload, [m.name for m in members], {m.mtime for m in members}, files


def test_bundle_contract_with_deterministic_ids_and_time(tmp_path, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_TIME

    ids = iter(["event-fixed", "output-fixed", "input-fixed"])
    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module, "uuid4", lambda: next(ids))
    adapter = bundles(allowed_write_directory=str(tmp_path))

    bundle = write(
        adapter, tmp_path, "fixture", output_image=image(True), input_image=image(False)
    )

    assert bundle.event_id == "event-fixed"
    assert bundle.bundle_path == str(tmp_path / "fixture.tar.gz")
    payload, names, mtimes, files = inspect(bundle.bundle_path)
    assert names == [
        "payload.json",
        "images/output-fixed.jpg",
        "images/input-fixed.jpg",
    ]
    assert payload == {
        "bundleFormatVersion": 1,
        "eventId": "event-fixed",
        "eventType": "inventory_count",
        "timestamp": "2026-10-13T10:20:30+00:00",
        "images": [
            {
                "file": "images/output-fixed.jpg",
                "inputFile": "images/input-fixed.jpg",
                "label": "workflow",
            }
        ],
        "useCaseId": "use-case",
        "eventData": {"itemCount": 0},
        "customMetadata": {"camera": "north"},
        "displayImagePosition": 0,
    }
    assert mtimes == {int(FIXED_TIME.timestamp())}
    assert is_red(files["images/output-fixed.jpg"])
    assert not is_red(files["images/input-fixed.jpg"])
    assert [path.name for path in tmp_path.iterdir()] == ["fixture.tar.gz"]


def test_bundle_generated_name_without_use_case_or_images(tmp_path):
    bundle = write(bundles(), tmp_path / "new", solution=None)

    name = Path(bundle.bundle_path).name
    assert re.fullmatch(
        rf"event_\d{{8}}T\d{{6}}_\d{{6}}_{bundle.event_id}\.tar\.gz", name
    )
    payload, names, _, _ = inspect(bundle.bundle_path)
    assert names == ["payload.json"] and payload["images"] == []
    # Air-gapped bundles leave the use case to the uploader.
    assert {"useCaseId", "displayImagePosition"}.isdisjoint(payload)


def test_destination_refused_without_filesystem_permission(tmp_path):
    adapter = module.VisionEventBundleAdapter()

    with pytest.raises(RuntimeError, match="forbidden"):
        adapter.prepare_destination(
            target_directory=str(tmp_path / "x"), file_name=None
        )
    with pytest.raises(RuntimeError, match="forbidden"):
        adapter.create_bundle(
            request(), target_directory=str(tmp_path / "x"), file_name=None
        )

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name", ["", "../escape", "/absolute", "a/b", ".hidden", "-dash", "x" * 201]
)
def test_unsafe_file_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        bundles().prepare_destination(
            target_directory=str(tmp_path / "out"), file_name=name
        )

    assert not list(tmp_path.iterdir())


def test_directories_outside_the_allowed_root_are_refused(tmp_path):
    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "link").symlink_to(outside, target_is_directory=True)
    adapter = bundles(allowed_write_directory=str(allowed))

    for directory in [outside, allowed / "link", allowed / ".." / "outside" / "new"]:
        with pytest.raises(ValueError, match="allowed write directory"):
            adapter.prepare_destination(target_directory=str(directory), file_name="b")

    assert not list(outside.iterdir())
    write(adapter, allowed / "nested" / "dir", "inside")
    assert (allowed / "nested" / "dir" / "inside.tar.gz").exists()


def test_unwritable_directory_is_refused(tmp_path):
    directory = tmp_path / "read-only"
    directory.mkdir()
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(ValueError, match="not writable"):
            bundles().prepare_destination(
                target_directory=str(directory), file_name=None
            )
    finally:
        directory.chmod(stat.S_IRWXU)


def test_taken_names_are_refused_and_never_replaced(tmp_path):
    adapter = bundles()
    first = write(adapter, tmp_path, "taken.tar.gz")
    original = Path(first.bundle_path).read_bytes()
    (tmp_path / "dangling.tar.gz").symlink_to(tmp_path / "missing")

    for name in ["taken", "taken.tar.gz", "dangling"]:
        with pytest.raises(ValueError, match="already exists"):
            adapter.prepare_destination(target_directory=str(tmp_path), file_name=name)
    # A name taken after preparation, e.g. by a parallel run, is also refused.
    with pytest.raises(ValueError, match="already exists"):
        adapter.create_bundle(
            request(), target_directory=str(tmp_path), file_name="taken"
        )

    assert Path(first.bundle_path).read_bytes() == original


def test_concurrent_writers_of_one_name_have_exactly_one_winner(tmp_path, monkeypatch):
    barrier, real_link = Barrier(2), os.link

    def synchronized_link(source, target):
        barrier.wait(timeout=5)
        real_link(source, target)

    monkeypatch.setattr(os, "link", synchronized_link)
    adapter = bundles()

    def publish(_):
        try:
            return adapter.create_bundle(
                request(), target_directory=str(tmp_path), file_name="race"
            ).event_id
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [event_id for event_id in pool.map(publish, range(2)) if event_id]

    assert len(winners) == 1
    assert inspect(tmp_path / "race.tar.gz")[0]["eventId"] == winners[0]
    assert [path.name for path in tmp_path.iterdir()] == ["race.tar.gz"]


def test_filesystem_without_hard_links_uses_v1_checked_rename(tmp_path, monkeypatch):
    def unsupported(source, target):
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported)

    bundle = write(bundles(), tmp_path, "fat")

    assert inspect(bundle.bundle_path)[0]["eventId"] == bundle.event_id
    assert [path.name for path in tmp_path.iterdir()] == ["fat.tar.gz"]


@pytest.mark.parametrize("failing", ["link", "fsync"])
def test_write_failures_leave_no_files(tmp_path, monkeypatch, failing):
    def fail(*args):
        raise OSError(errno.EIO, "invented disk failure")

    monkeypatch.setattr(os, failing, fail)

    with pytest.raises(OSError, match="invented disk failure"):
        write(bundles(), tmp_path, "broken")

    assert not list(tmp_path.iterdir())


def test_bundles_over_the_consumer_limit_are_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(module.v1_bundles, "MAX_BUNDLE_SIZE_BYTES", 10)

    with pytest.raises(ValueError, match=r"exceeds the companion API limit"):
        write(bundles(), tmp_path, "large")

    assert not list(tmp_path.iterdir())


def test_archive_build_failure_is_fixed(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError(KEY)

    monkeypatch.setattr(module, "encode_image_to_jpeg_bytes", fail)

    with pytest.raises(PlatformRequestError) as caught:
        write(bundles(), tmp_path, "broken", input_image=image(True))

    assert_fixed_failure(
        caught, "vision event bundle creation failed: invalid host data"
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("count", [0, 1001])
def test_bundle_annotation_lists_are_capped_like_v1(tmp_path, count):
    detections = Detections(
        xyxy=torch.zeros((count, 4)),
        class_id=torch.zeros(count, dtype=torch.int64),
        confidence=torch.ones(count),
    )

    bundle = write(
        bundles(),
        tmp_path,
        f"count-{count}",
        input_image=image(True),
        predictions=detections,
    )

    (entry,) = inspect(bundle.bundle_path)[0]["images"]
    assert len(entry.get("objectDetections", [])) == min(count, 1000)


@pytest.mark.parametrize(
    "prediction,expected",
    [
        (
            ClassificationPrediction(
                class_id=torch.tensor([1]),
                confidence=torch.tensor([[0.2, 0.8]]),
                images_metadata=[{"class_names": {1: "pass"}}],
            ),
            [{"class": "pass", "confidence": pytest.approx(0.8)}],
        ),
        (
            MultiLabelClassificationPrediction(
                class_ids=torch.tensor([0, 1]),
                confidence=torch.tensor([0.7, 0.9]),
                image_metadata={"class_names": {0: "A", 1: "B"}},
            ),
            [
                {"class": "A", "confidence": pytest.approx(0.7)},
                {"class": "B", "confidence": pytest.approx(0.9)},
            ],
        ),
    ],
)
def test_classifications_use_each_class_confidence(tmp_path, prediction, expected):
    bundle = write(
        bundles(),
        tmp_path,
        "classes",
        input_image=image(True),
        predictions=prediction,
    )

    (entry,) = inspect(bundle.bundle_path)[0]["images"]
    assert entry["classifications"] == expected


@pytest.mark.parametrize("rle", [False, True])
def test_segmentation_polygon_fallback_and_keypoint_padding(tmp_path, rle):
    masks = torch.zeros((1, 12, 12), dtype=torch.bool)
    masks[0, 2:8, 3:9] = True
    masks[0, 10, 10] = True
    if rle:
        from pycocotools import mask as mask_utils

        from inference_models.models.base.types import InstancesRLEMasks

        masks = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(12, 12),
            masks=[
                mask_utils.encode(np.asfortranarray(masks[0].numpy().astype(np.uint8)))
            ],
        )
    segmentation = InstanceDetections(
        xyxy=torch.tensor([[3.0, 2.0, 9.0, 8.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.8]),
        mask=masks,
        image_metadata={"class_names": {0: "part"}},
    )
    adapter = bundles()

    polygon_bundle = write(
        adapter, tmp_path, "polygon", input_image=image(True), predictions=segmentation
    )
    # The largest contour wins; the stray pixel at (10, 10) is dropped.
    (entry,) = inspect(polygon_bundle.bundle_path)[0]["images"]
    points = entry["instanceSegmentations"][0]["points"]
    assert len(points) >= 3 and max(x for x, _ in points) < 10

    segmentation.bboxes_metadata = [{"polygon": [[1, 1], [2, 2]]}]
    fallback = write(
        adapter, tmp_path, "fallback", input_image=image(True), predictions=segmentation
    )
    (entry,) = inspect(fallback.bundle_path)[0]["images"]
    assert "instanceSegmentations" not in entry and len(entry["objectDetections"]) == 1

    boxes = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.6]),
        bboxes_metadata=[
            {
                "keypoints_xy": [[1, 2], [3, 4], [0, 0]],
                "keypoints_class_id": [7, 9, 0],
                "keypoints_class_name": ["one", "two", ""],
            }
        ],
    )
    keypoints = write(
        adapter,
        tmp_path,
        "keypoints",
        input_image=image(True),
        predictions=(None, boxes),
    )
    (entry,) = inspect(keypoints.bundle_path)[0]["images"]
    assert entry["keypoints"][0]["keypoints"] == [
        {"id": 7, "x": 1.0, "y": 2.0},
        {"id": 9, "x": 3.0, "y": 4.0},
    ]


def test_bundle_block_with_the_real_adapter_fails_fast_on_a_taken_name(tmp_path):
    (tmp_path / "taken.tar.gz").write_bytes(b"existing")
    arguments = VisionEventBundleBlock.Params(
        event_type="custom", target_directory=str(tmp_path), file_name="taken"
    ).model_dump()

    with BackgroundEffects() as effects:
        block = VisionEventBundleBlock(
            vision_event_bundle_port=bundles(allowed_write_directory=str(tmp_path)),
            sink_effects=effects,
        )
        with pytest.raises(ValueError, match="already exists"):
            block.run(**arguments)

    assert effects.stats().submitted == 0
    assert (tmp_path / "taken.tar.gz").read_bytes() == b"existing"


@pytest.mark.parametrize(
    "operation", [HostOperation.VISION_EVENT, HostOperation.EVENT_BUNDLE]
)
def test_fixed_diagnostics_on_worker(tmp_path, operation, caplog, monkeypatch):
    """Negative probe target: without call_host this test must fail."""

    def emit_and_fail(**kwargs):
        for name in AUDITED_LOGGERS:
            logging.getLogger(name).warning(
                "credential=%s", KEY, extra={"credential": KEY}
            )
        raise ValueError(KEY)

    if operation is HostOperation.VISION_EVENT:
        adapter = module.RoboflowVisionEventsAdapter()
        monkeypatch.setattr(adapter, "_submit", emit_and_fail)

        def effect():
            adapter.submit(
                request(),
                api_key=Secret(KEY, name="api_key"),
                write_to_event_store=False,
                event_store_url="",
            )
            return False, "unexpected success"

    else:
        adapter = bundles()
        monkeypatch.setattr(adapter, "_archive", emit_and_fail)

        def effect():
            write(adapter, tmp_path, "diagnostics")
            return False, "unexpected success"

    with caplog.at_level(logging.WARNING):
        with BackgroundEffects() as effects:
            effects.deliver(effect, label="credential-test")

    (failure,) = effects.failures()
    assert failure.message == (
        f"PlatformRequestError: {operation.value} failed: invalid host data"
    )
    assert KEY not in caplog.text
    audited = [record for record in caplog.records if record.name in AUDITED_LOGGERS]
    assert len(audited) == len(AUDITED_LOGGERS)
    assert all(record.v2_operation == operation.value for record in audited)
    assert all("credential" not in record.__dict__ for record in caplog.records)
    # Outside an operation the filters change nothing.
    logging.getLogger("urllib3.connectionpool").warning("outside-context-sentinel")
    assert "outside-context-sentinel" in caplog.text
