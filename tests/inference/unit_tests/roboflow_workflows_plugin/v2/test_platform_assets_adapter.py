"""Real pinned credit accounting and HTTP transport, restricted to loopback."""

import copy
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Lock, Thread
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.roboflow_platform.asset_library import (
    AssetLibraryAttributesBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.dataset_upload import (
    DatasetUploadBlock,
    DatasetUploadRequest,
)
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.compilation import compile_workflow

from inference.core import roboflow_api
from inference.core.active_learning import cache_operations
from inference.core.active_learning.entities import StrategyLimit, StrategyLimitType
from inference.core.cache.base import BaseCache
from inference.core.utils import url_utils
from inference.roboflow_workflows_plugin.v2.credential_boundary import HostOperation
from inference.roboflow_workflows_plugin.v2.platform_assets_adapter import (
    RoboflowAssetsAdapter,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

KEY = "invented-R2 key/+sentinel"


class SharedCache(BaseCache):
    """Test storage with real BaseCache locks; not distributed-service evidence."""

    def __init__(self):
        self.data = {}
        self.locks = {}
        self.mutex = Lock()

    def get(self, key):
        with self.mutex:
            return copy.deepcopy(self.data.get(key))

    def set(self, key, value, expire=None):
        with self.mutex:
            self.data[key] = copy.deepcopy(value)

    def acquire_lock(self, key, expire=None):
        with self.mutex:
            lock = self.locks.setdefault(key, Lock())
        assert lock.acquire(timeout=5)
        return lock


class Loopback:
    def __init__(self):
        self.requests = []
        self.duplicate = False
        self.status = 200
        self.batch_response = {"taskId": "task"}
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                platform.answer(self)

            def do_POST(self):
                platform.answer(self)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def answer(self, handler):
        url = urlsplit(handler.path)
        body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
        self.requests.append(
            {
                "path": url.path,
                "query": parse_qs(url.query),
                "body": body,
                "headers": dict(handler.headers),
            }
        )
        status = self.status
        if url.path == "/":
            data = {"workspace": "workspace"}
            status = 200
        elif url.path.endswith("/upload"):
            data = {"success": True, "duplicate": self.duplicate, "id": "registered"}
        elif url.path == "/workspace/images/metadata":
            data = self.batch_response
        else:
            data = {"success": True}
        payload = json.dumps(data).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture
def wire(monkeypatch):
    loopback = Loopback()
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", loopback.url)
    monkeypatch.setattr(roboflow_api, "API_PROXY_BASE_URL", loopback.url + "/proxy")
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", None)
    roboflow_api.get_roboflow_workspace.cache_clear()
    yield loopback
    roboflow_api.get_roboflow_workspace.cache_clear()
    loopback.close()


def request(**changes):
    rgb = torch.zeros((3, 7, 11), dtype=torch.uint8)
    rgb[0] = 230
    value = DatasetUploadRequest(
        image=ImageData.from_tensor(rgb, image_id="fixture"),
        prediction=None,
        target_project="project",
        usage_quota_name="shared",
        minutely_usage_limit=1,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(6, 4),
        compression_level=95,
        registration_tags=["first", "second"],
        labeling_batch_prefix="collection",
        labeling_batches_recreation_frequency="never",
        image_name="fixture",
        metadata={"camera": 7},
    )
    return replace(value, **changes)


def multipart(record):
    raw = (
        b"Content-Type: "
        + record["headers"]["Content-Type"].encode()
        + b"\r\n\r\n"
        + record["body"]
    )
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        )
        for part in message.iter_parts()
    }


def track_refunds(monkeypatch):
    refunds = []
    original = cache_operations.return_strategy_credit

    def returning(**arguments):
        refunds.append(arguments)
        return original(**arguments)

    monkeypatch.setattr(
        "inference.roboflow_workflows_plugin.v2.platform_upload_quota.return_strategy_credit",
        returning,
    )
    return refunds


def test_explicit_cache_and_operation_labels():
    with pytest.raises(ValueError, match="explicitly supplied"):
        RoboflowAssetsAdapter(cache=None)
    assert HostOperation.ASSET_ATTRIBUTES.value == "asset attribute update"
    assert HostOperation.DATASET_UPLOAD.value == "dataset upload"


def test_real_upload_wire_rgb_jpeg_tags_metadata_and_success_retains_credit(
    wire, monkeypatch
):
    refunds = track_refunds(monkeypatch)
    cache = SharedCache()
    first, second = RoboflowAssetsAdapter(cache=cache), RoboflowAssetsAdapter(
        cache=cache
    )
    assert (
        first.upload(request(), api_key=Secret(KEY, name="RF_KEY"))[1]
        == "Successfully registered image"
    )
    before = len(wire.requests)
    assert (
        second.upload(request(), api_key=Secret(KEY, name="RF_KEY"))[1]
        == "Registration skipped due to usage quota exceeded"
    )
    assert len(wire.requests) == before
    assert not refunds
    record = wire.requests[-1]
    fields = multipart(record)
    assert fields["name"] == b"fixture.jpg"
    assert json.loads(fields["metadata"]) == {"camera": 7}
    decoded = cv2.imdecode(np.frombuffer(fields["file"], np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (3, 6)
    assert decoded[1, 1, 2] > 220 and decoded[1, 1, 0] < 10
    assert record["query"]["tag"] == ["first", "second"]
    assert record["query"]["batch"] == ["collection"]
    assert record["query"]["api_key"] == [KEY]


@pytest.mark.parametrize(
    "failure", ["duplicate", "http", "serialization", "annotation"]
)
def test_duplicate_and_failures_refund_once_with_original_scope(
    wire, monkeypatch, failure
):
    refunds = track_refunds(monkeypatch)
    cache = SharedCache()
    adapter = RoboflowAssetsAdapter(cache=cache)
    upload = request()
    if failure == "duplicate":
        wire.duplicate = True
    elif failure == "http":
        wire.status = 400
    elif failure == "serialization":
        upload = request(metadata={"bad": object()})
    else:
        prediction = ClassificationPrediction(
            class_id=torch.tensor([0]),
            confidence=torch.tensor([[0.9, 0.1]]),
            images_metadata=[{"class_names": {0: "red", 1: "blue"}}],
        )
        upload = request(prediction=prediction)
        original = roboflow_api.annotate_image_at_roboflow

        def broken(**arguments):
            raise ValueError(KEY)

        monkeypatch.setattr(roboflow_api, "annotate_image_at_roboflow", broken)
    error, message = adapter.upload(upload, api_key=Secret(KEY, name="RF_KEY"))
    assert error is (failure != "duplicate")
    assert KEY not in message
    assert len(refunds) == 1
    assert {k: v for k, v in refunds[0].items() if k != "cache"} == {
        "workspace": "workspace",
        "project": "project",
        "strategy_name": "shared",
    }
    wire.duplicate, wire.status = False, 200
    if failure == "annotation":
        monkeypatch.setattr(roboflow_api, "annotate_image_at_roboflow", original)
    assert adapter.upload(request(), api_key=Secret(KEY, name="RF_KEY")) == (
        False,
        "Successfully registered image",
    )
    assert len(refunds) == 1


def test_zero_limit_never_dispatches_or_refunds(wire, monkeypatch):
    refunds = track_refunds(monkeypatch)
    adapter = RoboflowAssetsAdapter(cache=SharedCache())
    assert (
        "quota exceeded"
        in adapter.upload(
            request(minutely_usage_limit=0), api_key=Secret(KEY, name="RF_KEY")
        )[1]
    )
    assert [record["path"] for record in wire.requests] == ["/"]
    assert not refunds


def test_classification_and_detection_annotation_wire(wire):
    adapter = RoboflowAssetsAdapter(cache=SharedCache())
    prediction = ClassificationPrediction(
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9, 0.1]]),
        images_metadata=[
            {"inference_id": "class-id", "class_names": {0: "red", 1: "blue"}}
        ],
    )
    assert (
        adapter.upload(
            request(prediction=prediction), api_key=Secret(KEY, name="RF_KEY")
        )[1]
        == "Successfully registered image and annotation"
    )
    assert wire.requests[-1]["body"] == b"red"
    assert wire.requests[-1]["query"]["name"] == ["fixture.txt"]
    assert wire.requests[-2]["query"]["inference_id"] == ["class-id"]
    detection = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 11.0, 7.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.8]),
        image_metadata={
            "image_dimensions": np.array([7, 11]),
            "inference_id": "det-id",
            "class_names": {0: "red"},
        },
        bboxes_metadata=[{"class_name": "red", "detection_id": "box-1"}],
    )
    assert (
        adapter.upload(
            request(prediction=detection, usage_quota_name="detection"),
            api_key=Secret(KEY, name="RF_KEY"),
        )[0]
        is False
    )
    content = json.loads(wire.requests[-1]["body"])
    assert (
        content["predictions"][0]["width"] == 6
        and content["predictions"][0]["height"] == 3
    )
    assert detection.xyxy.tolist() == [[0.0, 0.0, 11.0, 7.0]]


def test_asset_single_batch_wire_and_malformed_response(wire):
    adapter = RoboflowAssetsAdapter(cache=SharedCache())
    block = AssetLibraryAttributesBlock(asset_library_port=adapter, api_key=KEY)
    assert not block.run(
        source_id="one/slash", metadata={"x": 1}, tags=["tag"], disable_sink=False
    )["error_status"]
    assert wire.requests[-1]["path"] == "/workspace/images/one%2Fslash/metadata"
    assert json.loads(wire.requests[-1]["body"]) == {
        "metadata": {"x": 1},
        "addTags": ["tag"],
    }
    assert not block.run(
        source_id=Batch.of(["a", "b"]), metadata={"x": 1}, tags=None, disable_sink=False
    )[0]["error_status"]
    assert json.loads(wire.requests[-1]["body"]) == {
        "updates": [
            {"imageId": "a", "metadata": {"x": 1}},
            {"imageId": "b", "metadata": {"x": 1}},
        ]
    }
    wire.batch_response = {}
    results = block.run(
        source_id=Batch.of(["a", "b"]), metadata={"x": 1}, tags=None, disable_sink=False
    )
    assert all(
        row["error_status"] and "invalid host data" in row["message"] for row in results
    )


@pytest.mark.parametrize("kind", ["empty", "multilabel", "dense", "keypoints", "rle"])
def test_native_prediction_branches(wire, kind):
    from pycocotools import mask as mask_utils

    from inference.roboflow_workflows_plugin.sinks.dataset_upload import (
        v1_tensor as native,
    )
    from inference_models.models.base.types import InstancesRLEMasks

    image_metadata = {
        "image_dimensions": np.array([7, 11]),
        "class_names": {0: "red", 1: "blue"},
    }
    boxes = dict(
        xyxy=torch.tensor([[0.0, 0.0, 11.0, 7.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.8]),
        image_metadata=image_metadata,
        bboxes_metadata=[{"detection_id": "box"}],
    )
    if kind == "empty":
        prediction = Detections(
            xyxy=torch.empty((0, 4)),
            class_id=torch.empty(0, dtype=torch.int64),
            confidence=torch.empty(0),
        )
    elif kind == "multilabel":
        prediction = MultiLabelClassificationPrediction(
            class_ids=torch.tensor([1, 0]),
            confidence=torch.tensor([0.8, 0.9]),
            image_metadata=image_metadata,
        )
    elif kind in ("dense", "rle"):
        mask = torch.ones((1, 7, 11), dtype=torch.bool)
        if kind == "rle":
            mask = InstancesRLEMasks.from_coco_rle_masks(
                image_size=(7, 11),
                masks=[
                    mask_utils.encode(
                        np.asfortranarray(mask[0].numpy().astype(np.uint8))
                    )
                ],
            )
        prediction = InstanceDetections(**boxes, mask=mask)
    else:
        points = KeyPoints(
            xy=torch.tensor([[[2.0, 2.0], [8.0, 5.0]]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([[0.9, 0.8]]),
            image_metadata=image_metadata,
        )
        prediction = (points, Detections(**boxes))
    adapter = RoboflowAssetsAdapter(cache=SharedCache())
    error, message = adapter.upload(
        request(prediction=prediction), api_key=Secret(KEY, name="RF_KEY")
    )
    assert error is False, message
    if kind == "empty":
        assert message == "Successfully registered image"
        assert wire.requests[-1]["path"].endswith("/upload")
    elif kind == "multilabel":
        assert wire.requests[-1]["body"] == b"blue,red"
    else:
        assert message == "Successfully registered image and annotation"
        content = json.loads(wire.requests[-1]["body"])
        assert content["predictions"][0]["width"] == 6
        if kind == "dense":
            assert content["predictions"][0]["points"]
        if kind == "rle":
            scaled = native.scale_tensor_native_prediction(
                prediction=prediction,
                scale=(6 / 11, 3 / 7),
                target_size_wh=(6, 3),
                update_scaling_metadata=False,
            )
            assert scaled.mask is prediction.mask


@pytest.mark.parametrize(
    "frequency,expected",
    [
        ("never", "collection"),
        ("daily", "collection_2026_10_13"),
        ("weekly", "collection_2026_10_12"),
        ("monthly", "collection_2026_10_01"),
    ],
)
def test_labeling_batch_calendar_parity(wire, monkeypatch, frequency, expected):
    from datetime import datetime

    from inference.roboflow_workflows_plugin.sinks.dataset_upload import (
        v1_tensor as native,
    )

    class Clock(datetime):
        @classmethod
        def today(cls):
            return cls(2026, 10, 13, 12)

    monkeypatch.setattr(native, "datetime", Clock)
    adapter = RoboflowAssetsAdapter(cache=SharedCache())
    assert (
        adapter.upload(
            request(labeling_batches_recreation_frequency=frequency),
            api_key=Secret(KEY, name="RF_KEY"),
        )[0]
        is False
    )
    assert wire.requests[-1]["query"]["batch"] == [expected]


def test_mixed_v1_v2_compete_and_scopes_are_separate(wire):
    cache = SharedCache()
    adapter = RoboflowAssetsAdapter(cache=cache)
    strategies = OrderedDict(
        shared=[
            StrategyLimit(StrategyLimitType.MINUTELY, 1),
            StrategyLimit(StrategyLimitType.HOURLY, 100),
            StrategyLimit(StrategyLimitType.DAILY, 1000),
        ]
    )
    assert (
        cache_operations.use_credit_of_matching_strategy(
            cache=cache,
            workspace="workspace",
            project="project",
            matching_strategies_limits=strategies,
        )
        == "shared"
    )
    assert (
        "quota exceeded"
        in adapter.upload(request(), api_key=Secret(KEY, name="RF_KEY"))[1]
    )
    assert (
        adapter.upload(
            request(target_project="other"), api_key=Secret(KEY, name="RF_KEY")
        )[1]
        == "Successfully registered image"
    )
    assert (
        cache_operations.use_credit_of_matching_strategy(
            cache=cache,
            workspace="workspace",
            project="other",
            matching_strategies_limits=strategies,
        )
        is None
    )
    assert (
        cache_operations.use_credit_of_matching_strategy(
            cache=cache,
            workspace="different",
            project="other",
            matching_strategies_limits=strategies,
        )
        == "shared"
    )


def test_two_v2_sessions_and_full_v1_upload_share_one_credit(wire):
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    from inference.roboflow_workflows_plugin.sinks.dataset_upload import (
        v1_tensor as native,
    )

    cache = SharedCache()
    catalogue = create_catalogue().with_blocks([DatasetUploadBlock])
    plan = compile_workflow(
        {
            "version": "2.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [
                {
                    "type": "v2/roboflow_dataset_upload@v2",
                    "name": "collect",
                    "images": "$inputs.image",
                    "target_project": "project",
                    "usage_quota_name": "shared",
                    "minutely_usage_limit": 1,
                    "fire_and_forget": False,
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "message",
                    "selector": "$steps.collect.message",
                }
            ],
        },
        catalogue=catalogue,
    )
    sessions = [
        plan.create_session(
            resources={
                "api_key": KEY,
                "dataset_upload_port": RoboflowAssetsAdapter(cache=cache),
            }
        )
        for _ in range(2)
    ]
    start = Barrier(3)

    def v2(session):
        start.wait(5)
        return session.run({"image": [request().image]}).rows()[0]["message"]

    def v1():
        start.wait(5)
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="legacy"),
            numpy_image=np.zeros((7, 11, 3), np.uint8),
        )
        return native.execute_registration(
            image=image,
            prediction=None,
            target_project="project",
            persist_predictions=True,
            usage_quota_name="shared",
            minutely_usage_limit=1,
            hourly_usage_limit=100,
            daily_usage_limit=1000,
            max_image_size=(6, 4),
            compression_level=95,
            registration_tags=[],
            labeling_batch_prefix="legacy",
            new_labeling_batch_frequency="never",
            cache=cache,
            api_key=KEY,
        )[1]

    try:
        with ThreadPoolExecutor(max_workers=3) as workers:
            futures = [workers.submit(v2, session) for session in sessions] + [
                workers.submit(v1)
            ]
            messages = [future.result(timeout=10) for future in futures]
    finally:
        for session in sessions:
            session.close()
    assert messages.count("Successfully registered image") == 1
    assert messages.count("Registration skipped due to usage quota exceeded") == 2
    assert (
        len([record for record in wire.requests if record["path"].endswith("/upload")])
        == 1
    )


def test_boundary_runs_in_worker_and_scrubs_host_diagnostics(wire, monkeypatch, caplog):
    adapter = RoboflowAssetsAdapter(cache=SharedCache())

    def broken(**arguments):
        logging.getLogger("urllib3.connectionpool").warning(
            "secret %s", KEY, extra={"payload": KEY}
        )
        raise ValueError(KEY)

    monkeypatch.setattr(roboflow_api, "register_image_at_roboflow", broken)
    upload = request()
    with caplog.at_level(logging.DEBUG), BackgroundEffects() as effects:
        effects.deliver(
            lambda: adapter.upload(upload, api_key=Secret(KEY, name="RF_KEY")),
            label="dataset",
        )
    assert effects.stats().failed == 1
    assert KEY not in caplog.text + repr(effects.failures())
    records = [r for r in caplog.records if r.name == "urllib3.connectionpool"]
    assert records and records[-1].msg == "V2 host diagnostic: dataset upload"
    assert "payload" not in records[-1].__dict__
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("urllib3.connectionpool").warning("outside %s", "V1-visible")
    assert "V1-visible" in caplog.text


def test_asset_host_exception_does_not_retain_original_chain(wire, monkeypatch):
    adapter = RoboflowAssetsAdapter(cache=SharedCache())

    def broken(**arguments):
        raise ValueError(KEY)

    monkeypatch.setattr(roboflow_api, "update_image_metadata_at_roboflow", broken)
    with pytest.raises(PlatformRequestError) as error:
        adapter.update_attributes(
            api_key=Secret(KEY, name="RF_KEY"),
            workspace_id="workspace",
            updates=[{"imageId": "a"}],
        )
    assert str(error.value) == "asset attribute update failed: invalid host data"
    assert error.value.__cause__ is None and error.value.__context__ is None


def test_missing_image_id_refunds_and_failed_scheduling_never_reserves(
    wire, monkeypatch
):
    refunds = track_refunds(monkeypatch)
    adapter = RoboflowAssetsAdapter(cache=SharedCache())
    original = roboflow_api.register_image_at_roboflow
    monkeypatch.setattr(
        roboflow_api,
        "register_image_at_roboflow",
        lambda **arguments: {"success": True},
    )
    error, message = adapter.upload(request(), api_key=Secret(KEY, name="RF_KEY"))
    assert error and KEY not in message
    assert len(refunds) == 1
    monkeypatch.setattr(roboflow_api, "register_image_at_roboflow", original)
    effects = BackgroundEffects()
    effects.close()
    args = dict(vars(request()))
    args["images"] = Batch.of([args.pop("image")])
    args["predictions"] = args.pop("prediction")
    args.update(
        data_percentage=100,
        persist_predictions=True,
        disable_sink=False,
        fire_and_forget=True,
    )
    before = copy.deepcopy(adapter._cache.data)
    with pytest.raises(RuntimeError, match="closed"):
        DatasetUploadBlock(
            dataset_upload_port=adapter, api_key=KEY, sink_effects=effects
        ).run(**args)
    assert adapter._cache.data == before
    assert adapter.upload(request(), api_key=Secret(KEY, name="RF_KEY")) == (
        False,
        "Successfully registered image",
    )
    assert len(refunds) == 1
