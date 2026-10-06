import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import requests
from fastapi import Request, Response
from PIL import Image

from inference_server import configuration, pingback
from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.framework.entities import (
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    SyncLegacyBridge,
)
from inference_server.legacy.entities import (
    ClassificationInferenceRequest,
    ClassificationInferenceResponse,
    ClipEmbeddingResponse,
    ClipTextEmbeddingRequest,
    InferenceRequestImage,
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy.conftest import FakeGateway

PLANTED_KEY = "planted-secret-key-123"
PLANTED_URL = "https://gateway.example.test/proxy"
NOW = 1_700_000_000.0


@pytest.fixture(autouse=True)
def pingback_settings(monkeypatch):
    monkeypatch.setattr(configuration, "METRICS_ENABLED", True)
    monkeypatch.setattr(configuration, "METRICS_INTERVAL", 60)
    monkeypatch.setattr(
        configuration, "METRICS_URL", "https://api.test/inference-stats"
    )
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", False)
    monkeypatch.setattr(configuration, "DISABLE_INFERENCE_CACHE", False)
    monkeypatch.setattr(configuration, "TINY_CACHE", True)
    monkeypatch.setattr(configuration, "METRICS_API_KEY", None)
    monkeypatch.setattr(configuration, "ROBOFLOW_API_VERIFY_SSL", True)
    monkeypatch.setattr(configuration, "SERVER_VERSION", "9.9.9")
    monkeypatch.setattr(configuration, "INFERENCE_SERVER_ID", "srv-1")
    monkeypatch.setattr(configuration, "DEVICE_ID", "device-7")
    monkeypatch.setattr(configuration, "TAGS", ["a", "b"])
    monkeypatch.setattr(pingback, "RECORDER", pingback.InferenceRecorder())
    monkeypatch.setattr(pingback, "_SENDER", None)


@pytest.fixture
def clock(monkeypatch):
    current = [NOW]
    monkeypatch.setattr(pingback, "_clock", lambda: current[0])

    return current


@pytest.fixture
def fixed_system(monkeypatch):
    monkeypatch.setattr(pingback.platform, "system", lambda: "Linux")
    monkeypatch.setattr(pingback.platform, "release", lambda: "6.1")
    monkeypatch.setattr(pingback.platform, "version", lambda: "#1 SMP")
    monkeypatch.setattr(pingback.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(pingback.platform, "processor", lambda: "x86_64")
    monkeypatch.setattr(pingback.socket, "gethostname", lambda: "host-1")
    monkeypatch.setattr(pingback.socket, "gethostbyname", lambda name: "10.0.0.5")
    monkeypatch.setattr(pingback.uuid, "getnode", lambda: 0x001122334455)


@pytest.fixture
def module_log():
    lines = []

    class _Handler(logging.Handler):
        def emit(self, record):
            lines.append(self.format(record))

    handler = _Handler(level=logging.DEBUG)
    module_logger = logging.getLogger("inference_server.pingback")
    previous_level = module_logger.level
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.DEBUG)
    yield lines
    module_logger.removeHandler(handler)
    module_logger.setLevel(previous_level)


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def _post(url, **kwargs):
        calls.append((url, kwargs))
        return MagicMock(status_code=200)

    monkeypatch.setattr(pingback.requests, "post", _post)

    return calls


IMAGE = {"type": "url", "value": "https://images.test/a.jpg"}
DIMENSIONS = InferenceResponseImage(width=8, height=6)


def _od_request(**kwargs):
    arguments = {
        "id": "req-1",
        "model_id": "ds/1",
        "api_key": "req-key",
        "image": IMAGE,
        "confidence": 0.4,
    }
    arguments.update(kwargs)
    request = ObjectDetectionInferenceRequest(**arguments)

    return request


def _od_prediction(class_name, confidence, prediction_type=ObjectDetectionPrediction):
    prediction = prediction_type(
        x=1.0,
        y=1.0,
        width=2.0,
        height=2.0,
        confidence=confidence,
        class_id=0,
        **{"class": class_name},
    )

    return prediction


def _od_response(pairs=(("cat", 0.9), ("dog", 0.8)), elapsed=0.25):
    response = ObjectDetectionInferenceResponse(
        predictions=[_od_prediction(name, score) for name, score in pairs],
        image=DIMENSIONS,
    )
    response.time = elapsed

    return response


def _record(model_id="ds/1", request=None, response=None):
    pingback.record_inference(
        model_id,
        _od_request() if request is None else request,
        _od_response() if response is None else response,
    )


def _results(start=NOW - 60, stop=NOW):
    results = pingback.RECORDER.results(start=start, stop=stop)

    return results


def _jpeg_b64(width=8, height=6):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    return encoded


def test_posted_payload_equals_the_legacy_payload(
    clock, fixed_system, posts, monkeypatch
):
    monkeypatch.setattr(configuration, "METRICS_API_KEY", "env-key")
    sender = pingback.PingbackSender()
    clock[0] = NOW + 30
    _record()
    clock[0] = NOW + 40
    classification = ClassificationInferenceResponse(
        image=DIMENSIONS,
        predictions=[
            {"class": "b", "class_id": 1, "confidence": 0.7123},
            {"class": "c", "class_id": 2, "confidence": 0.6},
        ],
        top="b",
        confidence=0.7123,
    )
    classification.time = 0.5
    _record(
        "cls/2",
        ClassificationInferenceRequest(
            id="req-2",
            model_id="cls/2",
            api_key="other-key",
            image=IMAGE,
            confidence=0.5,
            source="workflow-execution",
            source_info="step-1",
        ),
        classification,
    )
    clock[0] = NOW + 60

    sender.post()

    assert len(posts) == 1
    url, kwargs = posts[0]
    assert url == "https://api.test/inference-stats"
    assert kwargs == {
        "json": {
            "api_key": "env-key",
            "timestamp": "1700000000",
            "device_id": "device-7",
            "inference_server_id": "srv-1",
            "inference_server_version": "9.9.9",
            "tags": ["a", "b"],
            "platform": "Linux",
            "platform_release": "6.1",
            "platform_version": "#1 SMP",
            "architecture": "x86_64",
            "hostname": "host-1",
            "ip_address": "10.0.0.5",
            "mac_address": "00:11:22:33:44:55",
            "processor": "x86_64",
            "inference_results": [
                {
                    "request_time": NOW + 30,
                    "inference": {
                        "inference_id": "req-1",
                        "inference_server_version": "9.9.9",
                        "inference_server_id": "srv-1",
                        "request": {
                            "api_key": "req-key",
                            "confidence": 0.4,
                            "model_id": "ds/1",
                            "model_type": None,
                            "source": None,
                            "source_info": None,
                        },
                        "response": [
                            {
                                "predictions": [
                                    {"class": "cat", "confidence": 0.9},
                                    {"class": "dog", "confidence": 0.8},
                                ],
                                "time": 0.25,
                            }
                        ],
                    },
                },
                {
                    "request_time": NOW + 40,
                    "inference": {
                        "inference_id": "req-2",
                        "inference_server_version": "9.9.9",
                        "inference_server_id": "srv-1",
                        "request": {
                            "api_key": "other-key",
                            "confidence": 0.5,
                            "model_id": "cls/2",
                            "model_type": "classification",
                            "source": "workflow-execution",
                            "source_info": "step-1",
                        },
                        "response": [
                            {
                                "predictions": [
                                    {"class": "b", "confidence": 0.7123},
                                    {"class": "c", "confidence": 0.6},
                                ],
                                "time": 0.5,
                            }
                        ],
                    },
                },
            ],
        },
        "timeout": 10,
    }


def test_system_info_keeps_the_keys_read_before_a_failure(fixed_system, monkeypatch):
    def _unresolvable(name):
        raise OSError(PLANTED_KEY)

    monkeypatch.setattr(pingback.socket, "gethostbyname", _unresolvable)

    assert pingback.get_system_info() == {
        "platform": "Linux",
        "platform_release": "6.1",
        "platform_version": "#1 SMP",
        "architecture": "x86_64",
        "hostname": "host-1",
    }


def test_device_id_falls_back_to_the_host_name(fixed_system, monkeypatch):
    monkeypatch.setattr(configuration, "DEVICE_ID", None)
    monkeypatch.setattr(pingback.platform, "node", lambda: "node-9")

    assert pingback.PingbackSender().build_payload()["device_id"] == "node-9"


def test_item_of_a_list_response_keeps_no_image(clock):
    request = _od_request(image=[IMAGE, IMAGE])
    responses = [
        _od_response((("cat", 0.9),), elapsed=0.25),
        _od_response((("dog", 0.7),), elapsed=0.5),
    ]

    item = pingback.to_cachable_inference_item(request, responses)

    assert item == {
        "inference_id": "req-1",
        "inference_server_version": "9.9.9",
        "request": {
            "api_key": "req-key",
            "confidence": 0.4,
            "model_id": "ds/1",
            "model_type": None,
            "source": None,
            "source_info": None,
        },
        "response": [
            {"predictions": [{"class": "cat", "confidence": 0.9}], "time": 0.25},
            {"predictions": [{"class": "dog", "confidence": 0.7}], "time": 0.5},
        ],
    }


def test_item_of_a_single_response_is_a_one_element_list(clock):
    item = pingback.to_cachable_inference_item(
        _od_request(), _od_response((("dog", 0.9),))
    )

    assert item["response"] == [
        {"predictions": [{"class": "dog", "confidence": 0.9}], "time": 0.25}
    ]


def test_response_without_predictions_is_left_out(clock):
    item = pingback.to_cachable_inference_item(
        _od_request(), [_od_response(()), _od_response((("cat", 0.6),))]
    )

    assert item["response"] == [
        {"predictions": [{"class": "cat", "confidence": 0.6}], "time": 0.25}
    ]


def test_client_supplied_source_and_model_type_are_kept(clock):
    request = _od_request(
        source="sdk", source_info="camera-3", model_type="object-detection"
    )

    item = pingback.to_cachable_inference_item(request, _od_response())

    assert item["request"] == {
        "api_key": "req-key",
        "confidence": 0.4,
        "model_id": "ds/1",
        "model_type": "object-detection",
        "source": "sdk",
        "source_info": "camera-3",
    }


def test_core_model_item_has_only_the_keys_of_its_request_class(clock):
    request = ClipTextEmbeddingRequest(id="req-3", text="hi", api_key="clip-key")
    response = ClipEmbeddingResponse(embeddings=[[1.0, 2.0]])

    item = pingback.to_cachable_inference_item(request, response)

    assert item == {
        "inference_id": "req-3",
        "inference_server_version": "9.9.9",
        "request": {
            "api_key": "clip-key",
            "model_id": request.model_id,
            "source": None,
            "source_info": None,
        },
        "response": [],
    }


def test_instance_segmentation_item(clock):
    response = InstanceSegmentationInferenceResponse(
        predictions=[
            InstanceSegmentationPrediction(
                x=1.0,
                y=1.0,
                width=2.0,
                height=2.0,
                confidence=0.9,
                class_id=0,
                points=[{"x": 0.0, "y": 0.0}],
                **{"class": "cat"},
            )
        ],
        image=DIMENSIONS,
    )
    response.time = 0.25

    item = pingback.to_cachable_inference_item(_od_request(), response)

    assert item["response"] == [
        {"predictions": [{"class": "cat", "confidence": 0.9}], "time": 0.25}
    ]


def test_multi_label_classification_item(clock):
    response = MultiLabelClassificationInferenceResponse(
        predictions={
            "a": {"confidence": 0.2, "class_id": 0},
            "b": {"confidence": 0.9, "class_id": 1},
        },
        predicted_classes=["b"],
        image=DIMENSIONS,
    )
    response.time = 0.25
    request = ClassificationInferenceRequest(
        id="req-4", model_id="mlc/1", api_key="req-key", image=IMAGE
    )

    item = pingback.to_cachable_inference_item(request, response)

    assert item["request"]["model_type"] == "classification"
    assert item["response"] == [
        {
            "predictions": [
                {"class": "a", "confidence": 0.2},
                {"class": "b", "confidence": 0.9},
            ],
            "time": 0.25,
        }
    ]


def test_keypoints_item_lists_every_keypoint(clock):
    prediction = KeypointsPrediction(
        x=1.0,
        y=1.0,
        width=2.0,
        height=2.0,
        confidence=0.8,
        class_id=0,
        keypoints=[
            {"x": 1.0, "y": 1.0, "confidence": 0.9, "class_id": 0, "class": "nose"},
            {"x": 2.0, "y": 2.0, "confidence": 0.5, "class_id": 2, "class": "ear"},
        ],
        **{"class": "person"},
    )
    response = KeypointsDetectionInferenceResponse(
        predictions=[prediction], image=DIMENSIONS
    )
    response.time = 0.25

    item = pingback.to_cachable_inference_item(_od_request(), response)

    assert item["response"] == [
        {
            "predictions": [
                {"class": "nose", "confidence": 0.9},
                {"class": "ear", "confidence": 0.5},
            ],
            "time": 0.25,
        }
    ]


def test_not_recorded_when_the_request_opted_out(clock):
    _record(request=_od_request(disable_model_monitoring=True))
    pingback.remember_api_key("req-key", monitoring=False)

    assert _results() == []
    assert pingback.RECORDER.fallback_api_key is None


def test_not_recorded_when_metrics_are_disabled(clock, monkeypatch):
    monkeypatch.setattr(configuration, "METRICS_ENABLED", False)
    _record()
    pingback.remember_api_key("req-key", monitoring=True)

    assert _results() == []
    assert pingback.RECORDER.fallback_api_key is None


def test_not_recorded_in_offline_mode(clock, monkeypatch):
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)
    _record()

    assert _results() == []


def test_not_recorded_when_the_inference_cache_is_disabled(clock, monkeypatch):
    monkeypatch.setattr(configuration, "DISABLE_INFERENCE_CACHE", True)
    _record()

    assert _results() == []


def _detections(confidences=(0.9,), class_ids=(0,)):
    count = len(confidences)
    detections = SimpleNamespace(
        xyxy=np.tile(np.array([1.0, 1.0, 3.0, 5.0]), (count, 1)),
        confidence=np.array(confidences, dtype=float),
        class_id=np.array(class_ids),
    )

    return detections


def _detection_gateway(detections=None):
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): detections or _detections()},
        model_info={"ds/1": {"class_names": ["cat", "dog"], "actions": {"infer": {}}}},
    )

    return gateway


def _infer_body(**extra):
    body = {
        "model_id": "ds/1",
        "api_key": "k",
        "image": {"type": "base64", "value": _jpeg_b64()},
        "confidence": 0.3,
    }
    body.update(extra)

    return body


def test_legacy_inference_route_records_exactly_one_item(
    legacy_client, fake_stat, clock
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway(_detections((0.9, 0.8), (0, 1))))

    response = client.post(
        "/infer/object_detection",
        json=_infer_body(class_filter=["cat"], source="sdk", source_info="camera-3"),
    )

    assert response.status_code == 200
    body = response.json()
    (result,) = _results()
    assert result == {
        "request_time": NOW,
        "inference": {
            "inference_id": body["inference_id"],
            "inference_server_version": "9.9.9",
            "inference_server_id": "srv-1",
            "request": {
                "api_key": "k",
                "confidence": 0.3,
                "model_id": "ds/1",
                "model_type": None,
                "source": "sdk",
                "source_info": "camera-3",
            },
            "response": [
                {
                    "predictions": [{"class": "cat", "confidence": 0.9}],
                    "time": body["time"],
                }
            ],
        },
    }
    assert pingback.RECORDER.fallback_api_key == "k"


def test_legacy_inference_route_opt_out_is_not_recorded(
    legacy_client, fake_stat, clock
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    response = client.post(
        "/infer/object_detection", json=_infer_body(disable_model_monitoring=True)
    )

    assert response.status_code == 200
    assert _results() == []
    assert pingback.RECORDER.fallback_api_key is None


def test_core_model_route_records_exactly_one_item(legacy_client, fake_stat, clock):
    gateway = FakeGateway(
        predictions={
            ("doctr/default", "infer"): (
                ["hi"],
                [
                    SimpleNamespace(
                        xyxy=np.zeros((0, 4)),
                        confidence=np.zeros(0),
                        class_id=np.zeros(0),
                    )
                ],
            )
        },
        model_info={"doctr/default": {"actions": {"infer": {}}}},
    )

    response = legacy_client(gateway).post(
        "/doctr/ocr",
        json={"image": {"type": "base64", "value": _jpeg_b64()}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    (result,) = _results()
    assert result["inference"]["response"] == []
    assert result["inference"]["request"] == {
        "api_key": "k",
        "model_id": "doctr/default",
        "source": None,
        "source_info": None,
    }


def test_embedding_route_records_exactly_one_item_with_empty_response(
    legacy_client, fake_stat, clock
):
    gateway = FakeGateway(
        predictions={("clip/ViT-B-16", "embed_text"): np.array([[1.0, 2.0]])},
        model_info={
            "clip/ViT-B-16": {"actions": {"embed_text": {}, "embed_images": {}}}
        },
    )

    response = legacy_client(gateway).post(
        "/clip/embed_text", json={"text": "hello", "api_key": "k"}
    )

    assert response.status_code == 200, response.text
    (result,) = _results()
    assert result["inference"]["response"] == []
    assert set(result["inference"]["request"]) == {
        "api_key",
        "model_id",
        "source",
        "source_info",
    }
    assert result["inference"]["request"]["api_key"] == "k"
    assert pingback.RECORDER.fallback_api_key == "k"


def test_embedding_route_opt_out_is_not_recorded(legacy_client, fake_stat, clock):
    gateway = FakeGateway(
        predictions={("clip/ViT-B-16", "embed_text"): np.array([[1.0, 2.0]])},
        model_info={
            "clip/ViT-B-16": {"actions": {"embed_text": {}, "embed_images": {}}}
        },
    )

    response = legacy_client(gateway).post(
        "/clip/embed_text",
        json={"text": "hello", "api_key": "k", "disable_model_monitoring": True},
    )

    assert response.status_code == 200, response.text
    assert _results() == []
    assert pingback.RECORDER.fallback_api_key is None


@pytest.mark.asyncio
async def test_workflow_step_records_exactly_one_item(fake_stat, clock):
    fake_stat["ds/1"] = ("object-detection", "infer")
    loop = asyncio.get_running_loop()
    sync = SyncLegacyBridge(LegacyModelBridge(_detection_gateway()), LoopBridge(loop))
    provider = GatewayModelsProvider(sync, api_key="wf-key")
    image = {"type": "numpy_object", "value": np.zeros((6, 8, 3), dtype=np.uint8)}

    def _step():
        predictions = provider.run_object_detection(
            "ds/1", [image], api_key="wf-key", confidence=0.2
        )

        return predictions

    with ThreadPoolExecutor(max_workers=1) as pool:
        predictions = await loop.run_in_executor(pool, _step)

    (result,) = _results()
    assert result["inference"]["inference_id"] == predictions[0]["inference_id"]
    assert result["inference"]["request"] == {
        "api_key": "wf-key",
        "confidence": 0.2,
        "model_id": "ds/1",
        "model_type": None,
        "source": "workflow-execution",
        "source_info": None,
    }
    assert result["inference"]["response"] == [
        {
            "predictions": [{"class": "cat", "confidence": 0.9}],
            "time": predictions[0]["time"],
        }
    ]
    assert pingback.RECORDER.fallback_api_key == "wf-key"


@pytest.mark.asyncio
async def test_bridge_call_alone_records_nothing(fake_stat, clock):
    fake_stat["ds/1"] = ("object-detection", "infer")
    bridge = LegacyModelBridge(_detection_gateway())
    route = await bridge.resolve("ds/1", "k")

    await bridge.infer(route, "k", "infer", [None], {})

    assert _results() == []
    assert pingback.RECORDER.fallback_api_key is None


@pytest.mark.asyncio
async def test_v2_inference_is_not_recorded(clock):
    async def handler(action, input_data, proxy, hooks):
        return _detections()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": b"model_id=ds/1&instance=",
        "headers": [(b"authorization", b"Bearer k1")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    proxy = SimpleNamespace(ensure_loaded=AsyncMock(return_value=("model_ready",)))
    keys_before = set(_HANDLERS)
    _HANDLERS[("pingback-task", "infer")] = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=handler,
        output_serializer=lambda prediction, common: Response(status_code=200),
        interface_provider=lambda: ModelInterfaceDescription(
            task="t", params={}, output_schema={}
        ),
    )
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("pingback-task", "infer")),
        ):
            response = await handle_model_inference_request(
                Request(scope, receive), proxy
            )
    finally:
        for key in set(_HANDLERS) - keys_before:
            del _HANDLERS[key]

    assert response.status_code == 200
    assert _results() == []
    assert pingback.RECORDER.fallback_api_key is None


def test_recording_failure_does_not_fail_the_inference(
    legacy_client, fake_stat, monkeypatch
):
    def _broken(*args, **kwargs):
        raise RuntimeError("recorder broke")

    monkeypatch.setattr(pingback.RECORDER, "record", _broken)
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    response = client.post("/infer/object_detection", json=_infer_body())

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1
    assert pingback.RECORDER.failures == 1


def test_unreadable_request_is_counted_not_raised(clock, module_log):
    _record(request=object())

    assert _results() == []
    assert pingback.RECORDER.failures == 1
    assert module_log == []


def test_unreadable_response_is_skipped_and_counted(clock, module_log):
    broken = _od_response()
    broken.predictions = [object()]

    _record(response=[broken, _od_response((("cat", 0.6),))])

    (result,) = _results()
    assert result["inference"]["response"] == [
        {"predictions": [{"class": "cat", "confidence": 0.6}], "time": 0.25}
    ]
    assert pingback.RECORDER.failures == 1
    assert module_log == []


def test_configured_api_key_wins_over_the_request_key(fixed_system, monkeypatch):
    monkeypatch.setattr(configuration, "METRICS_API_KEY", "env-key")
    sender = pingback.PingbackSender()
    pingback.remember_api_key("req-key", monitoring=True)

    assert sender.build_payload()["api_key"] == "env-key"


def test_most_recent_request_key_is_used_without_a_configured_key(fixed_system):
    sender = pingback.PingbackSender()
    pingback.remember_api_key("first-key", monitoring=True)
    pingback.remember_api_key("second-key", monitoring=True)

    assert sender.build_payload()["api_key"] == "second-key"


def test_api_key_is_null_without_any_key(fixed_system):
    payload = pingback.PingbackSender().build_payload()

    assert "api_key" in payload
    assert payload["api_key"] is None


def test_empty_configured_key_is_posted_as_is(fixed_system, monkeypatch):
    monkeypatch.setattr(configuration, "METRICS_API_KEY", "")

    assert pingback.PingbackSender().build_payload()["api_key"] == ""


def test_per_model_cap_keeps_the_newest_items(clock):
    total = pingback.MAX_ITEMS_PER_MODEL + 5
    for index in range(total):
        clock[0] = NOW + index * 0.001
        _record(request=_od_request(id=f"id-{index + 1}"))

    results = _results(start=NOW, stop=NOW + 1)

    assert len(results) == pingback.MAX_ITEMS_PER_MODEL
    assert results[0]["inference"]["inference_id"] == "id-6"
    assert results[-1]["inference"]["inference_id"] == f"id-{total}"


def test_items_older_than_twice_the_interval_are_dropped(clock):
    _record(request=_od_request(id="id-1"))
    clock[0] = NOW + 121
    _record(request=_od_request(id="id-2"))

    results = _results(start=0, stop=NOW + 200)

    assert [result["inference"]["inference_id"] for result in results] == ["id-2"]


def test_results_cover_only_the_last_interval(clock):
    _record(request=_od_request(id="id-1"))
    clock[0] = NOW + 90
    _record(request=_od_request(id="id-2"))
    clock[0] = NOW + 100

    payload = pingback.PingbackSender().build_payload()

    assert [
        result["inference"]["inference_id"] for result in payload["inference_results"]
    ] == ["id-2"]


def test_model_cap_drops_the_least_recently_recorded_model(clock):
    total = pingback.MAX_RECORDED_MODELS + 3
    for index in range(total):
        _record(f"ds/{index}", _od_request(model_id=f"ds/{index}"))

    model_ids = [result["inference"]["request"]["model_id"] for result in _results()]

    assert model_ids == [f"ds/{index}" for index in range(3, total)]


def test_predictions_per_item_are_capped(clock):
    count = pingback.MAX_PREDICTIONS_PER_ITEM + 50
    _record(response=[_od_response([("cat", 0.5)] * count)] * 2)

    (result,) = _results()
    stored = sum(
        len(response["predictions"]) for response in result["inference"]["response"]
    )

    assert stored == pingback.MAX_PREDICTIONS_PER_ITEM
    assert all(response["predictions"] for response in result["inference"]["response"])


def test_sender_is_not_started_when_disabled(monkeypatch):
    monkeypatch.setattr(configuration, "METRICS_ENABLED", False)

    assert pingback.start_sender() is None
    assert _sender_threads() == []


def _sender_threads():
    threads = [
        thread
        for thread in threading.enumerate()
        if thread.name == pingback.SENDER_THREAD_NAME
    ]

    return threads


def _configuration_flags(env):
    code = (
        "from inference_server import configuration as c; "
        "print(c.METRICS_ENABLED, c.METRICS_INTERVAL, c.METRICS_URL, c.TAGS, "
        "repr(c.METRICS_API_KEY))"
    )
    cleared = {
        name: value
        for name, value in os.environ.items()
        if name
        not in {
            "METRICS_ENABLED",
            "METRICS_INTERVAL",
            "METRICS_URL",
            "TAGS",
            "LAMBDA",
            "GCP_SERVERLESS",
            "OFFLINE_MODE",
            "API_BASE_URL",
            "ROBOFLOW_API_KEY",
            "API_KEY",
        }
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**cleared, **env},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, "True 60 https://api.roboflow.com/inference-stats [''] None"),
        (
            {
                "METRICS_ENABLED": "False",
                "METRICS_INTERVAL": "15",
                "METRICS_URL": "https://stats.test/x",
                "TAGS": "a,b",
            },
            "False 15 https://stats.test/x ['a', 'b'] None",
        ),
        (
            {"GCP_SERVERLESS": "true", "API_BASE_URL": "https://api.test"},
            "False 60 https://api.test/inference-stats [''] None",
        ),
        (
            {"LAMBDA": "true", "ROBOFLOW_API_KEY": "", "API_KEY": ""},
            "False 60 https://api.roboflow.com/inference-stats [''] ''",
        ),
        (
            {"ROBOFLOW_API_KEY": "", "API_KEY": "second"},
            "True 60 https://api.roboflow.com/inference-stats [''] 'second'",
        ),
        (
            {"ROBOFLOW_API_KEY": ""},
            "True 60 https://api.roboflow.com/inference-stats [''] None",
        ),
        (
            {"_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START": "true"},
            "False 60 https://api.roboflow.com/inference-stats [''] None",
        ),
    ],
)
def test_settings_and_the_rule_forcing_metrics_off(env, expected):
    assert _configuration_flags(env) == expected


class _StubProxy:
    async def start(self):
        pass

    async def shutdown(self):
        pass


@pytest.fixture
def lifespan_app(monkeypatch):
    import inference_server.app as app_mod

    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: _StubProxy()
    )

    return app_mod


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_the_sender(lifespan_app, posts):
    async with lifespan_app._lifespan(lifespan_app.app):
        running = _sender_threads()

    assert len(running) == 1
    assert running[0].daemon is True
    assert running[0].is_alive() is False
    assert _sender_threads() == []
    assert posts == []


@pytest.mark.asyncio
async def test_lifespan_starts_no_sender_when_metrics_are_disabled(
    lifespan_app, monkeypatch
):
    monkeypatch.setattr(configuration, "METRICS_ENABLED", False)

    async with lifespan_app._lifespan(lifespan_app.app):
        running = _sender_threads()

    assert running == []


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)

    return False


def test_failed_posts_do_not_stop_later_posts_and_log_no_secret(
    fixed_system, module_log, monkeypatch
):
    monkeypatch.setattr(configuration, "METRICS_INTERVAL", 0.01)
    monkeypatch.setattr(configuration, "METRICS_URL", PLANTED_URL)
    monkeypatch.setattr(configuration, "METRICS_API_KEY", PLANTED_KEY)
    outcomes = [
        RuntimeError(f"boom {PLANTED_KEY} {PLANTED_URL}"),
        requests.exceptions.Timeout(f"slow {PLANTED_URL}?api_key={PLANTED_KEY}"),
        MagicMock(status_code=502, text=PLANTED_KEY),
    ]
    calls = []

    def _post(url, **kwargs):
        calls.append(kwargs["json"]["api_key"])
        outcome = outcomes.pop(0) if outcomes else MagicMock(status_code=200)
        if isinstance(outcome, Exception):
            raise outcome

        return outcome

    monkeypatch.setattr(pingback.requests, "post", _post)
    _record(request=_od_request(api_key=PLANTED_KEY))
    sender = pingback.PingbackSender()

    sender.start()
    try:
        assert _wait_for(lambda: len(calls) >= 4)
    finally:
        sender.stop()

    assert calls[:4] == [PLANTED_KEY] * 4
    assert _sender_threads() == []
    logged = "\n".join(module_log)
    assert "RuntimeError" in logged
    assert "Timeout" in logged
    assert "502" in logged
    assert PLANTED_KEY not in logged
    assert PLANTED_URL not in logged
    assert "boom" not in logged
    assert "slow" not in logged
    assert "Traceback" not in logged


def test_posts_do_not_overlap(fixed_system, monkeypatch):
    monkeypatch.setattr(configuration, "METRICS_INTERVAL", 0.01)
    running = []
    overlaps = []
    calls = []

    def _post(url, **kwargs):
        overlaps.append(bool(running))
        running.append(1)
        time.sleep(0.05)
        running.pop()
        calls.append(url)

        return MagicMock(status_code=200)

    monkeypatch.setattr(pingback.requests, "post", _post)
    sender = pingback.PingbackSender()

    sender.start()
    sender.start()
    try:
        assert _wait_for(lambda: len(calls) >= 3)
    finally:
        sender.stop()

    assert len(_sender_threads()) == 0
    assert overlaps and not any(overlaps)


def test_post_goes_through_the_gateway_without_verification(
    fixed_system, posts, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_API_VERIFY_SSL", False)
    monkeypatch.setattr(pingback, "wrap_url", lambda url: f"{PLANTED_URL}?url={url}")

    pingback.PingbackSender().post()

    url, kwargs = posts[0]
    assert url == f"{PLANTED_URL}?url=https://api.test/inference-stats"
    assert kwargs["verify"] is False
    assert kwargs["timeout"] == 10


def test_no_post_in_offline_mode(fixed_system, posts, monkeypatch):
    sender = pingback.PingbackSender()
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)

    sender.post()

    assert posts == []


BIG = "x" * 1_000_000


def _string_length(value):
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_string_length(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_string_length(element) for element in value)

    return 0


def test_item_strings_are_truncated_without_touching_the_objects(clock):
    request = _od_request(
        id=BIG, api_key=BIG, model_id=BIG, source=BIG, source_info=BIG, model_type=BIG
    )
    response = _od_response(((BIG, 0.9),))

    _record(BIG, request, response)

    (result,) = _results()
    inference = result["inference"]
    limit = pingback.MAX_STRING_LENGTH
    assert limit == 256
    assert len(inference["inference_id"]) == limit
    for name in ("api_key", "model_id", "model_type", "source", "source_info"):
        assert len(inference["request"][name]) == limit
    assert len(inference["response"][0]["predictions"][0]["class"]) == limit
    assert [len(key) for key in pingback.RECORDER._items] == [limit]
    assert len(pingback.RECORDER.fallback_api_key) == limit
    assert len(request.source_info) == len(BIG)
    assert len(request.id) == len(BIG)
    assert len(response.predictions[0].class_name) == len(BIG)


def test_non_string_values_where_strings_are_expected_are_dropped(clock):
    request = _od_request()
    request.source = {"nested": BIG}
    request.source_info = [BIG]
    request.model_type = True
    request.api_key = 7

    item = pingback.to_cachable_inference_item(request, _od_response())

    assert item["request"]["source"] is None
    assert item["request"]["source_info"] is None
    assert item["request"]["model_type"] is True
    assert item["request"]["api_key"] == 7


def test_retained_string_volume_is_bounded_over_many_requests(clock):
    for index in range(500):
        clock[0] = NOW + index * 0.001
        _record(
            BIG,
            _od_request(id=BIG, api_key=BIG, model_id=BIG, source=BIG, source_info=BIG),
            _od_response([(BIG, 0.5)] * pingback.MAX_PREDICTIONS_PER_ITEM),
        )

    retained = sum(
        _string_length(item)
        for items in pingback.RECORDER._items.values()
        for _, item in items
    )

    assert sum(len(items) for items in pingback.RECORDER._items.values()) == 500
    assert retained < 500 * (8 + pingback.MAX_PREDICTIONS_PER_ITEM) * 256


def test_http_response_keeps_the_full_class_name(legacy_client, fake_stat, clock):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _detections((0.9,), (0,))},
        model_info={"ds/1": {"class_names": [BIG, "dog"], "actions": {"infer": {}}}},
    )

    response = legacy_client(gateway).post(
        "/infer/object_detection", json=_infer_body(source_info=BIG)
    )

    assert response.status_code == 200
    assert response.json()["predictions"][0]["class"] == BIG
    (result,) = _results()
    assert len(result["inference"]["request"]["source_info"]) == 256
    assert len(result["inference"]["response"][0]["predictions"][0]["class"]) == 256


class _CountingList(list):
    touched = 0

    def __iter__(self):
        for element in list.__iter__(self):
            self.touched += 1
            yield element


def _keypoints_response_without_keypoints(detections):
    prediction = KeypointsPrediction(
        x=1.0,
        y=1.0,
        width=2.0,
        height=2.0,
        confidence=0.8,
        class_id=0,
        keypoints=[],
        **{"class": "person"},
    )
    response = KeypointsDetectionInferenceResponse(predictions=[], image=DIMENSIONS)
    response.time = 0.25
    response.predictions = detections(prediction)

    return response


def test_detections_inspected_per_item_are_bounded(clock):
    response = _keypoints_response_without_keypoints(
        lambda prediction: _CountingList([prediction] * 100000)
    )
    counted = response.predictions

    item = pingback.to_cachable_inference_item(_od_request(), response)

    assert counted.touched <= pingback.MAX_DETECTIONS_INSPECTED_PER_ITEM
    assert item["response"] == [{"predictions": [], "time": 0.25}]


def test_responses_inspected_per_item_are_bounded(clock):
    responses = _CountingList([_od_response(())] * 10000)

    item = pingback.to_cachable_inference_item(_od_request(), responses)

    assert responses.touched <= pingback.MAX_RESPONSES_PER_ITEM
    assert item["response"] == []


def test_response_entries_per_item_are_bounded(clock):
    one = _keypoints_response_without_keypoints(list)
    responses = _CountingList([one] * 10000)

    item = pingback.to_cachable_inference_item(_od_request(), responses)

    assert responses.touched <= pingback.MAX_RESPONSES_PER_ITEM
    assert len(item["response"]) <= pingback.MAX_RESPONSES_PER_ITEM


def test_ordinary_requests_are_not_cut_by_the_inspection_bounds():
    assert pingback.MAX_RESPONSES_PER_ITEM == 100
    assert pingback.MAX_DETECTIONS_INSPECTED_PER_ITEM == 1000


def _run_scheduled(clock, monkeypatch, durations, builds_wanted):
    builds = []
    durations = list(durations)

    def _post(url, **kwargs):
        builds.append(clock[0] - NOW)
        clock[0] += durations.pop(0) if durations else 10
        return MagicMock(status_code=200)

    def _wait(timeout):
        assert timeout >= 0
        if len(builds) >= builds_wanted:
            return True
        clock[0] += timeout
        return False

    monkeypatch.setattr(pingback.requests, "post", _post)
    sender = pingback.PingbackSender(monotonic=lambda: clock[0], wait=_wait)
    sender._run()

    return builds


def test_report_windows_have_no_gaps(clock, fixed_system, monkeypatch):
    assert _run_scheduled(clock, monkeypatch, [10, 10, 10], 3) == [60, 120, 180]


def test_a_long_post_skips_missed_ticks_without_a_burst(
    clock, fixed_system, monkeypatch
):
    assert _run_scheduled(clock, monkeypatch, [130, 10, 10], 3) == [60, 240, 300]


def test_fallback_key_is_remembered_from_the_request_entity(clock):
    _record(request=_od_request(api_key="first-key"))
    _record(request=_od_request(api_key="second-key"))

    assert pingback.RECORDER.fallback_api_key == "second-key"


def test_request_without_a_key_clears_the_fallback_key_as_legacy_does(clock):
    _record(request=_od_request(api_key="first-key"))
    _record(request=_od_request(api_key=None))

    assert pingback.RECORDER.fallback_api_key is None


@pytest.mark.parametrize("flag", ["LEGACY_OFFLINE_MODE", "DISABLE_INFERENCE_CACHE"])
def test_fallback_key_does_not_depend_on_the_storage_gates(clock, monkeypatch, flag):
    monkeypatch.setattr(configuration, flag, True)

    _record(request=_od_request(api_key="kept-key"))

    assert _results() == []
    assert pingback.RECORDER.fallback_api_key == "kept-key"


def test_a_sender_is_started_once_per_process(fixed_system, posts):
    first = pingback.start_sender()
    second = pingback.start_sender()
    try:
        assert first is second
        assert len(_sender_threads()) == 1
    finally:
        first.stop()


def test_a_new_sender_starts_after_the_previous_one_stopped(fixed_system, posts):
    first = pingback.start_sender()
    first.stop()
    second = pingback.start_sender()
    try:
        assert second is not first
        assert len(_sender_threads()) == 1
    finally:
        second.stop()


def _slow_posts(monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    lock = threading.Lock()
    state = {"running": 0, "peak": 0}

    def _post(url, **kwargs):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        entered.set()
        release.wait(5)
        with lock:
            state["running"] -= 1

        return MagicMock(status_code=200)

    monkeypatch.setattr(pingback.requests, "post", _post)

    return release, entered, state


def test_no_second_sender_while_a_stopped_one_is_still_posting(
    fixed_system, monkeypatch
):
    monkeypatch.setattr(configuration, "METRICS_INTERVAL", 0.01)
    monkeypatch.setattr(pingback, "PREVIOUS_SENDER_JOIN_TIMEOUT_S", 0.05)
    release, entered, state = _slow_posts(monkeypatch)
    first = pingback.start_sender()
    assert entered.wait(5)
    first.stop(timeout=0.01)

    second = pingback.start_sender()
    release.set()
    first._thread.join(5)

    assert second is None
    assert state["peak"] == 1
    assert _sender_threads() == []


def test_a_new_sender_waits_for_the_stopped_one_to_finish_its_post(
    fixed_system, monkeypatch
):
    monkeypatch.setattr(configuration, "METRICS_INTERVAL", 0.01)
    release, entered, state = _slow_posts(monkeypatch)
    first = pingback.start_sender()
    assert entered.wait(5)
    first.stop(timeout=0.01)
    threading.Timer(0.05, release.set).start()

    second = pingback.start_sender()
    try:
        assert second is not None and second is not first
        assert _wait_for(lambda: state["running"] == 1 or state["peak"] >= 1)
        assert state["peak"] == 1
        assert len(_sender_threads()) == 1
    finally:
        second.stop()


class _Unserialisable:
    def __repr__(self):
        raise AssertionError("image was touched")


def _full_od_request(**kwargs):
    arguments = {
        "iou_threshold": 0.3,
        "max_detections": 300,
        "max_candidates": 3000,
    }
    arguments.update(kwargs)

    return _od_request(**arguments)


def _full_od_response(pairs=(("cat", 0.9), ("dog", 0.8)), elapsed=0.25):
    response = _od_response(pairs, elapsed)
    for index, prediction in enumerate(response.predictions):
        prediction.detection_id = f"det-{index}"

    return response


def _full_request_dump(**overrides):
    dump = {
        "id": "req-1",
        "api_key": "req-key",
        "usage_billable": True,
        "start": None,
        "source": None,
        "source_info": None,
        "disable_model_monitoring": False,
        "model_id": "ds/1",
        "model_type": None,
        "disable_preproc_auto_orient": False,
        "disable_preproc_contrast": False,
        "disable_preproc_grayscale": False,
        "disable_preproc_static_crop": False,
        "class_agnostic_nms": False,
        "class_filter": None,
        "confidence": 0.4,
        "fix_batch_size": False,
        "iou_threshold": 0.3,
        "max_detections": 300,
        "max_candidates": 3000,
        "visualization_labels": False,
        "visualization_stroke_width": 1,
        "visualize_predictions": False,
        "disable_active_learning": False,
        "active_learning_target_dataset": None,
    }
    dump.update(overrides)

    return dump


def _full_response_dump(index, class_name, confidence, elapsed):
    dump = {
        "visualization": None,
        "inference_id": None,
        "frame_id": None,
        "time": elapsed,
        "resolved_model": None,
        "predictions": [
            {
                "x": 1.0,
                "y": 1.0,
                "width": 2.0,
                "height": 2.0,
                "confidence": confidence,
                "class": class_name,
                "class_confidence": None,
                "class_id": 0,
                "tracker_id": None,
                "detection_id": f"det-{index}",
                "parent_id": None,
            }
        ],
    }

    return dump


@pytest.fixture
def full_cache(monkeypatch):
    monkeypatch.setattr(configuration, "TINY_CACHE", False)


def test_tiny_cache_is_on_by_default_and_read_from_the_environment():
    code = "from inference_server import configuration as c; print(c.TINY_CACHE)"
    outputs = []
    for env in ({}, {"TINY_CACHE": "False"}):
        cleared = {k: v for k, v in os.environ.items() if k != "TINY_CACHE"}
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**cleared, **env},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip().splitlines()[-1])

    assert outputs == ["True", "False"]


def test_full_item_of_a_single_response_keeps_every_field_but_the_images(
    clock, full_cache
):
    request = _full_od_request()
    response = _full_od_response((("cat", 0.9),))

    item = pingback.to_cachable_inference_item(request, response)

    assert item == {
        "inference_id": "req-1",
        "inference_server_version": "9.9.9",
        "request": _full_request_dump(),
        "response": _full_response_dump(0, "cat", 0.9, 0.25),
    }


def test_full_item_of_a_list_response_is_a_list_of_dumps(clock, full_cache):
    request = _full_od_request(image=[IMAGE, IMAGE])
    responses = [
        _full_od_response((("cat", 0.9),), elapsed=0.25),
        _full_od_response((("dog", 0.7),), elapsed=0.5),
    ]

    item = pingback.to_cachable_inference_item(request, responses)

    assert item["response"] == [
        _full_response_dump(0, "cat", 0.9, 0.25),
        _full_response_dump(0, "dog", 0.7, 0.5),
    ]
    assert "image" not in item["request"]


def test_full_item_is_recorded_and_reported_with_the_server_id(clock, full_cache):
    _record(request=_full_od_request(), response=_full_od_response((("cat", 0.9),)))

    (result,) = _results()

    assert result["inference"] == {
        "inference_id": "req-1",
        "inference_server_version": "9.9.9",
        "inference_server_id": "srv-1",
        "request": _full_request_dump(),
        "response": _full_response_dump(0, "cat", 0.9, 0.25),
    }


def test_full_item_never_dumps_the_request_image(clock, full_cache):
    request = _full_od_request(
        image=InferenceRequestImage(type="base64", value=_Unserialisable())
    )

    item = pingback.to_cachable_inference_item(request, _full_od_response())

    assert "image" not in item["request"]
    assert item["response"]["predictions"][0]["class"] == "cat"


def test_full_item_of_a_response_with_too_many_predictions_is_condensed(
    clock, full_cache
):
    count = pingback.MAX_FULL_NODES
    response = _od_response([("cat", 0.5)] * count)

    item = pingback.to_cachable_inference_item(_full_od_request(), response)

    assert item["request"]["model_id"] == "ds/1"
    assert set(item["request"]) == {
        "api_key",
        "confidence",
        "model_id",
        "model_type",
        "source",
        "source_info",
    }
    assert item["response"] == [
        {
            "predictions": [{"class": "cat", "confidence": 0.5}]
            * pingback.MAX_PREDICTIONS_PER_ITEM,
            "time": 0.25,
        }
    ]


def test_full_item_over_the_size_budget_is_condensed(clock, full_cache):
    name = "\x01" * (pingback.MAX_FULL_STRING_CHARS // 2)
    response = _od_response(((name, 0.9),))

    assert len(json.dumps(response.model_dump(mode="json", by_alias=True))) > (
        pingback.MAX_FULL_ITEM_BYTES
    )

    item = pingback.to_cachable_inference_item(_full_od_request(), response)

    assert set(item["request"]) == {
        "api_key",
        "confidence",
        "model_id",
        "model_type",
        "source",
        "source_info",
    }
    assert item["response"] == [
        {
            "predictions": [
                {"class": name[: pingback.MAX_STRING_LENGTH], "confidence": 0.9}
            ],
            "time": 0.25,
        }
    ]


def test_full_item_with_a_string_over_the_character_threshold_is_condensed(
    clock, full_cache
):
    request = _full_od_request(source_info="x" * (pingback.MAX_FULL_STRING_CHARS + 1))

    item = pingback.to_cachable_inference_item(request, _full_od_response())

    assert len(item["request"]["source_info"]) == pingback.MAX_STRING_LENGTH
    assert isinstance(item["response"], list)


def test_full_item_keeps_long_strings_within_the_budget_untruncated(clock, full_cache):
    request = _full_od_request(source_info="x" * 5000)

    item = pingback.to_cachable_inference_item(request, _full_od_response())

    assert item["request"]["source_info"] == "x" * 5000


def test_full_item_with_a_non_finite_number_is_condensed(clock, full_cache):
    response = _full_od_response((("cat", float("nan")),))

    item = pingback.to_cachable_inference_item(_full_od_request(), response)

    assert isinstance(item["response"], list)
    assert set(item["request"]) == {
        "api_key",
        "confidence",
        "model_id",
        "model_type",
        "source",
        "source_info",
    }


def test_full_item_of_a_response_that_is_not_an_entity_is_condensed(clock, full_cache):
    item = pingback.to_cachable_inference_item(
        _full_od_request(), {"normalized_depth": [[0.0]], "image": {"x": 1}}
    )

    assert item["response"] == []
    assert set(item["request"]) == {
        "api_key",
        "confidence",
        "model_id",
        "model_type",
        "source",
        "source_info",
    }


def test_full_item_work_does_not_grow_with_the_response(clock, full_cache):
    huge = _CountingList([_od_prediction("cat", 0.5)] * 1_000_000)
    response = _od_response(())
    response.predictions = huge

    item = pingback.to_cachable_inference_item(_full_od_request(), response)

    assert huge.touched <= (
        pingback.MAX_FULL_NODES + pingback.MAX_DETECTIONS_INSPECTED_PER_ITEM
    )
    assert item["response"] == [
        {
            "predictions": [{"class": "cat", "confidence": 0.5}]
            * pingback.MAX_PREDICTIONS_PER_ITEM,
            "time": 0.25,
        }
    ]


class _Captured(Exception):
    def __init__(self, request):
        super().__init__("captured")
        self.request = request


def _capture_request(position):
    def _capture(*args, **kwargs):
        raise _Captured(args[position])

    return _capture


_PIXELS = {"type": "numpy_object", "value": np.zeros((2, 2, 3), dtype=np.uint8)}
_LABELLED_METHODS = {
    "depth": ("_request_payloads", 0, lambda p: p.run_depth_estimation("m", _PIXELS)),
    "moondream2": (
        "_run_vlm",
        1,
        lambda p: p.run_moondream2("moondream2/m", _PIXELS, "q", ["t"], api_key="k"),
    ),
    "clip_text": (
        "_run_embedding",
        1,
        lambda p: p.run_clip_text_embedding("clip/v", "v", ["t"], api_key="k"),
    ),
    "clip_image": (
        "_run_embedding",
        1,
        lambda p: p.run_clip_image_embedding("clip/v", "v", [_PIXELS], api_key="k"),
    ),
    "clip_comparison": (
        "_run_embedding",
        1,
        lambda p: p.run_clip_comparison(
            _PIXELS, "image", ["t"], "text", api_key="k", version_id="v"
        ),
    ),
    "pe_text": (
        "_run_embedding",
        1,
        lambda p: p.run_perception_encoder_text_embedding(
            "pe/v", "v", ["t"], api_key="k"
        ),
    ),
    "pe_image": (
        "_run_embedding",
        1,
        lambda p: p.run_perception_encoder_image_embedding(
            "pe/v", "v", [_PIXELS], api_key="k"
        ),
    ),
    "doctr": (
        "_run_ocr",
        1,
        lambda p: p.run_doctr_ocr("doctr/default", _PIXELS, api_key="k"),
    ),
    "easy_ocr": (
        "_run_ocr",
        1,
        lambda p: p.run_easy_ocr("easy_ocr/v", "v", _PIXELS, api_key="k"),
    ),
    "pp_ocr": ("_run_ocr", 1, lambda p: p.run_pp_ocr(_PIXELS, api_key="k")),
    "yolo_world": (
        "_run_open_vocabulary",
        1,
        lambda p: p.run_yolo_world("yolo_world/v", "v", _PIXELS, ["t"], api_key="k"),
    ),
    "sam3": (
        "_run_interactive_segmentation",
        1,
        lambda p: p.run_sam3_segmentation(
            "sam3/m", _PIXELS, [{"type": "text", "text": "t"}], api_key="k"
        ),
    ),
}


@pytest.mark.parametrize("method", list(_LABELLED_METHODS))
def test_workflow_model_requests_are_recorded_with_the_workflow_source(clock, method):
    hook, position, call = _LABELLED_METHODS[method]
    provider = GatewayModelsProvider(SimpleNamespace(), api_key="k")
    provider.add_model = lambda *args, **kwargs: None
    provider._resolve = lambda *args, **kwargs: None
    setattr(provider, hook, _capture_request(position))

    with pytest.raises(_Captured) as captured:
        call(provider)
    pingback.record_inference("m", captured.value.request, [])

    (result,) = _results()
    assert result["inference"]["request"]["source"] == "workflow-execution"
