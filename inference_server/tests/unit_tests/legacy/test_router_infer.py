import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tests.unit_tests.legacy.conftest import FakeGateway


def _jpeg_b64(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _det():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def test_infer_object_detection_single(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
            "confidence": 0.3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 8, "height": 6}
    assert body["predictions"][0]["class"] == "cat"
    assert "time" in body and "inference_id" in body
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[3]["confidence"] == 0.3


def test_infer_object_detection_batch_shapes(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )
    c = legacy_client(gw)
    two = c.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "image": [{"type": "base64", "value": _jpeg_b64()}] * 2,
        },
    )
    assert two.status_code == 200 and isinstance(two.json(), list)
    assert len(two.json()) == 2
    one = c.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": [{"type": "base64", "value": _jpeg_b64()}]},
    )
    assert isinstance(one.json(), list) and len(one.json()) == 1
    none = c.post("/infer/object_detection", json={"model_id": "ds/1", "image": []})
    assert none.status_code == 200 and none.json() == []


def test_infer_response_carries_resolved_model_and_registry_tracks_alias(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("coco/3", "infer"): _det()},
        model_info={"coco/3": {"class_names": ["cat"]}},
    )
    c = legacy_client(gw)
    r = c.post(
        "/infer/object_detection",
        json={
            "model_id": "yolov8n-640",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )
    assert r.json()["resolved_model"] == {"model_id": "coco/3"}
    entry = c.get("/model/registry").json()["models"][0]
    assert entry["model_id"] == "coco/3"
    assert entry["request_aliases"] == ["yolov8n-640"]
    assert entry["request_paths"] == ["/infer/object_detection"]


def test_infer_response_carries_full_resolved_model_when_reported(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={
            "ds/1": {
                "class_names": ["cat"],
                "resolved_model": {
                    "model_id": "ds/1",
                    "model_package_id": "pkg",
                    "backend": "onnx",
                    "quantization": "fp32",
                },
            }
        },
    )
    r = legacy_client(gw).post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert r.json()["resolved_model"] == {
        "model_id": "ds/1",
        "model_package_id": "pkg",
        "backend": "onnx",
        "quantization": "fp32",
    }


def test_oversized_content_length_is_413(legacy_client, fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", 10)
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert r.status_code == 413 and r.json() == {
        "message": "Request payload too large."
    }


def test_wrong_task_type_is_400(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("classification", "infer")
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert r.status_code == 400


def test_bearer_header_used_when_body_key_missing(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )
    legacy_client(gw).post(
        "/infer/object_detection",
        headers={"Authorization": "Bearer H"},
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert ("ensure_loaded", "ds/1", "H") in gw.calls


def test_unsupported_legacy_param_is_501(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "image": {"type": "base64", "value": _jpeg_b64()},
            "disable_preproc_auto_orient": True,
        },
    )
    assert r.status_code == 501
