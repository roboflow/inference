import asyncio
import base64
import io
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

from inference_models.errors import UnauthorizedModelAccessError
from inference_sdk.config import apply_duration_minimum
from inference_server import configuration
from inference_server.legacy.errors import SERVICE_MISCONFIGURATION_MESSAGE
from inference_server.usage import request_hook
from inference_server.usage.collector import UsageCollector
from inference_server.usage.request_hook import (
    MODEL_INVOCATIONS,
    record_model_invocation,
    report_request_usage,
)
from tests.unit_tests.legacy.conftest import FakeGateway

SECRET = "service-secret-1"
OBJECT_DETECTION = ("object-detection", "infer", "yolov8", "yolov8-n")


def _jpeg(w=8, h=6) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (w, h)).save(buffer, format="JPEG")

    return buffer.getvalue()


def _image(w=8, h=6) -> dict:
    return {"type": "base64", "value": base64.b64encode(_jpeg(w, h)).decode()}


def _det():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def _raising(error):
    def _raise(image, params):
        raise error

    return _raise


@pytest.fixture
def detection_gateway(fake_stat):
    fake_stat["ds/1"] = OBJECT_DETECTION

    def _slow_det(image, params):
        time.sleep(0.02)
        return _det()

    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _slow_det},
        model_info={
            "ds/1": {
                "class_names": ["cat"],
                "actions": {"infer": {}},
                "input_height": 640,
                "input_width": 640,
            }
        },
    )

    return gateway


def _only_row(usage_collector):
    assert len(usage_collector.rows) == 1

    return usage_collector.rows[0]


def _assert_success_row(row, *, resource_id, api_key="k"):
    assert row["category"] == "request"
    assert row["api_key"] == api_key
    assert row["resource_id"] == resource_id
    assert row["frames"] == 1
    assert row["billable"] is True
    assert row["error_type"] is None
    assert row["error_status_code"] is None
    assert row["roboflow_internal_secret"] is None
    details = row["resource_details"]
    assert "error" not in details
    assert "dedicated_deployment_id" not in details
    assert "device_id" not in details


def test_object_detection_request_records_one_billable_row(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="ds/1")
    assert row["resource_details"]["models"] == [
        {
            "model_id": "ds/1",
            "model_architecture": "yolov8",
            "model_variant": "yolov8-n",
            "task_type": "object-detection",
            "model_input_height": 640,
            "model_input_width": 640,
            "execution_duration": pytest.approx(
                row["resource_details"]["models"][0]["execution_duration"]
            ),
            "frames": 1,
        }
    ]
    entry = row["resource_details"]["models"][0]
    assert "model_latency_ms" not in entry
    assert entry["execution_duration"] >= 0.02
    assert row["execution_duration"] >= entry["execution_duration"]
    assert "source" not in row["resource_details"]
    assert "source_info" not in row["resource_details"]
    assert row["roboflow_service_name"] is None


def test_source_tags_and_deployment_ids_are_recorded(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "DEDICATED_DEPLOYMENT_ID", "dd-1")
    monkeypatch.setattr(configuration, "DEVICE_ID", "device-1")
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection?source=sdk&source_info=sdk-1.0",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    details = row["resource_details"]
    assert details["dedicated_deployment_id"] == "dd-1"
    assert details["device_id"] == "device-1"
    assert details["source"] == "sdk"
    assert details["source_info"] == "sdk-1.0"
    assert row["roboflow_service_name"] == "sdk-1.0"


def test_source_tags_of_the_body_are_used_and_external_is_dropped(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection?source=external",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": _image(),
            "source": "ignored-by-query-external",
            "source_info": "body-info",
        },
    )

    assert response.status_code == 200, response.text
    details = _only_row(usage_collector)["resource_details"]
    assert details["source"] == "ignored-by-query-external"
    assert details["source_info"] == "body-info"


def test_anonymous_request_is_attributed_to_the_default_api_key(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", "env-key")
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection", json={"model_id": "ds/1", "image": _image()}
    )

    assert response.status_code == 200, response.text
    assert _only_row(usage_collector)["api_key"] == "env-key"


def test_batch_request_counts_one_frame_per_image_in_the_models_entry(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": [_image()] * 3},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["frames"] == 1
    assert row["resource_details"]["models"][0]["frames"] == 3


def test_backend_failure_records_an_error_row(usage_client, usage_collector, fake_stat):
    fake_stat["ds/1"] = OBJECT_DETECTION
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _raising(ValueError("bad shape"))}
    )
    client = usage_client(gateway)

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 400
    row = _only_row(usage_collector)
    assert row["resource_id"] == "ds/1"
    assert row["error_type"] == "ModelInputError"
    assert row["error_status_code"] is None
    assert row["resource_details"]["error"] == "ModelInputError: bad shape"
    assert row["resource_details"]["error_type"] == "ModelInputError"
    models = row["resource_details"]["models"]
    assert len(models) == 1
    assert models[0]["model_id"] == "ds/1"
    assert models[0]["frames"] == 1
    assert models[0]["execution_duration"] > 0
    assert "model_latency_ms" not in models[0]
    assert row["execution_duration"] >= models[0]["execution_duration"]


def test_registry_denial_records_the_cause_as_error_type(
    usage_client, usage_collector, fake_stat
):
    fake_stat["ds/1"] = UnauthorizedModelAccessError("denied")
    client = usage_client(FakeGateway())

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 401
    row = _only_row(usage_collector)
    assert row["error_type"] == "UnauthorizedModelAccessError"
    assert row["resource_details"]["error"] == "UnauthorizedModelAccessError: denied"
    assert "error_status_code" not in row["resource_details"]


def test_invalid_image_records_the_status_of_the_error(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": "bm90LWFuLWltYWdl"},
        },
    )

    assert response.status_code == 400
    row = _only_row(usage_collector)
    assert row["error_type"] == "LegacyHTTPError"
    assert row["error_status_code"] == 400
    assert row["resource_details"]["error"].startswith("LegacyHTTPError: ")


def test_wrong_task_type_is_an_error_row_with_the_status(
    usage_client, usage_collector, fake_stat
):
    fake_stat["ds/1"] = ("classification", "infer")
    client = usage_client(FakeGateway())

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 400
    row = _only_row(usage_collector)
    assert (row["error_type"], row["error_status_code"]) == ("LegacyHTTPError", 400)


def _all_log_text(caplog) -> str:
    formatter = logging.Formatter()

    return "\n".join(formatter.format(record) for record in caplog.records)


def test_planted_key_is_absent_from_the_error_and_the_logs(
    usage_client, usage_collector, fake_stat, caplog
):
    fake_stat["ds/1"] = OBJECT_DETECTION
    message = "https://api.roboflow.com/x?api_key=PLANTED-KEY api_key=PLANTED-KEY"
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _raising(RuntimeError(message))}
    )
    client = usage_client(gateway)

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/infer/object_detection",
            json={"model_id": "ds/1", "api_key": "k", "image": _image()},
        )

    assert response.status_code == 500
    row = _only_row(usage_collector)
    assert "PLANTED-KEY" not in row["resource_details"]["error"]
    assert row["resource_details"]["error"].startswith("RuntimeError: ")
    assert any(record.exc_info for record in caplog.records)
    assert "PLANTED-KEY" not in _all_log_text(caplog)


def test_bare_request_credentials_are_absent_from_the_error_and_the_logs(
    usage_client, usage_collector, fake_stat, caplog
):
    fake_stat["ds/1"] = OBJECT_DETECTION
    message = (
        "upstream said REQUEST-KEY-123 and SERVICE-SECRET-9 api_key=REQUEST-KEY-123 "
        "again REQUEST-KEY-123"
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _raising(RuntimeError(message))}
    )
    client = usage_client(gateway)

    with caplog.at_level(logging.DEBUG):
        caplog.set_level(logging.WARNING, logger="httpx")
        response = client.post(
            "/infer/object_detection?service_secret=SERVICE-SECRET-9",
            json={"model_id": "ds/1", "api_key": "REQUEST-KEY-123", "image": _image()},
        )

    assert response.status_code == 500
    error = _only_row(usage_collector)["resource_details"]["error"]
    assert error.startswith("RuntimeError: upstream said *** and *** ")
    assert "REQUEST-KEY-123" not in error
    assert "SERVICE-SECRET-9" not in error
    assert any(record.exc_info for record in caplog.records)
    log_text = _all_log_text(caplog)
    assert "REQUEST-KEY-123" not in log_text
    assert "SERVICE-SECRET-9" not in log_text


def test_credentials_shorter_than_six_characters_are_not_searched_by_value(
    usage_client, usage_collector, fake_stat
):
    fake_stat["ds/1"] = OBJECT_DETECTION
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _raising(RuntimeError("model abcde failed"))}
    )
    client = usage_client(gateway)

    client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "abcde", "image": _image()},
    )

    error = _only_row(usage_collector)["resource_details"]["error"]
    assert error == "RuntimeError: model abcde failed"


def test_error_message_is_bounded(usage_client, usage_collector, fake_stat):
    fake_stat["ds/1"] = OBJECT_DETECTION
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): _raising(RuntimeError("x" * 2000))}
    )
    client = usage_client(gateway)

    client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    error = _only_row(usage_collector)["resource_details"]["error"]
    assert error == "RuntimeError: " + "x" * 512


@pytest.mark.parametrize(
    "path,task_type",
    [
        ("/infer/instance_segmentation", "instance-segmentation"),
        ("/infer/classification", "classification"),
        ("/infer/keypoints_detection", "keypoint-detection"),
    ],
)
def test_other_cv_routes_record_the_wrong_task_type_as_an_error_row(
    usage_client, usage_collector, fake_stat, path, task_type
):
    fake_stat["ds/1"] = ("unknown-task", "infer")
    client = usage_client(FakeGateway())

    response = client.post(
        path, json={"model_id": "ds/1", "api_key": "k", "image": _image()}
    )

    assert response.status_code == 400, response.text
    row = _only_row(usage_collector)
    assert row["resource_id"] == "ds/1"
    assert row["error_status_code"] == 400


def _catch_all(client, query: str, **kwargs):
    return client.post(
        f"/ds/1?api_key=k{query}",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        **kwargs,
    )


def test_catch_all_post_records_a_row_keyed_by_the_path(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = _catch_all(client, "&source=sdk&source_info=sdk-2")

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="ds/1")
    assert row["resource_details"]["source"] == "sdk"
    assert row["resource_details"]["source_info"] == "sdk-2"
    assert row["resource_details"]["models"][0]["model_id"] == "ds/1"
    assert row["roboflow_service_name"] == "sdk-2"


def test_catch_all_default_source_tags_are_not_recorded(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = _catch_all(client, "")

    assert response.status_code == 200, response.text
    details = _only_row(usage_collector)["resource_details"]
    assert "source" not in details and "source_info" not in details


def test_catch_all_get_records_a_row(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    async def _fetch(urls, destination_policy=None):
        return [_jpeg() for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)
    client = usage_client(detection_gateway)

    response = client.get("/ds/1?api_key=k&image=https://images.example.com/cat.jpg")

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="ds/1")
    assert row["resource_details"]["models"][0]["frames"] == 1


def test_catch_all_missing_content_type_is_an_error_row(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = client.post("/ds/1?api_key=k")

    assert response.status_code == 400
    row = _only_row(usage_collector)
    assert row["resource_id"] == "ds/1"
    assert (row["error_type"], row["error_status_code"]) == ("LegacyHTTPError", 400)


def test_catch_all_refuses_countinference_false_without_a_secret(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(detection_gateway)

    response = _catch_all(client, "&countinference=false")

    assert response.status_code == 500
    assert response.json() == {"message": SERVICE_MISCONFIGURATION_MESSAGE}
    assert not any(call[0] == "infer" for call in detection_gateway.calls)
    row = _only_row(usage_collector)
    assert row["billable"] is True
    assert row["error_type"] == "MissingServiceSecretError"
    assert row["error_status_code"] == 500


def test_catch_all_refuses_countinference_false_with_a_wrong_secret(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(detection_gateway)

    response = _catch_all(client, "&countinference=false&service_secret=wrong")

    assert response.status_code == 500
    assert _only_row(usage_collector)["error_type"] == "MissingServiceSecretError"


def test_catch_all_refuses_countinference_false_when_no_secret_is_configured(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", None)
    client = usage_client(detection_gateway)

    response = _catch_all(client, f"&countinference=false&service_secret={SECRET}")

    assert response.status_code == 500
    assert _only_row(usage_collector)["error_type"] == "MissingServiceSecretError"


@pytest.mark.parametrize("spelling", ["false", "0", "no", "off"])
def test_catch_all_with_a_valid_secret_is_served_and_not_billable(
    usage_client, usage_collector, detection_gateway, monkeypatch, spelling
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(detection_gateway)

    response = _catch_all(
        client, f"&countinference={spelling}&service_secret={SECRET}&source_info=svc"
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["billable"] is False
    assert row["error_type"] is None
    assert row["roboflow_service_name"] == "svc"
    assert row["roboflow_internal_secret"] == SECRET


def test_catch_all_countinference_true_with_a_secret_stays_billable(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(detection_gateway)

    response = _catch_all(client, f"&countinference=true&service_secret={SECRET}")

    assert response.status_code == 200, response.text
    assert _only_row(usage_collector)["billable"] is True


def test_infer_route_ignores_countinference_false_without_a_secret(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection?countinference=false",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["billable"] is True
    assert row["error_type"] is None


@pytest.mark.parametrize("spelling", ["false", "0", "no", "off"])
def test_infer_route_with_a_valid_secret_is_not_billable(
    usage_client, usage_collector, detection_gateway, monkeypatch, spelling
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(detection_gateway)

    response = client.post(
        f"/infer/object_detection?countinference={spelling}&service_secret={SECRET}",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    assert _only_row(usage_collector)["billable"] is False


def _clip_gateway():
    gateway = FakeGateway(
        predictions={
            ("clip/ViT-B-16", "embed_text"): lambda image, params: np.array(
                [[1.0, 0.0]] * len(params["texts"])
            ),
            ("clip/ViT-B-16", "embed_images"): lambda image, params: np.array(
                [[1.0, 0.0]]
            ),
        },
        model_info={
            "clip/ViT-B-16": {
                "actions": {"embed_text": {}, "embed_images": {}, "compare": {}}
            }
        },
    )

    return gateway


def test_clip_embed_text_records_a_text_only_invocation(usage_client, usage_collector):
    client = usage_client(_clip_gateway())

    response = client.post("/clip/embed_text", json={"text": "hello", "api_key": "k"})

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="clip/ViT-B-16")
    assert row["resource_details"]["models"] == [
        {
            "model_id": "clip/ViT-B-16",
            "model_architecture": "clip",
            "model_variant": "ViT-B-16",
            "task_type": "embedding",
            "execution_duration": pytest.approx(
                row["resource_details"]["models"][0]["execution_duration"]
            ),
            "frames": 1,
        }
    ]


def test_clip_embed_image_batch_counts_every_image_on_one_entry(
    usage_client, usage_collector
):
    client = usage_client(_clip_gateway())

    response = client.post(
        "/clip/embed_image", json={"image": [_image()] * 4, "api_key": "k"}
    )

    assert response.status_code == 200, response.text
    models = _only_row(usage_collector)["resource_details"]["models"]
    assert len(models) == 1
    assert models[0]["frames"] == 4
    assert models[0]["model_id"] == "clip/ViT-B-16"


def test_clip_compare_combines_the_image_and_text_calls(usage_client, usage_collector):
    client = usage_client(_clip_gateway())

    response = client.post(
        "/clip/compare",
        json={
            "api_key": "k",
            "subject": _image(),
            "subject_type": "image",
            "prompt": {"x": "b", "y": "c"},
            "prompt_type": "text",
        },
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["resource_id"] == "clip/ViT-B-16"
    models = row["resource_details"]["models"]
    assert len(models) == 1
    assert models[0]["frames"] == 2


def test_perception_encoder_keeps_the_requested_model_id(usage_client, usage_collector):
    gateway = FakeGateway(
        predictions={
            ("perception-encoder/PE-Core-L14-336", "embed_text"): np.array([[1.0]])
        },
        model_info={
            "perception-encoder/PE-Core-L14-336": {"actions": {"embed_text": {}}}
        },
    )
    client = usage_client(gateway)

    response = client.post(
        "/perception_encoder/embed_text", json={"text": "hi", "api_key": "k"}
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["resource_id"] == "perception_encoder/PE-Core-L14-336"
    entry = row["resource_details"]["models"][0]
    assert entry["model_id"] == "perception_encoder/PE-Core-L14-336"
    assert entry["model_architecture"] == "perception_encoder"
    assert entry["model_variant"] == "PE-Core-L14-336"


def test_aliased_model_id_is_kept_on_the_entry(
    usage_client, usage_collector, fake_stat
):
    fake_stat["coco/3"] = OBJECT_DETECTION
    gateway = FakeGateway(
        predictions={("coco/3", "infer"): _det()},
        model_info={"coco/3": {"class_names": ["cat"]}},
    )
    client = usage_client(gateway)

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "yolov8n-640", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["resource_id"] == "yolov8n-640"
    assert row["resource_details"]["models"][0]["model_id"] == "yolov8n-640"


def test_doctr_records_a_structured_ocr_invocation(usage_client, usage_collector):
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
    client = usage_client(gateway)

    response = client.post("/doctr/ocr", json={"image": _image(), "api_key": "k"})

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="doctr/default")
    entry = row["resource_details"]["models"][0]
    assert entry["model_id"] == "doctr/default"
    assert entry["task_type"] == "structured-ocr"
    assert entry["frames"] == 1


def test_grounding_dino_records_an_open_vocabulary_invocation(
    usage_client, usage_collector
):
    model_id = "grounding_dino/default"
    gateway = FakeGateway(
        predictions={(model_id, "infer"): _det()},
        model_info={model_id: {"actions": {"infer": {}}}},
    )
    client = usage_client(gateway)

    response = client.post(
        "/grounding_dino/infer",
        json={"image": _image(), "api_key": "k", "text": ["cat"]},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id=model_id)
    entry = row["resource_details"]["models"][0]
    assert entry["model_id"] == model_id
    assert entry["model_architecture"] == "grounding-dino"
    assert entry["task_type"] == "open-vocabulary-object-detection"


def test_yolo_world_404_is_an_error_row(usage_client, usage_collector):
    client = usage_client(FakeGateway())

    response = client.post(
        "/yolo_world/infer", json={"image": _image(), "api_key": "k", "text": ["cat"]}
    )

    assert response.status_code == 404
    row = _only_row(usage_collector)
    assert row["resource_id"] == "unknown"
    assert (row["error_type"], row["error_status_code"]) == ("LegacyHTTPError", 404)


def test_lmm_path_route_records_the_path_model(
    usage_client, usage_collector, fake_stat
):
    fake_stat["smolvlm2/x"] = ("vlm", "prompt", "smolvlm2", "2.2b")
    gateway = FakeGateway(
        predictions={("smolvlm2/x", "prompt"): ["a cat"]},
        model_info={"smolvlm2/x": {"actions": {"prompt": {}}}},
    )
    client = usage_client(gateway)

    response = client.post(
        "/infer/lmm/smolvlm2/x",
        json={
            "model_id": "smolvlm2/x",
            "image": _image(),
            "prompt": "what is it?",
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="smolvlm2/x")
    entry = row["resource_details"]["models"][0]
    assert entry["model_id"] == "smolvlm2/x"
    assert entry["task_type"] == "vlm"
    assert entry["model_variant"] == "2.2b"


def test_lmm_model_id_mismatch_records_the_http_exception(
    usage_client, usage_collector, fake_stat
):
    fake_stat["smolvlm2/x"] = ("vlm", "prompt")
    client = usage_client(FakeGateway())

    response = client.post(
        "/infer/lmm/smolvlm2/x",
        json={"model_id": "other/1", "image": _image(), "prompt": "hi", "api_key": "k"},
    )

    assert response.status_code == 400
    row = _only_row(usage_collector)
    assert (row["error_type"], row["error_status_code"]) == ("HTTPException", 400)


def test_depth_estimation_records_the_default_model(usage_client, usage_collector):
    gateway = FakeGateway(
        predictions={
            ("depth-anything-v2/small", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={"depth-anything-v2/small": {"actions": {"infer": {}}}},
    )
    client = usage_client(gateway)

    response = client.post(
        "/infer/depth-estimation", json={"image": _image(), "api_key": "k"}
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="depth-anything-v2/small")
    entry = row["resource_details"]["models"][0]
    assert entry["model_architecture"] == "depth-anything-v2"
    assert entry["model_variant"] == "small"
    assert entry["task_type"] == "depth-estimation"


def test_sam2_embed_image_records_an_interactive_segmentation_invocation(
    usage_client, usage_collector
):
    from inference_model_manager.hash_namespacing import namespace_client_hash_id

    gateway = FakeGateway(
        predictions={
            ("sam2/hiera_large", "embed"): [
                SimpleNamespace(image_hash=namespace_client_hash_id("img-1", "k"))
            ]
        },
        model_info={"sam2/hiera_large": {"actions": {"embed": {}}}},
    )
    client = usage_client(gateway)

    response = client.post(
        "/sam2/embed_image",
        json={"image": _image(), "image_id": "img-1", "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    _assert_success_row(row, resource_id="sam2/hiera_large")
    entry = row["resource_details"]["models"][0]
    assert entry["model_architecture"] == "sam2"
    assert entry["model_variant"] == "hiera_large"
    assert entry["task_type"] == "interactive-instance-segmentation"
    assert "execution_mode" not in row["resource_details"]


def test_sam3_concept_segment_records_the_execution_mode(
    usage_client, usage_collector, monkeypatch
):
    monkeypatch.setattr(configuration, "SAM3_EXEC_MODE", "local")
    gateway = FakeGateway(
        predictions={
            ("sam3/sam3_final", "segment"): [
                SimpleNamespace(masks=np.zeros((0, 6, 8)), scores=np.zeros((0,)))
            ]
        },
        model_info={"sam3/sam3_final": {"actions": {"segment": {}}}},
    )
    client = usage_client(gateway)

    response = client.post(
        "/sam3/concept_segment?source=app&source_info=app-info",
        json={
            "image": _image(),
            "model_id": "sam3/sam3_final",
            "prompts": [{"text": "cat"}],
            "api_key": "k",
        },
    )

    row = _only_row(usage_collector)
    assert row["resource_id"] == "sam3/sam3_final"
    assert row["resource_details"]["execution_mode"] == "local"
    assert row["resource_details"]["source"] == "app"
    assert row["resource_details"]["source_info"] == "app-info"
    assert (row["error_type"] is None) == (response.status_code == 200)


@pytest.mark.parametrize(
    "path,flag",
    [
        ("/infer/action_recognition", "ACTION_RECOGNITION_ENABLED"),
        ("/sam3_3d/infer", "SAM3_3D_OBJECTS_ENABLED"),
        ("/owlv2/infer", "CORE_MODEL_OWLV2_ENABLED"),
    ],
)
def test_501_stubs_record_an_error_row(usage_collector, monkeypatch, path, flag):
    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr(configuration, flag, True)
    app = FastAPI()
    app.state.usage_collector = usage_collector
    include_legacy_routers(app)
    client = TestClient(app)

    response = client.post(f"{path}?api_key=k", json={})

    assert response.status_code == 501
    row = _only_row(usage_collector)
    assert row["resource_id"] == "unknown"
    assert row["api_key"] == "k"
    assert (row["error_type"], row["error_status_code"]) == ("LegacyHTTPError", 501)
    assert row["resource_details"]["models"] == []


STUB_ROUTES = [
    ("/infer/action_recognition", "ACTION_RECOGNITION_ENABLED"),
    ("/sam3_3d/infer", "SAM3_3D_OBJECTS_ENABLED"),
    ("/owlv2/infer", "CORE_MODEL_OWLV2_ENABLED"),
]


def _stub_client(usage_collector, monkeypatch, flag):
    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr(configuration, flag, True)
    app = FastAPI()
    app.state.usage_collector = usage_collector
    include_legacy_routers(app)

    return TestClient(app)


@pytest.mark.parametrize("path,flag", STUB_ROUTES)
def test_501_stubs_attribute_the_row_to_the_body_key_and_model(
    usage_collector, monkeypatch, path, flag
):
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", "env-key")
    client = _stub_client(usage_collector, monkeypatch, flag)

    response = client.post(
        path,
        json={
            "api_key": "body-key",
            "model_id": "ws/model/7",
            "source": "body-source",
            "source_info": "body-info",
        },
    )

    assert response.status_code == 501
    row = _only_row(usage_collector)
    assert row["api_key"] == "body-key"
    assert row["resource_id"] == "ws/model/7"
    assert row["resource_details"]["source"] == "body-source"
    assert row["resource_details"]["source_info"] == "body-info"
    assert row["roboflow_service_name"] == "body-info"
    assert (row["error_type"], row["error_status_code"]) == ("LegacyHTTPError", 501)


@pytest.mark.parametrize(
    "content",
    [b"not json", b"[1, 2]", b'"text"', b"", b'{"api_key": 5, "model_id": ["x"]}'],
)
def test_501_stub_with_an_unusable_body_gets_no_body_attribution(
    usage_collector, monkeypatch, content
):
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", "env-key")
    client = _stub_client(usage_collector, monkeypatch, "SAM3_3D_OBJECTS_ENABLED")

    response = client.post("/sam3_3d/infer", content=content)

    assert response.status_code == 501
    row = _only_row(usage_collector)
    assert row["api_key"] == "env-key"
    assert row["resource_id"] == "unknown"


def test_validation_failure_answers_422_and_records_no_row(
    usage_client, usage_collector, detection_gateway
):
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection", json={"api_key": "k", "image": _image()}
    )

    assert response.status_code == 422
    assert usage_collector.rows == []


def test_v2_request_records_nothing(
    usage_client, usage_collector, fake_stat, monkeypatch
):
    import inference_server.app as app_mod

    fake_stat["ds/1"] = OBJECT_DETECTION
    monkeypatch.setattr(
        app_mod, "validate_api_key", AsyncMock(return_value=(True, "ws-1"))
    )
    gateway = FakeGateway(
        predictions={("ds/1", None): _det()},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )
    client = usage_client(gateway)

    with patch(
        "inference_server.handlers.object_detection.output_serializer."
        "serialize_detections_compact",
        return_value={"detections": [{"cls": 0, "conf": 0.9}]},
    ):
        response = client.post(
            "/v2/models/infer?model_id=ds/1",
            content=_jpeg(),
            headers={
                "Authorization": "Bearer k",
                "Content-Type": "application/octet-stream",
            },
        )

    assert response.status_code == 200, response.text
    assert any(call[0] == "infer" for call in gateway.calls)
    assert usage_collector.rows == []


def test_request_without_a_collector_is_served_and_records_nothing(
    legacy_client, detection_gateway
):
    import inference_server.app as app_mod

    client = legacy_client(detection_gateway)
    assert app_mod.app.state.usage_collector is None

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text


def test_recording_failure_leaves_the_response_unchanged(
    usage_client, usage_collector, detection_gateway, caplog
):
    usage_collector.error = RuntimeError("collector broke")
    client = usage_client(detection_gateway)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.request_hook"):
        response = client.post(
            "/infer/object_detection",
            json={"model_id": "ds/1", "api_key": "k", "image": _image()},
        )

    assert response.status_code == 200, response.text
    assert response.json()["predictions"][0]["class"] == "cat"
    assert usage_collector.rows == []
    assert "Usage of the request was not recorded: RuntimeError" in caplog.text
    assert "collector broke" not in caplog.text


def test_unprintable_exception_keeps_its_response(
    usage_client, usage_collector, fake_stat, caplog
):
    class Unprintable(Exception):
        def __str__(self):
            raise ValueError("no text")

    fake_stat["ds/1"] = OBJECT_DETECTION
    gateway = FakeGateway(predictions={("ds/1", "infer"): _raising(Unprintable())})
    client = usage_client(gateway)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.request_hook"):
        response = client.post(
            "/infer/object_detection",
            json={"model_id": "ds/1", "api_key": "k", "image": _image()},
        )

    assert response.status_code == 500
    assert response.json() == {"message": "Internal error."}
    assert usage_collector.rows == []
    assert "Usage of the request was not recorded: ValueError" in caplog.text


def test_request_after_shutdown_is_served_and_records_nothing(
    usage_client, usage_collector, detection_gateway
):
    import inference_server.app as app_mod

    client = usage_client(detection_gateway)
    app_mod.app.state.usage_collector = None

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    assert usage_collector.rows == []


def test_rows_reach_a_real_collector_and_are_posted_only_at_shutdown(
    usage_client, detection_gateway, monkeypatch
):
    posts = []

    def _post(url, **kwargs):
        posts.append((url, kwargs["json"]))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr("inference_server.usage.delivery.requests.post", _post)
    collector = UsageCollector()
    monkeypatch.setattr(
        "inference_server.app._start_usage_collector", lambda: collector
    )
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    rows = collector._usage["k"]
    assert list(rows) == ["request:ds/1:billable=true:outcome=success"]
    row = rows["request:ds/1:billable=true:outcome=success"]
    assert row["processed_frames"] == 1
    assert row["category"] == "request"
    assert row["execution_duration"] > 0
    assert posts == []

    client.__exit__(None, None, None)

    assert len(posts) == 1
    assert posts[0][1][0]["resource_id"] == "ds/1"
    assert posts[0][1][0]["api_key"] == "k"
    assert collector.stop() is True


def _hook_app(usage_collector, handler):
    app = FastAPI()
    app.state.usage_collector = usage_collector
    app.post("/probe")(report_request_usage(handler))

    return TestClient(app, raise_server_exceptions=False)


def test_returned_4xx_response_is_an_error_row(usage_collector):
    async def handler(request: Request):
        return JSONResponse(status_code=404, content={"message": "no"})

    response = _hook_app(usage_collector, handler).post("/probe?api_key=k")

    assert response.status_code == 404
    row = _only_row(usage_collector)
    assert row["error_type"] == "HTTPResponseError"
    assert row["error_status_code"] == 404
    assert row["resource_details"]["error"] == (
        "HTTPResponseError404: response returned status 404"
    )


@pytest.mark.asyncio
async def test_base_exception_is_not_recorded(usage_collector):
    async def handler(request: Request):
        raise asyncio.CancelledError()

    scope = {
        "type": "http",
        "app": SimpleNamespace(state=SimpleNamespace(usage_collector=usage_collector)),
        "query_string": b"api_key=k",
        "headers": [],
    }

    with pytest.raises(asyncio.CancelledError):
        await report_request_usage(handler)(request=Request(scope))

    assert usage_collector.rows == []
    assert MODEL_INVOCATIONS.get() is None


def test_holder_is_reset_after_the_handler(usage_collector):
    seen = []

    async def handler(request: Request):
        seen.append(MODEL_INVOCATIONS.get())
        record_model_invocation({"model_id": "m", "frames": 1})
        return {"ok": True}

    client = _hook_app(usage_collector, handler)
    assert client.post("/probe?api_key=k").status_code == 200

    assert seen == [[{"model_id": "m", "frames": 1}]]
    assert MODEL_INVOCATIONS.get() is None
    assert _only_row(usage_collector)["resource_details"]["models"] == [
        {"model_id": "m", "frames": 1}
    ]


def test_record_model_invocation_without_a_holder_is_a_no_op():
    assert MODEL_INVOCATIONS.get() is None

    record_model_invocation({"model_id": "m", "frames": 1})

    assert MODEL_INVOCATIONS.get() is None


def test_repeated_invocations_of_one_model_are_combined():
    holder = []
    token = MODEL_INVOCATIONS.set(holder)
    try:
        record_model_invocation(
            {
                "model_id": "m",
                "frames": 1,
                "execution_duration": 0.5,
                "model_variant": "first",
            }
        )
        record_model_invocation(
            {
                "model_id": "m",
                "frames": 3,
                "execution_duration": 0.25,
                "model_variant": "second",
            }
        )
        record_model_invocation({"model_id": "other", "frames": 1})
    finally:
        MODEL_INVOCATIONS.reset(token)

    assert holder == [
        {
            "model_id": "m",
            "frames": 4,
            "execution_duration": 0.75,
            "model_variant": "second",
        },
        {"model_id": "other", "frames": 1},
    ]


@pytest.mark.parametrize(
    "serverless,floor_flag,raw,expected",
    [
        (False, None, 0.01, 0.01),
        (True, None, 0.01, 0.1),
        (True, True, 0.01, 0.1),
        (True, False, 0.01, 0.01),
        (True, True, 0.5, 0.5),
    ],
)
def test_execution_duration_floor(monkeypatch, serverless, floor_flag, raw, expected):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", serverless)
    token = None
    if floor_flag is not None:
        token = apply_duration_minimum.set(floor_flag)
    try:
        duration = request_hook._execution_duration(raw)
    finally:
        if token is not None:
            apply_duration_minimum.reset(token)

    assert duration == expected


def test_serverless_floor_applies_to_the_recorded_row(
    usage_client, usage_collector, detection_gateway, monkeypatch
):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    monkeypatch.setattr(request_hook, "time", SimpleNamespace(perf_counter=lambda: 1.0))
    client = usage_client(detection_gateway)

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    assert _only_row(usage_collector)["execution_duration"] == 0.1
