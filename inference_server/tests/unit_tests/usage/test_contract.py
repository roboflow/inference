import base64
import importlib
import json
import re
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from inference_sdk.config import INTERNAL_REMOTE_EXEC_REQ_HEADER
from inference_server import configuration
from inference_server.hosted import serverless_auth
from inference_server.hosted.serverless_auth import AuthorizationCacheEntry
from inference_server.usage import delivery
from inference_server.usage.collector import UsageCollector
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.usage.test_request_hook import _image, _jpeg
from tests.unit_tests.usage.test_workflow_usage import (
    _detection_gateway,
    _detection_workflow,
    _jpeg_b64,
)

POST_TARGET = "inference_server.usage.delivery.requests.post"
COLLECTOR_BASE_URL = "https://collector.example.com"
SECRET = "internal-secret-1"
SYSTEM_INFO = {
    "hostname": "host1",
    "ip_address_hash": "ab12c",
    "is_gpu_available": False,
}
ROW_KEYS = {
    "api_key",
    "category",
    "processed_frames",
    "timestamp_start",
    "timestamp_stop",
    "execution_duration",
    "inference_version",
    "inference_models_version",
    "inference_model_manager_version",
    "streamvision_version",
    "roboflow_workflows_version",
    "hosted",
    "enterprise",
    "exec_session_id",
    "hostname",
    "ip_address_hash",
    "python_version",
    "is_gpu_available",
    "resource_id",
    "resource_details",
    "fps",
    "megapixel_buckets",
    "source_duration",
}
COMPONENT_VERSION_KEYS = (
    "inference_models_version",
    "inference_model_manager_version",
    "roboflow_workflows_version",
    "streamvision_version",
)
MODEL_ENTRY_KEYS = {
    "model_architecture",
    "model_variant",
    "task_type",
    "model_input_height",
    "model_input_width",
    "execution_duration",
    "frames",
}


class PostRecorder:
    def __init__(self):
        self.calls = []
        self.late_attempts = 0
        self.collector_stopped = False

    def __call__(self, url, **kwargs):
        self.calls.append(SimpleNamespace(url=url, **kwargs))

        return SimpleNamespace(status_code=200)

    def unreachable(self, url, **kwargs):
        self.late_attempts += 1
        raise AssertionError("usage post attempted after the collector was stopped")

    def only_row(self):
        assert len(self.calls) == 1
        body = self.calls[0].json
        assert isinstance(body, list)
        assert len(body) == 1

        return body[0]


@pytest.fixture
def posts():
    recorder = PostRecorder()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(POST_TARGET, recorder)
        patch.setattr(configuration, "METRICS_COLLECTOR_BASE_URL", COLLECTOR_BASE_URL)
        patch.setattr(
            configuration,
            "TELEMETRY_API_USAGE_ENDPOINT_URL",
            f"{COLLECTOR_BASE_URL}/usage/inference",
        )

        yield recorder

        calls_before_seal = len(recorder.calls)
        patch.setattr(POST_TARGET, recorder.unreachable)
        assert recorder.collector_stopped
        assert recorder.late_attempts == 0
        assert len(recorder.calls) == calls_before_seal


@pytest.fixture
def real_collector(request, posts):
    usage_collector = UsageCollector()
    usage_collector._system_info = dict(SYSTEM_INFO)
    usage_collector.start()

    def _drain():
        assert delivery.requests.post is posts
        stopped = usage_collector.stop(timeout=5)
        assert delivery.requests.post is posts
        posts.collector_stopped = True
        assert stopped is True

    request.addfinalizer(_drain)

    return usage_collector


@pytest.fixture
def contract_client(legacy_client, real_collector, posts, monkeypatch):
    monkeypatch.setattr(
        "inference_server.app._start_usage_collector", lambda: real_collector
    )

    return legacy_client


@pytest.fixture
def serverless_contract_client(contract_client, real_collector, posts):
    import inference_server.app as app_mod

    async def _authorized(http_request, api_key):
        return None, AuthorizationCacheEntry(expires_at=0.0, workspace_id="ws"), False

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(configuration, "GCP_SERVERLESS", True)
        patch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", SECRET)
        importlib.reload(app_mod)
        patch.setattr(
            "inference_server.app._start_usage_collector", lambda: real_collector
        )
        patch.setattr(serverless_auth, "_authorize", _authorized)

        yield contract_client

    importlib.reload(app_mod)


def _detection():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


@pytest.fixture
def detection_gateway(fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer", "yolov8", "yolov8-n")

    return FakeGateway(
        predictions={("ds/1", "infer"): _detection()},
        model_info={
            "ds/1": {
                "class_names": ["cat"],
                "actions": {"infer": {}},
                "input_height": 640,
                "input_width": 640,
            }
        },
    )


def _infer(client, **headers):
    return client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "key-1", "image": _image()},
        headers=headers,
    )


def _infer_catch_all(client, **headers):
    return client.post(
        "/ds/1?api_key=key-1",
        content=base64.b64encode(_jpeg()),
        headers={"Content-Type": "application/x-www-form-urlencoded", **headers},
    )


def _flush_and_stop(collector):
    collector.flush()
    assert collector.stop(timeout=5) is True


def test_object_detection_request_posts_the_row_the_platform_expects(
    contract_client, real_collector, posts, detection_gateway
):
    client = contract_client(detection_gateway)

    before = time.time_ns()
    response = _infer(client)
    after = time.time_ns()
    _flush_and_stop(real_collector)

    assert response.status_code == 200, response.text
    call = posts.calls[0]
    assert call.url == f"{COLLECTOR_BASE_URL}/usage/inference"
    assert call.headers == {
        "Authorization": "Bearer key-1",
        "X-Roboflow-Inference-Version": configuration.SERVER_VERSION,
        "X-Allow-Chunked": "true",
    }
    assert call.timeout == 1
    row = posts.only_row()
    assert set(row) == ROW_KEYS
    assert row["api_key"] == "key-1"
    assert row["category"] == "request"
    assert type(row["processed_frames"]) is int
    assert row["processed_frames"] == 1
    for key in ("timestamp_start", "timestamp_stop"):
        assert type(row[key]) is int
        assert before <= row[key] <= after
    assert row["timestamp_start"] <= row["timestamp_stop"]
    assert type(row["execution_duration"]) is float
    assert row["execution_duration"] > 0
    assert type(row["fps"]) is float
    assert row["fps"] == 0.0
    assert type(row["source_duration"]) is int
    assert row["source_duration"] == 0
    assert row["megapixel_buckets"] == {}
    assert row["hosted"] is False
    assert row["enterprise"] is False
    assert row["is_gpu_available"] is False
    assert type(row["python_version"]) is str
    assert row["hostname"] == SYSTEM_INFO["hostname"]
    assert row["ip_address_hash"] == SYSTEM_INFO["ip_address_hash"]
    assert row["inference_version"] == configuration.SERVER_VERSION
    for key in COMPONENT_VERSION_KEYS:
        assert row[key] is None or (type(row[key]) is str and row[key] != "")
    assert re.match(r"^\d+_[0-9a-f]{4}$", row["exec_session_id"])
    assert row["resource_id"] == "ds/1"
    assert "api_key_hash" not in row
    assert isinstance(row["resource_details"], str)
    details = json.loads(row["resource_details"])
    assert isinstance(details, dict)
    assert details["billable"] is True
    assert len(details["models"]) == 1
    entry = details["models"][0]
    assert set(entry) == MODEL_ENTRY_KEYS | {"model_id"}
    assert entry["model_id"] == "ds/1"
    assert "model_latency_ms" not in entry
    assert entry["model_architecture"] == "yolov8"
    assert entry["model_variant"] == "yolov8-n"
    assert entry["task_type"] == "object-detection"
    assert entry["model_input_height"] == 640
    assert entry["model_input_width"] == 640
    assert type(entry["frames"]) is int
    assert entry["frames"] == 1
    assert type(entry["execution_duration"]) is float
    assert entry["execution_duration"] > 0


def test_endpoint_is_routed_through_the_secure_gateway_when_configured(
    contract_client, real_collector, posts, detection_gateway, monkeypatch
):
    monkeypatch.setattr(
        "inference_models.weights_providers.roboflow.SECURE_GATEWAY",
        "https://gateway.local",
    )
    client = contract_client(detection_gateway)

    _infer(client)
    _flush_and_stop(real_collector)

    parsed = urlparse(posts.calls[0].url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://gateway.local/proxy"
    )
    assert parse_qs(parsed.query)["url"] == [f"{COLLECTOR_BASE_URL}/usage/inference"]
    posts.only_row()


def test_row_carries_the_execution_id_of_the_request_under_gcp_serverless(
    serverless_contract_client, real_collector, posts, detection_gateway
):
    client = serverless_contract_client(detection_gateway)

    response = _infer_catch_all(client, execution_id="exec-77")
    _flush_and_stop(real_collector)

    assert response.status_code == 200, response.text
    assert response.headers["execution_id"] == "exec-77"
    assert posts.only_row()["exec_session_id"] == "exec-77"


def test_row_carries_a_generated_execution_id_when_the_request_has_none(
    serverless_contract_client, real_collector, posts, detection_gateway
):
    client = serverless_contract_client(detection_gateway)

    response = _infer_catch_all(client)
    _flush_and_stop(real_collector)

    generated = response.headers["execution_id"]
    assert re.match(r"^\d+_[0-9a-f]{4}$", generated)
    assert posts.only_row()["exec_session_id"] == generated


def test_verified_internal_request_is_not_floored_and_a_plain_one_is(
    serverless_contract_client, real_collector, posts, detection_gateway
):
    client = serverless_contract_client(detection_gateway)

    internal = _infer_catch_all(
        client,
        execution_id="exec-internal",
        **{INTERNAL_REMOTE_EXEC_REQ_HEADER: SECRET},
    )
    real_collector.flush()
    plain = _infer_catch_all(client, execution_id="exec-plain")
    _flush_and_stop(real_collector)

    assert internal.status_code == plain.status_code == 200
    rows = {row["exec_session_id"]: row for call in posts.calls for row in call.json}
    assert rows["exec-internal"]["execution_duration"] < 0.1
    assert rows["exec-plain"]["execution_duration"] >= 0.1


def test_workflow_run_posts_one_row_for_the_workflow_with_its_model(
    contract_client, real_collector, posts, fake_stat
):
    client = contract_client(_detection_gateway(fake_stat, "ds/1"))

    response = client.post(
        "/workflows/run",
        json={
            "specification": _detection_workflow("ds/1"),
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "key-1",
            "workflow_id": "wf-1",
        },
    )
    _flush_and_stop(real_collector)

    assert response.status_code == 200, response.text
    row = posts.only_row()
    assert row["resource_id"] == "wf-1"
    assert row["category"] == "request"
    assert row["processed_frames"] == 1
    details = json.loads(row["resource_details"])
    assert len(details["models"]) == 1
    assert details["models"][0]["model_architecture"] == "yolov8"


def _echo_block():
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": "Echo",
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            "outputs": {"value": {"type": "DynamicOutputDefinition", "kind": []}},
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": "def run(self, value):\n    return {'value': value}\n",
            "run_function_name": "run",
        },
    }


def _model_and_custom_python_workflow():
    specification = _detection_workflow("ds/1")
    specification["inputs"].append({"type": "WorkflowParameter", "name": "x"})
    specification["dynamic_blocks_definitions"] = [_echo_block()]
    specification["steps"].append(
        {"type": "Echo", "name": "echo", "value": "$inputs.x"}
    )
    specification["outputs"].append(
        {"type": "JsonField", "name": "echoed", "selector": "$steps.echo.value"}
    )

    return specification


def test_custom_python_duration_is_floored_for_a_plain_caller_only(
    serverless_contract_client, real_collector, posts, fake_stat, monkeypatch
):
    monkeypatch.setattr(
        "inference_server.usage.observer.consume_block_duration",
        lambda: SimpleNamespace(duration=0.02),
    )
    client = serverless_contract_client(_detection_gateway(fake_stat, "ds/1"))

    def _run(**headers):
        return client.post(
            "/workflows/run",
            json={
                "specification": _model_and_custom_python_workflow(),
                "inputs": {
                    "image": {"type": "base64", "value": _jpeg_b64()},
                    "x": 3,
                },
                "api_key": "key-1",
                "workflow_id": "wf-1",
            },
            headers=headers,
        )

    internal = _run(
        execution_id="exec-internal", **{INTERNAL_REMOTE_EXEC_REQ_HEADER: SECRET}
    )
    real_collector.flush()
    plain = _run(execution_id="exec-plain")
    _flush_and_stop(real_collector)

    assert internal.status_code == plain.status_code == 200, plain.text
    assert len(posts.calls) == 2
    rows = {row["exec_session_id"]: row for call in posts.calls for row in call.json}
    assert set(rows) == {"exec-internal", "exec-plain"}
    custom_python = {
        key: json.loads(row["resource_details"])["custom_python"]
        for key, row in rows.items()
    }
    assert [
        entry["execution_duration"] for entry in custom_python["exec-internal"]
    ] == [0.02]
    assert [entry["execution_duration"] for entry in custom_python["exec-plain"]] == [
        0.1
    ]
    assert rows["exec-internal"]["execution_duration"] < 0.1
    assert rows["exec-plain"]["execution_duration"] >= 0.1
