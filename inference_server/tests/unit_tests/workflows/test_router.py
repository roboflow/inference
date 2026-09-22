import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from inference_models.errors import UnauthorizedModelAccessError
from tests.unit_tests.legacy.conftest import FakeGateway, route_paths


def _jpeg_b64():
    buf = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


PASSTHROUGH_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "y", "selector": "$inputs.x"}],
}

OD_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "det",
            "image": "$inputs.image",
            "model_id": "ds/1",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.det.predictions",
        }
    ],
}


def test_run_workflow_without_models(legacy_client):
    client = legacy_client(FakeGateway())

    response = client.post(
        "/workflows/run",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["outputs"] == [{"y": 3}]


def test_run_workflow_with_object_detection_block(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): detections},
        model_info={"ds/1": {"class_names": ["cat"], "tasks": {"infer": {}}}},
    )

    response = legacy_client(gateway).post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    predictions = response.json()["outputs"][0]["predictions"]
    assert predictions["predictions"][0]["class"] == "cat"
    assert predictions["image"] == {"width": 8, "height": 6}


def test_two_segment_workflow_paths_beat_catch_all(legacy_client):
    client = legacy_client(FakeGateway())

    run = client.post(
        "/workflows/run", json={"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}
    )
    deprecated = client.post(
        "/infer/workflows", json={"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}
    )

    assert run.status_code == 200, run.text
    assert run.json()["outputs"] == [{"y": 1}]
    assert deprecated.status_code == 200, deprecated.text
    assert deprecated.json()["outputs"] == [{"y": 1}]


def test_syntax_error_is_400_with_workflow_error_shape(legacy_client):
    response = legacy_client(FakeGateway()).post(
        "/workflows/run",
        json={
            "specification": {
                "version": "1.0",
                "inputs": [],
                "steps": [{"type": "nope"}],
                "outputs": [],
            },
            "inputs": {},
        },
    )

    assert response.status_code == 400
    assert {"message", "error_type", "context"} <= set(response.json())


@pytest.mark.parametrize(
    "error,status",
    [
        (PermissionError("bad key"), 401),
        (UnauthorizedModelAccessError("bad key"), 401),
        (LookupError("ds/1"), 404),
    ],
)
def test_step_error_status_from_client_caused_error(
    legacy_client, fake_stat, error, status
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = FakeGateway(model_info={"ds/1": {"class_names": ["cat"]}})

    async def _boom(**kwargs):
        raise error

    gateway.infer = _boom

    response = legacy_client(gateway).post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == status, response.text
    assert response.json()["blocks_errors"][0]["block_id"] == "det"


def test_describe_interface_keeps_null_fields_like_legacy(legacy_client):
    response = legacy_client(FakeGateway()).post(
        "/workflows/describe_interface",
        json={"specification": PASSTHROUGH_WF, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    assert set(response.json()) == {
        "inputs",
        "outputs",
        "typing_hints",
        "kinds_schemas",
    }


def test_describe_interface_requires_key(legacy_client):
    response = legacy_client(FakeGateway()).post(
        "/workflows/describe_interface", json={"specification": PASSTHROUGH_WF}
    )

    assert response.status_code == 400
    assert "API key is missing" in response.json()["message"]


def test_schema_route_gzips_when_requested(legacy_client):
    response = legacy_client(FakeGateway()).get(
        "/workflows/definition/schema", headers={"Accept-Encoding": "gzip"}
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-encoding"] == "gzip"
    assert "schema" in response.json()


def test_blocks_describe_gzip(legacy_client):
    response = legacy_client(FakeGateway()).get(
        "/workflows/blocks/describe", headers={"Accept-Encoding": "gzip"}
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-encoding"] == "gzip"
    assert "blocks" in response.json()


def test_blocks_describe_post_without_body(legacy_client):
    response = legacy_client(FakeGateway()).post("/workflows/blocks/describe")

    assert response.status_code == 200, response.text
    assert "blocks" in response.json()


def test_versions_schema_and_validate(legacy_client):
    client = legacy_client(FakeGateway())

    assert client.get("/workflows/execution_engine/versions").json()["versions"]
    assert "schema" in client.get("/workflows/definition/schema").json()
    assert client.post("/workflows/validate", json=PASSTHROUGH_WF).json() == {
        "status": "ok"
    }


def test_dynamic_outputs_route(legacy_client):
    response = legacy_client(FakeGateway()).post(
        "/workflows/blocks/dynamic_outputs",
        json={
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "det",
            "image": "$inputs.image",
            "model_id": "ds/1",
        },
    )

    assert response.status_code == 200, response.text
    assert "predictions" in [output["name"] for output in response.json()]


def test_predefined_workflow_route(legacy_client, monkeypatch):
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: PASSTHROUGH_WF,
    )

    response = legacy_client(FakeGateway()).post(
        "/ws/workflows/wf", json={"inputs": {"x": 1}, "api_key": "k"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["outputs"] == [{"y": 1}]


def test_predefined_workflow_describe_interface_route(legacy_client, monkeypatch):
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: PASSTHROUGH_WF,
    )

    response = legacy_client(FakeGateway()).post(
        "/ws/workflows/wf/describe_interface", json={"api_key": "k"}
    )

    assert response.status_code == 200, response.text
    assert set(response.json()) == {
        "inputs",
        "outputs",
        "typing_hints",
        "kinds_schemas",
    }


def _reloaded_app(monkeypatch, **overrides):
    import importlib

    import inference_server.app as app_mod
    from inference_server import configuration

    for name, value in overrides.items():
        monkeypatch.setattr(configuration, name, value)
    return importlib.reload(app_mod)


def test_body_limit_guards_workflows_when_legacy_routes_disabled(monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    import inference_server.app as app_mod

    try:
        module = _reloaded_app(
            monkeypatch, LEGACY_ROUTES_ENABLED=False, MAX_BODY_BYTES=32
        )
        assert "/workflows/run" in route_paths(module.app)
        client = TestClient(module.app, raise_server_exceptions=False)

        declared = client.post(
            "/workflows/run",
            content=b"x" * 64,
            headers={"Content-Type": "application/json"},
        )
        streamed = client.post(
            "/workflows/run",
            content=iter([b"x" * 20, b"x" * 20]),
            headers={
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
            },
        )

        assert declared.status_code == 413
        assert declared.json() == {"message": "Request payload too large."}
        assert streamed.status_code == 413
        assert streamed.json() == {"message": "Request payload too large."}
    finally:
        monkeypatch.undo()
        importlib.reload(app_mod)
