import base64
import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from PIL import Image

from inference_models.errors import UnauthorizedModelAccessError
from inference_server.workflows import host
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
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
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


def test_workflow_image_input_ignores_the_declared_type_like_legacy(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")

    response = legacy_client(_od_gateway()).post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "BASE64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    predictions = response.json()["outputs"][0]["predictions"]
    assert predictions["image"] == {"width": 8, "height": 6}


@pytest.mark.parametrize("declared_type", ["file", "base64"])
def test_workflow_image_input_never_reads_a_local_path_like_legacy(
    legacy_client, fake_stat, tmp_path, declared_type
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _od_gateway()
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(base64.b64decode(_jpeg_b64()))

    response = legacy_client(gateway).post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": declared_type, "value": str(image_path)}},
            "api_key": "k",
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["message"].endswith(
        "Detected runtime parameter `image` defined as `WorkflowImage` that is "
        "invalid. Failed on input validation. Details: NumPy image type is not "
        "supported in this configuration of `inference`."
    )
    assert [call for call in gateway.calls if call[0] == "infer"] == []


def _od_gateway():
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): detections},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )
    return gateway


def _registry_rows(client):
    return [
        (model["model_id"], model["request_aliases"], model["request_paths"])
        for model in client.get("/model/registry").json()["models"]
    ]


def test_run_workflow_records_the_step_model_under_the_request_path(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_od_gateway())

    response = client.post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    assert _registry_rows(client) == [("ds/1", [], ["/workflows/run"])]


def test_predefined_workflow_records_the_step_model_under_the_request_path(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: OD_WF,
    )
    client = legacy_client(_od_gateway())

    response = client.post(
        "/ws/workflows/wf",
        json={
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    assert _registry_rows(client) == [("ds/1", [], ["/ws/workflows/wf"])]


@pytest.mark.parametrize(
    "path,body",
    [
        ("/workflows/run", {"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}),
        ("/infer/workflows", {"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}),
        ("/ws/workflows/wf", {"inputs": {"x": 1}, "api_key": "k"}),
        ("/infer/workflows/ws/wf", {"inputs": {"x": 1}, "api_key": "k"}),
        ("/workflows/validate", PASSTHROUGH_WF),
    ],
)
def test_workflow_routes_hand_the_provider_their_scope_path(
    legacy_client, monkeypatch, path, body
):
    from inference_server.workflows import router as router_mod

    request_paths = []

    class _CapturingProvider(router_mod.GatewayModelsProvider):
        def __init__(self, bridge, api_key, request_path=None):
            request_paths.append(request_path)
            super().__init__(bridge, api_key, request_path)

    monkeypatch.setattr(router_mod, "GatewayModelsProvider", _CapturingProvider)
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: PASSTHROUGH_WF,
    )

    response = legacy_client(FakeGateway()).post(path, json=body)

    assert response.status_code == 200, response.text
    assert request_paths == [path]


@pytest.mark.parametrize(
    "path,body",
    [
        ("/workflows/run", {"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}),
        ("/infer/workflows", {"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}),
        ("/ws/workflows/wf", {"inputs": {"x": 1}, "api_key": "k"}),
        ("/infer/workflows/ws/wf", {"inputs": {"x": 1}, "api_key": "k"}),
    ],
)
@pytest.mark.parametrize(
    "flag,value,expect_background_tasks",
    [
        (None, False, True),
        ("LAMBDA", True, False),
        ("GCP_SERVERLESS", True, False),
    ],
)
def test_workflow_run_routes_defer_sinks_unless_serverless(
    legacy_client, monkeypatch, path, body, flag, value, expect_background_tasks
):
    from fastapi import BackgroundTasks

    from inference_server import configuration
    from inference_server.workflows import execution

    captured = []
    original = execution.build_init_parameters

    def _capturing(**kwargs):
        init_parameters = original(**kwargs)
        captured.append(init_parameters["workflows_core.background_tasks"])
        return init_parameters

    monkeypatch.setattr(execution, "build_init_parameters", _capturing)
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: PASSTHROUGH_WF,
    )
    monkeypatch.setattr(configuration, "LAMBDA", False)
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", False)
    if flag is not None:
        monkeypatch.setattr(configuration, flag, value)

    response = legacy_client(FakeGateway()).post(path, json=body)

    assert response.status_code == 200, response.text
    assert len(captured) == 1
    assert isinstance(captured[0], BackgroundTasks) is expect_background_tasks
    if not expect_background_tasks:
        assert captured[0] is None


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
    "error,status,error_type",
    [
        (PermissionError("bad key"), 500, "StepExecutionError"),
        (
            UnauthorizedModelAccessError("bad key"),
            401,
            "ClientCausedStepExecutionError",
        ),
        (LookupError("ds/1"), 500, "StepExecutionError"),
    ],
)
def test_step_error_status_follows_the_class_raised_in_the_step(
    legacy_client, fake_stat, error, status, error_type
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
    assert response.json()["error_type"] == error_type
    assert response.json()["inner_error_type"] == type(error).__name__
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


MISSING_API_KEY = (
    "Required Roboflow API key is missing. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/ws/workflows/wf/describe_interface", {}),
        ("/workflows/describe_interface", {"specification": PASSTHROUGH_WF}),
        ("/ws/workflows/wf/describe_workload", {}),
        ("/workflows/describe_workload", {"specification": PASSTHROUGH_WF}),
    ],
)
def test_describe_route_without_key_answers_like_legacy(legacy_client, path, payload):
    response = legacy_client(FakeGateway()).post(path, json=payload)

    assert response.status_code == 400
    assert response.json() == {"message": MISSING_API_KEY}


UNAUTHORIZED = (
    "Unauthorized access to roboflow API - check API key and make sure the key is "
    "valid for workspace you use. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
PAYMENT_REQUIRED = (
    "Not enough credits to perform this request. Verify your workspace billing page."
)
FORBIDDEN = (
    "Unauthorized access to roboflow API - check API key and make sure the key is "
    "valid and have required scopes. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
NOT_FOUND = (
    "Requested Roboflow resource not found. Make sure that workspace, project or "
    "model you referred in request exists."
)
USAGE_PAUSED = (
    "Roboflow API usage is paused. Please contact your workspace administrator to "
    "re-enable api keys."
)
REQUEST_FAILED = "Internal error. Request to Roboflow API failed."
INTERNAL_ERROR = "Internal error."


def _platform_response(status_code, content):
    response = requests.Response()
    response.status_code = status_code
    response._content = content

    return response


@pytest.mark.parametrize(
    "platform_status,platform_content,status,message",
    [
        (401, b'{"message": "platform text"}', 401, UNAUTHORIZED),
        (402, b'{"message": "platform text"}', 402, PAYMENT_REQUIRED),
        (403, b'{"message": "platform text"}', 403, FORBIDDEN),
        (404, b'{"message": "platform text"}', 404, NOT_FOUND),
        (423, b'{"message": "platform text"}', 423, USAGE_PAUSED),
        (400, b'{"message": "platform text"}', 502, REQUEST_FAILED),
        (429, b'{"message": "platform text"}', 502, REQUEST_FAILED),
        (500, b'{"message": "platform text"}', 502, REQUEST_FAILED),
        (503, b"", 502, REQUEST_FAILED),
        (504, b"", 502, REQUEST_FAILED),
        (200, b"not json", 502, REQUEST_FAILED),
        (200, b"{}", 502, REQUEST_FAILED),
        (200, b"[]", 502, REQUEST_FAILED),
        (200, b'"abc"', 502, REQUEST_FAILED),
        (200, b'"a workflow"', 500, INTERNAL_ERROR),
        (200, b"null", 500, INTERNAL_ERROR),
        (200, b"5", 500, INTERNAL_ERROR),
        (200, b'{"workflow": null}', 500, INTERNAL_ERROR),
        (200, b'{"workflow": 5}', 500, INTERNAL_ERROR),
        (200, b'{"workflow": []}', 502, REQUEST_FAILED),
        (200, b'{"workflow": "abc"}', 502, REQUEST_FAILED),
        (200, b'{"workflow": "a config"}', 502, REQUEST_FAILED),
        (200, b'{"workflow": ["config"]}', 502, REQUEST_FAILED),
        (200, b'{"workflow": {}}', 502, REQUEST_FAILED),
        (200, b'{"workflow": {"config": null}}', 502, REQUEST_FAILED),
        (200, b'{"workflow": {"config": 5}}', 502, REQUEST_FAILED),
        (200, b'{"workflow": {"config": "[]"}}', 502, REQUEST_FAILED),
        (200, b'{"workflow": {"config": "not json"}}', 502, REQUEST_FAILED),
        (200, b'{"workflow": {"config": "{}"}}', 502, REQUEST_FAILED),
        (
            200,
            b'{"workflow": {"config": "{\\"specification\\": 1}"}}',
            502,
            REQUEST_FAILED,
        ),
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        "/ws/workflows/wf",
        "/ws/workflows/wf/describe_interface",
        "/ws/workflows/wf/describe_workload",
    ],
)
def test_workflow_definition_fetch_failure_answers_like_legacy(
    legacy_client, monkeypatch, path, platform_status, platform_content, status, message
):
    monkeypatch.setattr(
        host,
        "_platform_request",
        lambda *args, **kwargs: _platform_response(platform_status, platform_content),
    )

    response = legacy_client(FakeGateway()).post(
        path, json={"api_key": "k", "use_cache": False, "inputs": {}}
    )

    assert response.status_code == status, response.text
    assert response.json() == {"message": message}


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


_CUSTOM_MODEL_RUN_FUNCTION = """
def infer(self, image: WorkflowImageData) -> BlockResult:
    model = self._init_results["model"]
    return {"predictions": sv.Detections.empty()}
"""

_CUSTOM_MODEL_INIT_FUNCTION = """
def init_model() -> Dict[str, Any]:
    return {"model": AutoModel}
"""


def _custom_model_workflow(imports):
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "CustomModel",
                    "inputs": {
                        "image": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_image"],
                        },
                    },
                    "outputs": {
                        "predictions": {
                            "type": "DynamicOutputDefinition",
                            "kind": ["object_detection_prediction"],
                        }
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": _CUSTOM_MODEL_RUN_FUNCTION,
                    "run_function_name": "infer",
                    "init_function_code": _CUSTOM_MODEL_INIT_FUNCTION,
                    "init_function_name": "init_model",
                    "imports": imports,
                },
            },
        ],
        "steps": [
            {"type": "CustomModel", "name": "model", "image": "$inputs.image"},
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            },
        ],
    }


def test_validate_custom_python_block_with_inference_models_import(legacy_client):
    assert host.SERVER_WORKFLOWS_CONFIGURATION.engine.allow_custom_python_execution

    workflow = _custom_model_workflow(
        imports=["from inference_models.models.auto_loaders.core import AutoModel"]
    )

    response = legacy_client(FakeGateway()).post("/workflows/validate", json=workflow)

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}


def test_validate_custom_python_block_rejects_legacy_inference_import(
    legacy_client, monkeypatch
):
    assert host.SERVER_WORKFLOWS_CONFIGURATION.engine.allow_custom_python_execution
    monkeypatch.setitem(sys.modules, "inference", None)

    workflow = _custom_model_workflow(
        imports=["from inference.models.yolov8 import YOLOv8ObjectDetection"]
    )

    response = legacy_client(FakeGateway()).post("/workflows/validate", json=workflow)

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error_type"] == "DynamicBlockCodeError"
    assert "ModuleNotFoundError" in body["message"]


_SIMPLE_DYNAMIC_BLOCK = {
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
_DYNAMIC_BLOCK_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "dynamic_blocks_definitions": [_SIMPLE_DYNAMIC_BLOCK],
    "steps": [{"type": "Echo", "name": "echo", "value": "$inputs.x"}],
    "outputs": [{"type": "JsonField", "name": "y", "selector": "$steps.echo.value"}],
}


@pytest.fixture
def modal_custom_python(monkeypatch):
    from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
        block_scaffolding,
    )

    monkeypatch.setattr(
        block_scaffolding, "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "modal"
    )
    host.clear_workspace_cache()
    yield
    host.clear_workspace_cache()


def _refused_key_requests(client):
    describe = client.post(
        "/workflows/blocks/describe",
        json={"dynamic_blocks_definitions": [_SIMPLE_DYNAMIC_BLOCK], "api_key": "bad"},
    )
    validate = client.post("/workflows/validate?api_key=bad", json=_DYNAMIC_BLOCK_WF)
    run = client.post(
        "/workflows/run",
        json={"specification": _DYNAMIC_BLOCK_WF, "inputs": {"x": 1}, "api_key": "bad"},
    )

    return [describe, validate, run]


@pytest.mark.parametrize(
    "platform_status,status,message",
    [
        (401, 401, UNAUTHORIZED),
        (402, 402, PAYMENT_REQUIRED),
        (403, 403, FORBIDDEN),
        (404, 404, NOT_FOUND),
        (423, 423, USAGE_PAUSED),
        (500, 502, REQUEST_FAILED),
    ],
)
def test_dynamic_block_compilation_with_a_refused_key_answers_like_legacy(
    legacy_client, modal_custom_python, platform_status, status, message
):
    import requests_mock

    client = legacy_client(FakeGateway())

    with requests_mock.Mocker() as mocker:
        mocker.get(
            "https://api.roboflow.com/?api_key=bad&nocache=true",
            status_code=platform_status,
        )
        responses = _refused_key_requests(client)

    for response in responses:
        assert response.status_code == status, response.text
        assert response.json() == {"message": message}
        assert "retry-after" not in response.headers
