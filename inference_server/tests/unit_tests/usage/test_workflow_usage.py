import asyncio
import base64
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from roboflow_workflows.execution_engine.entities.base import Batch
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    record_block_duration,
)

from inference_sdk.config import apply_duration_minimum
from inference_server import configuration
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    SyncLegacyBridge,
)
from inference_server.usage.observer import UsageExecutionObserver
from inference_server.usage.rows import (
    USAGE_SCOPE,
    UsageScope,
    bound_scope,
    steps_resource_id,
    workflow_steps,
)
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.usage.conftest import FakeUsageCollector

SECRET = "service-secret-1"

PASSTHROUGH_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "y", "selector": "$inputs.x"}],
}
PASSTHROUGH_WF_SHA = (
    "sha:efb29512bd5060e5ca82ef117fcc99bf71d2a82b4a3d298c9ecdec0771377545"
)
EMPTY_STEPS_HASH = "07562"

PREDEFINED_PATHS = ["/ws/workflows/wf", "/infer/workflows/ws/wf"]
SPECIFICATION_PATHS = ["/workflows/run", "/infer/workflows"]


def _jpeg_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()


def _detection_step(name, model_id):
    return {
        "type": "roboflow_core/roboflow_object_detection_model@v2",
        "name": name,
        "image": "$inputs.image",
        "model_id": model_id,
    }


def _detection_workflow(*model_ids):
    steps = [
        _detection_step(f"det{index}", model_id)
        for index, model_id in enumerate(model_ids)
    ]
    outputs = [
        {
            "type": "JsonField",
            "name": step["name"],
            "selector": f"$steps.{step['name']}.predictions",
        }
        for step in steps
    ]

    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": outputs,
    }


def _detection_gateway(fake_stat, *model_ids):
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    for model_id in model_ids:
        fake_stat[model_id] = ("object-detection", "infer", "yolov8", "yolov8-n")
    gateway = FakeGateway(
        predictions={(model_id, "infer"): detections for model_id in model_ids},
        model_info={
            model_id: {"class_names": ["cat"], "actions": {"infer": {}}}
            for model_id in model_ids
        },
    )

    return gateway


def _categories(usage_collector):
    return [row["category"] for row in usage_collector.rows]


def _rows_of(usage_collector, category):
    return [row for row in usage_collector.rows if row["category"] == category]


def _only_row_of(usage_collector, category):
    rows = _rows_of(usage_collector, category)
    assert len(rows) == 1

    return rows[0]


@pytest.fixture
def predefined_specification(monkeypatch):
    holder = {"specification": PASSTHROUGH_WF}
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: holder["specification"],
    )

    return holder


@pytest.mark.parametrize("path", PREDEFINED_PATHS)
def test_predefined_route_records_a_request_row_and_a_workflows_row(
    usage_client, usage_collector, predefined_specification, path
):
    client = usage_client(FakeGateway())

    response = client.post(path, json={"inputs": {"x": 3}, "api_key": "k"})

    assert response.status_code == 200, response.text
    assert _categories(usage_collector) == ["workflows", "request"]
    row = _only_row_of(usage_collector, "request")
    assert row["api_key"] == "k"
    assert row["resource_id"] == "wf"
    assert row["frames"] == 1
    assert row["billable"] is True
    assert row["is_preview"] is False
    assert row["error_type"] is None
    assert row["execution_duration"] > 0
    assert row["resource_details"] == {
        "steps": [],
        "is_preview": False,
        "workspace_id": "ws",
    }
    workflow = _only_row_of(usage_collector, "workflows")
    assert workflow == {
        "api_key": "k",
        "category": "workflows",
        "resource_id": "wf",
        "resource_details": {"steps": [], "is_preview": False},
        "frames": 1,
        "execution_duration": workflow["execution_duration"],
        "fps": 0.0,
        "source_duration": 0.0,
        "billable": True,
        "is_preview": False,
        "error_type": None,
        "error_status_code": None,
        "roboflow_service_name": None,
        "roboflow_internal_secret": None,
        "megapixel_buckets": None,
    }
    assert 0 < workflow["execution_duration"] <= row["execution_duration"]


@pytest.mark.parametrize("path", SPECIFICATION_PATHS)
def test_specification_route_records_a_request_row_and_a_workflows_row(
    usage_client, usage_collector, path
):
    client = usage_client(FakeGateway())

    response = client.post(
        path,
        json={
            "specification": PASSTHROUGH_WF,
            "inputs": {"x": 3},
            "api_key": "k",
            "workflow_id": "body-wf",
        },
    )

    assert response.status_code == 200, response.text
    assert _categories(usage_collector) == ["workflows", "request"]
    row = _only_row_of(usage_collector, "request")
    assert row["api_key"] == "k"
    assert row["resource_id"] == "body-wf"
    assert row["frames"] == 1
    assert row["billable"] is True
    assert row["resource_details"] == {"steps": [], "is_preview": False}
    workflow = _only_row_of(usage_collector, "workflows")
    assert workflow["resource_id"] == "body-wf"
    assert workflow["api_key"] == "k"


def test_path_workflow_id_beats_the_body_workflow_id(
    usage_client, usage_collector, predefined_specification
):
    client = usage_client(FakeGateway())

    response = client.post(
        "/ws/workflows/wf",
        json={"inputs": {"x": 3}, "api_key": "k", "workflow_id": "other"},
    )

    assert response.status_code == 200, response.text
    assert _only_row_of(usage_collector, "request")["resource_id"] == "wf"
    assert _only_row_of(usage_collector, "workflows")["resource_id"] == "other"


def test_resource_id_fallbacks_hash_the_specification_and_the_step_list(
    usage_client, usage_collector
):
    client = usage_client(FakeGateway())

    response = client.post(
        "/workflows/run",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    assert _only_row_of(usage_collector, "request")["resource_id"] == (
        PASSTHROUGH_WF_SHA
    )
    workflow = _only_row_of(usage_collector, "workflows")
    assert workflow["resource_id"] == EMPTY_STEPS_HASH
    assert workflow["resource_id"] == steps_resource_id([])


def test_internal_workflow_id_of_the_specification_attributes_the_workflows_row(
    usage_client, usage_collector
):
    client = usage_client(FakeGateway())

    response = client.post(
        "/workflows/run",
        json={
            "specification": {**PASSTHROUGH_WF, "id": "internal-123"},
            "inputs": {"x": 3},
            "api_key": "k",
            "workflow_id": "body-wf",
        },
    )

    assert response.status_code == 200, response.text
    assert _only_row_of(usage_collector, "request")["resource_id"] == "body-wf"
    assert _only_row_of(usage_collector, "workflows")["resource_id"] == ("internal-123")


def test_steps_follow_the_legacy_format(usage_client, usage_collector, fake_stat):
    client = usage_client(_detection_gateway(fake_stat, "ds/1", "ds/2"))

    response = client.post(
        "/workflows/run",
        json={
            "specification": _detection_workflow("ds/1", "ds/2"),
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    steps = [
        "roboflow_core/roboflow_object_detection_model@v2:det0",
        "roboflow_core/roboflow_object_detection_model@v2:det1",
    ]
    assert _only_row_of(usage_collector, "request")["resource_details"]["steps"] == (
        steps
    )
    workflow = _only_row_of(usage_collector, "workflows")
    assert workflow["resource_details"]["steps"] == steps
    assert workflow["resource_id"] == steps_resource_id(steps)


def test_steps_of_a_predefined_workflow_come_from_the_fetched_specification(
    usage_client, usage_collector, fake_stat, predefined_specification
):
    predefined_specification["specification"] = _detection_workflow("ds/1")
    client = usage_client(_detection_gateway(fake_stat, "ds/1"))

    response = client.post(
        "/ws/workflows/wf",
        json={
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    for category in ("request", "workflows"):
        assert _only_row_of(usage_collector, category)["resource_details"]["steps"] == [
            "roboflow_core/roboflow_object_detection_model@v2:det0"
        ]


def test_local_model_steps_record_a_model_row_each_attributed_to_the_workflow_key(
    usage_client, usage_collector, fake_stat
):
    client = usage_client(_detection_gateway(fake_stat, "ds/1", "ds/2"))

    response = client.post(
        "/workflows/run?source=app&source_info=app-1",
        json={
            "specification": _detection_workflow("ds/1", "ds/2"),
            "inputs": {"image": [{"type": "base64", "value": _jpeg_b64()}] * 3},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    assert sorted(_categories(usage_collector)) == [
        "model",
        "model",
        "request",
        "workflows",
    ]
    models = sorted(_rows_of(usage_collector, "model"), key=lambda r: r["resource_id"])
    assert [model["resource_id"] for model in models] == ["ds/1", "ds/2"]
    for model in models:
        assert model["api_key"] == "k"
        assert model["frames"] == 3
        assert model["billable"] is True
        assert model["resource_details"] == {
            "model_architecture": "yolov8",
            "model_variant": "yolov8-n",
            "task_type": "object-detection",
            "source": "app",
            "source_info": "app-1",
        }
        assert model["megapixel_buckets"] == {
            "0-0.25": {
                "processed_frames": 3,
                "execution_duration": pytest.approx(model["execution_duration"]),
            }
        }
    workflow = _only_row_of(usage_collector, "workflows")
    assert workflow["resource_details"]["source_info"] == "app-1"
    assert "source" not in workflow["resource_details"]
    assert workflow["roboflow_service_name"] == "app-1"


@pytest.mark.parametrize("is_preview", [True, False])
def test_preview_flag_is_recorded(usage_client, usage_collector, is_preview):
    client = usage_client(FakeGateway())

    response = client.post(
        "/workflows/run",
        json={
            "specification": PASSTHROUGH_WF,
            "inputs": {"x": 3},
            "api_key": "k",
            "is_preview": is_preview,
        },
    )

    assert response.status_code == 200, response.text
    for category in ("request", "workflows"):
        row = _only_row_of(usage_collector, category)
        assert row["is_preview"] is is_preview
        assert row["resource_details"]["is_preview"] is is_preview
        assert row["billable"] is True


def test_failing_step_records_an_error_row_per_category(
    usage_client, usage_collector, fake_stat
):
    gateway = _detection_gateway(fake_stat, "ds/1")

    async def _boom(**kwargs):
        raise RuntimeError("step broke")

    gateway.infer = _boom
    client = usage_client(gateway)

    response = client.post(
        "/workflows/run",
        json={
            "specification": _detection_workflow("ds/1"),
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    assert response.status_code == 500, response.text
    assert _categories(usage_collector) == ["model", "workflows", "request"]
    for category in ("workflows", "request"):
        row = _only_row_of(usage_collector, category)
        assert row["error_type"] == "RuntimeError"
        assert row["resource_details"]["error_type"] == "RuntimeError"
        assert row["resource_details"]["error"].startswith("RuntimeError: ")
    model = _only_row_of(usage_collector, "model")
    assert model["resource_id"] == "ds/1"
    assert model["error_type"] == "RuntimeError"
    assert model["resource_details"]["error"] == "RuntimeError: step broke"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("POST", "/workflows/validate", PASSTHROUGH_WF),
        (
            "POST",
            "/workflows/describe_interface",
            {"specification": PASSTHROUGH_WF, "api_key": "k"},
        ),
        ("POST", "/ws/workflows/wf/describe_interface", {"api_key": "k"}),
        (
            "POST",
            "/workflows/describe_workload",
            {"specification": PASSTHROUGH_WF, "api_key": "k"},
        ),
        ("POST", "/ws/workflows/wf/describe_workload", {"api_key": "k"}),
        ("GET", "/workflows/execution_engine/versions", None),
        ("GET", "/workflows/blocks/describe", None),
        ("POST", "/workflows/blocks/describe", {"api_key": "k"}),
        ("GET", "/workflows/definition/schema", None),
        (
            "POST",
            "/workflows/blocks/dynamic_outputs",
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "det",
                "image": "$inputs.image",
                "model_id": "ds/1",
            },
        ),
    ],
)
def test_other_workflow_routes_record_nothing(
    usage_client, usage_collector, predefined_specification, method, path, body
):
    client = usage_client(FakeGateway())

    response = client.request(method, path, json=body)

    assert response.status_code == 200, response.text
    assert usage_collector.rows == []


def test_recording_failure_leaves_the_workflow_response_unchanged(
    usage_client, usage_collector, caplog
):
    usage_collector.error = RuntimeError("collector broke")
    client = usage_client(FakeGateway())

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage"):
        response = client.post(
            "/workflows/run",
            json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["outputs"] == [{"y": 3}]
    assert usage_collector.rows == []
    assert "Usage of the request was not recorded: RuntimeError" in caplog.text
    assert "Usage of the workflow run was not recorded: RuntimeError" in caplog.text


def test_countinference_false_with_a_valid_secret_is_not_billable(
    usage_client, usage_collector, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(FakeGateway())

    response = client.post(
        f"/workflows/run?countinference=false&service_secret={SECRET}",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    for category in ("request", "workflows"):
        row = _only_row_of(usage_collector, category)
        assert row["billable"] is False
        assert row["roboflow_internal_secret"] == SECRET


def test_countinference_false_without_a_secret_stays_billable(
    usage_client, usage_collector, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_SERVICE_SECRET", SECRET)
    client = usage_client(FakeGateway())

    response = client.post(
        "/workflows/run?countinference=false",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    assert _only_row_of(usage_collector, "request")["billable"] is True
    assert _only_row_of(usage_collector, "workflows")["billable"] is True


def test_workflow_run_without_a_collector_records_nothing(legacy_client):
    import inference_server.app as app_mod

    client = legacy_client(FakeGateway())
    assert app_mod.app.state.usage_collector is None

    response = client.post(
        "/workflows/run",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text


def _custom_python_block(step_type, step_name, block_kind="custom_python", **extra):
    return SimpleNamespace(
        _usage_block_kind=block_kind,
        _usage_block_type="Snippet",
        _usage_resource_id="custom_python/abc123",
        _workflow_step_type=step_type,
        _workflow_step_name=step_name,
        **extra,
    )


def _scoped_observer(**overrides):
    collector = FakeUsageCollector()
    arguments = {"collector": collector, "api_key": "k", **overrides}
    scope = UsageScope(**arguments)

    return UsageExecutionObserver(), scope, collector


def _block_row(block_type, step_name, **overrides):
    row = {
        "api_key": "k",
        "category": "workflow_block",
        "resource_id": "custom_python/abc123",
        "resource_details": {
            "block_kind": "custom_python",
            "block_type": block_type,
            "step_name": step_name,
            "duration_source": "local_runtime",
            "execution_mode": "local",
            "is_preview": False,
        },
        "frames": 1,
        "execution_duration": 0.25,
        "fps": 0.0,
        "source_duration": 0.0,
        "billable": True,
        "is_preview": False,
        "error_type": None,
        "error_status_code": None,
        "roboflow_service_name": None,
        "roboflow_internal_secret": None,
        "megapixel_buckets": None,
    }
    row.update(overrides)

    return row


def test_step_and_run_hooks_leave_the_scope_unbound():
    observer, _, _ = _scoped_observer()
    seen = []

    def _in_fresh_thread():
        context = observer.capture_step_context()
        with observer.step_scope(context=context, step_name="det"):
            seen.append(USAGE_SCOPE.get())
        seen.append(
            observer.observe_workflow_run(
                workflow=None,
                runtime_parameters={},
                workflow_id="wf",
                fps=0,
                is_preview=False,
                run=lambda: "result",
            )
        )

    thread = threading.Thread(target=_in_fresh_thread)
    thread.start()
    thread.join()

    assert seen == [None, "result"]


def test_observe_block_run_records_the_measured_duration_of_a_custom_block():
    observer, scope, collector = _scoped_observer()

    def _run():
        record_block_duration(duration=0.25, source="local_runtime")
        return {"value": 1}

    with bound_scope(scope):
        result = observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={"value": 1},
            run=_run,
        )

    assert result == {"value": 1}
    assert collector.rows == [_block_row("Echo", "echo")]


def test_observe_block_run_falls_back_to_its_own_clock():
    observer, scope, collector = _scoped_observer()

    def _run():
        time.sleep(0.02)
        return {"value": 1}

    with bound_scope(scope):
        observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={},
            run=_run,
        )

    (row,) = collector.rows
    assert row["execution_duration"] >= 0.02
    assert row["resource_details"]["duration_source"] == "decorator_wall_clock"
    assert "execution_mode" not in row["resource_details"]


@pytest.mark.parametrize(
    "source,execution_mode",
    [
        ("remote_runtime", "modal"),
        ("client_wall_clock", "modal"),
        ("unavailable", "modal"),
        ("local_runtime", "local"),
    ],
)
def test_observe_block_run_derives_the_execution_mode_from_the_duration_source(
    source, execution_mode
):
    observer, scope, collector = _scoped_observer()

    with bound_scope(scope):
        observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={},
            run=lambda: record_block_duration(duration=0.5, source=source),
        )

    (row,) = collector.rows
    assert row["execution_duration"] == 0.5
    assert row["resource_details"]["duration_source"] == source
    assert row["resource_details"]["execution_mode"] == execution_mode


def test_observe_block_run_records_a_failing_custom_block():
    observer, scope, collector = _scoped_observer()

    def _run():
        record_block_duration(duration=0.5, source="local_runtime")
        raise ValueError("bad code")

    with bound_scope(scope):
        with pytest.raises(ValueError):
            observer.observe_block_run(
                block=_custom_python_block("Echo", "echo"),
                block_args=(),
                block_kwargs={},
                run=_run,
            )

    (row,) = collector.rows
    assert row["execution_duration"] == 0.5
    assert row["error_type"] == "ValueError"
    assert row["resource_details"]["error"] == "ValueError: bad code"
    assert row["resource_details"]["error_type"] == "ValueError"


def test_observe_block_run_counts_the_largest_batch_as_frames():
    observer, scope, collector = _scoped_observer()
    nested = Batch.init(
        content=[
            Batch.init(content=[1, 2], indices=[(0, 0), (0, 1)]),
            Batch.init(content=[3], indices=[(1, 0)]),
        ],
        indices=[(0,), (1,)],
    )

    with bound_scope(scope):
        observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={
                "images": Batch.init(content=[1, 2], indices=[(0,), (1,)]),
                "crops": nested,
                "x": 7,
            },
            run=lambda: record_block_duration(duration=0.25, source="local_runtime"),
        )

    assert collector.rows == [_block_row("Echo", "echo", frames=3)]


def test_observe_block_run_uses_the_block_key_and_resource_id():
    observer, scope, collector = _scoped_observer()
    block = _custom_python_block("Echo", "echo", _api_key="block-key")
    block._usage_resource_id = None

    with bound_scope(scope):
        observer.observe_block_run(
            block=block,
            block_args=(),
            block_kwargs={},
            run=lambda: record_block_duration(duration=0.25, source="local_runtime"),
        )

    (row,) = collector.rows
    assert row["api_key"] == "block-key"
    assert row["resource_id"] == "unknown"


def test_observe_block_run_records_every_run_of_one_step():
    observer, scope, collector = _scoped_observer()
    block = _custom_python_block("Echo", "echo")

    with bound_scope(scope):
        for duration in (0.25, 0.5):
            observer.observe_block_run(
                block=block,
                block_args=(),
                block_kwargs={},
                run=lambda duration=duration: record_block_duration(
                    duration=duration, source="local_runtime"
                ),
            )

    assert [row["execution_duration"] for row in collector.rows] == [0.25, 0.5]


def test_observe_block_run_ignores_blocks_that_are_not_custom_python():
    observer, scope, collector = _scoped_observer()

    with bound_scope(scope):
        result = observer.observe_block_run(
            block=_custom_python_block("Echo", "echo", block_kind=None),
            block_args=(),
            block_kwargs={},
            run=lambda: "ran",
        )

    assert result == "ran"
    assert collector.rows == []


def test_observe_block_run_without_a_scope_records_nothing():
    observer, _, collector = _scoped_observer()

    result = observer.observe_block_run(
        block=_custom_python_block("Echo", "echo"),
        block_args=(),
        block_kwargs={},
        run=lambda: "ran",
    )

    assert result == "ran"
    assert collector.rows == []


def test_block_rows_inherit_the_preview_source_and_billing_of_the_run():
    observer, scope, collector = _scoped_observer(
        billable=False, source="app", source_info="app-1", service_secret=SECRET
    )

    def _run_workflow():
        observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={},
            run=lambda: record_block_duration(duration=0.25, source="local_runtime"),
        )
        return "done"

    with bound_scope(scope):
        observer.observe_workflow_run(
            workflow=SimpleNamespace(workflow_json=PASSTHROUGH_WF),
            runtime_parameters={},
            workflow_id="wf",
            fps=0,
            is_preview=True,
            run=_run_workflow,
        )

    block, workflow = collector.rows
    assert block["category"] == "workflow_block"
    assert block["is_preview"] is True
    assert block["billable"] is False
    assert block["roboflow_service_name"] == "app-1"
    assert block["roboflow_internal_secret"] == SECRET
    assert block["resource_details"]["is_preview"] is True
    assert block["resource_details"]["source"] == "app"
    assert block["resource_details"]["source_info"] == "app-1"
    assert workflow["category"] == "workflows"
    assert workflow["is_preview"] is True
    assert workflow["billable"] is False


@pytest.mark.parametrize(
    "images,frames",
    [
        (None, 1),
        ([], 1),
        ([object(), object()], 2),
        (
            Batch.init(
                content=[object(), object(), object()], indices=[(0,), (1,), (2,)]
            ),
            3,
        ),
        (np.zeros((4, 2, 2, 3)), 1),
    ],
)
def test_observe_model_run_records_a_model_row(images, frames):
    observer, scope, collector = _scoped_observer()

    with bound_scope(scope):
        result = observer.observe_model_run(
            block=_custom_python_block("Sam2", "sam2", _api_key="block-key"),
            model_id="sam2/hiera_small",
            images=images,
            run=lambda: "segments",
        )

    assert result == "segments"
    (row,) = collector.rows
    assert row == {
        "api_key": "block-key",
        "category": "model",
        "resource_id": "sam2/hiera_small",
        "resource_details": {},
        "frames": frames,
        "execution_duration": row["execution_duration"],
        "fps": 0.0,
        "source_duration": 0.0,
        "billable": True,
        "is_preview": False,
        "error_type": None,
        "error_status_code": None,
        "roboflow_service_name": None,
        "roboflow_internal_secret": None,
        "megapixel_buckets": {
            "unknown": {
                "processed_frames": frames,
                "execution_duration": row["execution_duration"],
            }
        },
    }
    assert row["execution_duration"] >= 0


def test_observe_model_run_records_a_failing_model_call():
    observer, scope, collector = _scoped_observer()

    def _run():
        raise RuntimeError("model broke")

    with bound_scope(scope):
        with pytest.raises(RuntimeError):
            observer.observe_model_run(
                block=_custom_python_block("Sam2", "sam2"),
                model_id="sam2/hiera_small",
                images=[object()],
                run=_run,
            )

    (row,) = collector.rows
    assert row["resource_id"] == "sam2/hiera_small"
    assert row["api_key"] == "k"
    assert row["error_type"] == "RuntimeError"
    assert row["resource_details"]["error"] == "RuntimeError: model broke"


def test_observer_bookkeeping_failure_never_raises_into_the_run(caplog):
    observer, scope, collector = _scoped_observer()

    class _Broken:
        _usage_block_kind = "custom_python"

        @property
        def _workflow_step_type(self):
            raise RuntimeError("no type")

    with bound_scope(scope):
        with caplog.at_level(logging.DEBUG, logger="inference_server.usage.observer"):
            result = observer.observe_block_run(
                block=_Broken(), block_args=(), block_kwargs={}, run=lambda: "ran"
            )

    assert result == "ran"
    assert collector.rows == []
    assert "RuntimeError" in caplog.text


def test_concurrent_rows_reach_the_same_collector():
    observer, scope, collector = _scoped_observer()

    def _step(index):
        with bound_scope(scope):
            observer.observe_model_run(
                block=None, model_id=f"m{index % 4}", images=[1], run=lambda: None
            )
            observer.observe_block_run(
                block=_custom_python_block("Echo", f"echo{index % 4}"),
                block_args=(),
                block_kwargs={},
                run=lambda: record_block_duration(duration=1.0, source="x"),
            )

    threads = [threading.Thread(target=_step, args=(index,)) for index in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    models = _rows_of(collector, "model")
    blocks = _rows_of(collector, "workflow_block")
    assert sorted(row["resource_id"] for row in models) == sorted(
        [f"m{index % 4}" for index in range(40)]
    )
    assert sorted(row["resource_details"]["step_name"] for row in blocks) == sorted(
        [f"echo{index % 4}" for index in range(40)]
    )
    assert all(row["execution_duration"] == 1.0 for row in blocks)


@pytest.mark.parametrize("steps", [None, 5, "abc", {"a": 1}])
def test_malformed_steps_still_record_one_error_row(
    usage_client, usage_collector, steps
):
    client = usage_client(FakeGateway())

    response = client.post(
        "/workflows/run",
        json={
            "specification": {**PASSTHROUGH_WF, "steps": steps},
            "inputs": {"x": 1},
            "api_key": "k",
        },
    )

    assert response.status_code >= 400
    assert _categories(usage_collector) == ["request"]
    row = _only_row_of(usage_collector, "request")
    assert row["error_type"] is not None
    assert row["resource_details"]["steps"] == []


def test_steps_extraction_skips_malformed_entries():
    assert workflow_steps({"steps": [{"type": "a", "name": "b"}, 3, None]}) == ["a:b"]
    assert workflow_steps({}) == []


def _run_block(observer, scope, duration):
    with bound_scope(scope):
        observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={},
            run=lambda: record_block_duration(duration=duration, source="x"),
        )


def test_custom_python_duration_gets_the_serverless_floor(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    token = apply_duration_minimum.set(True)
    try:
        observer, scope, collector = _scoped_observer()
        _run_block(observer, scope, 0.02)
    finally:
        apply_duration_minimum.reset(token)

    assert collector.rows[0]["execution_duration"] == 0.1


def test_custom_python_duration_keeps_its_value_when_the_floor_is_disabled(
    monkeypatch,
):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    token = apply_duration_minimum.set(False)
    try:
        observer, scope, collector = _scoped_observer()
        _run_block(observer, scope, 0.02)
    finally:
        apply_duration_minimum.reset(token)

    assert collector.rows[0]["execution_duration"] == 0.02


def test_custom_python_duration_is_unchanged_without_serverless(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", False)
    observer, scope, collector = _scoped_observer()
    _run_block(observer, scope, 0.02)

    assert collector.rows[0]["execution_duration"] == 0.02


@pytest.fixture
def server_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _request_provider(gateway, loop):
    sync = SyncLegacyBridge(LegacyModelBridge(gateway), LoopBridge(loop))

    return GatewayModelsProvider(sync, api_key="k")


def _detect_in_own_thread(provider, model_id):
    image = {"type": "numpy_object", "value": np.zeros((6, 8, 3), dtype=np.uint8)}

    def _call():
        provider.add_model(model_id, "k")
        return provider.run_object_detection(
            model_id, [image], api_key="k", confidence=0.2
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_call).result(timeout=10)


def test_model_run_in_a_thread_the_block_created_reaches_the_scope_collector(
    fake_stat, server_loop
):
    gateway = _detection_gateway(fake_stat, "ds/1")
    collector = FakeUsageCollector()
    with bound_scope(UsageScope(collector=collector, api_key="k")):
        provider = _request_provider(gateway, server_loop)

    _detect_in_own_thread(provider, "ds/1")

    assert [(row["category"], row["resource_id"]) for row in collector.rows] == [
        ("model", "ds/1")
    ]


def test_provider_embedding_call_records_one_model_row(fake_stat, server_loop):
    gateway = FakeGateway(
        predictions={
            ("clip/ViT-B-16", "embed_images"): lambda image, params: np.array(
                [[1.0, 0.0]]
            )
        },
        model_info={"clip/ViT-B-16": {"actions": {"embed_images": {}}}},
    )
    collector = FakeUsageCollector()
    with bound_scope(UsageScope(collector=collector, api_key="k")):
        provider = _request_provider(gateway, server_loop)
    image = {"type": "numpy_object", "value": np.zeros((6, 8, 3), dtype=np.uint8)}

    def _call():
        provider.add_model("clip/ViT-B-16", "k")
        return provider.run_clip_image_embedding(
            "clip/ViT-B-16", "ViT-B-16", [image, image, image], api_key="k"
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        embeddings = pool.submit(_call).result(timeout=10)

    assert len(embeddings) == 3
    assert [(row["category"], row["resource_id"]) for row in collector.rows] == [
        ("model", "clip/ViT-B-16")
    ]
    assert collector.rows[0]["frames"] == 3
    assert collector.rows[0]["megapixel_buckets"]["0-0.25"]["processed_frames"] == 3


def test_concurrent_requests_keep_their_model_runs_apart(fake_stat, server_loop):
    gateway = _detection_gateway(fake_stat, "ds/1", "ds/2")
    collectors = {"ds/1": FakeUsageCollector(), "ds/2": FakeUsageCollector()}
    providers = {}
    for model_id, collector in collectors.items():
        with bound_scope(UsageScope(collector=collector, api_key="k")):
            providers[model_id] = _request_provider(gateway, server_loop)

    threads = [
        threading.Thread(target=_detect_in_own_thread, args=(provider, model_id))
        for model_id, provider in providers.items()
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for model_id, collector in collectors.items():
        assert [row["resource_id"] for row in collector.rows] == [model_id] * 3
        assert sum(row["frames"] for row in collector.rows) == 3
