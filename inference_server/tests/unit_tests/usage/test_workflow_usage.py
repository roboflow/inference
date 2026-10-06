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
from inference_server.usage.request_hook import (
    MODEL_INVOCATIONS,
    _workflow_steps,
)
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy.conftest import FakeGateway

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


def _only_row(usage_collector):
    assert len(usage_collector.rows) == 1

    return usage_collector.rows[0]


@pytest.fixture
def predefined_specification(monkeypatch):
    holder = {"specification": PASSTHROUGH_WF}
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: holder["specification"],
    )

    return holder


@pytest.mark.parametrize("path", PREDEFINED_PATHS)
def test_predefined_route_records_one_request_row(
    usage_client, usage_collector, predefined_specification, path
):
    client = usage_client(FakeGateway())

    response = client.post(path, json={"inputs": {"x": 3}, "api_key": "k"})

    assert response.status_code == 200, response.text
    row = _only_row(usage_collector)
    assert row["category"] == "request"
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
        "models": [],
        "custom_python": [],
    }


@pytest.mark.parametrize("path", SPECIFICATION_PATHS)
def test_specification_route_records_one_request_row(
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
    row = _only_row(usage_collector)
    assert row["category"] == "request"
    assert row["api_key"] == "k"
    assert row["resource_id"] == "body-wf"
    assert row["frames"] == 1
    assert row["billable"] is True
    assert row["resource_details"] == {
        "steps": [],
        "is_preview": False,
        "models": [],
        "custom_python": [],
    }


def test_path_workflow_id_beats_the_body_workflow_id(
    usage_client, usage_collector, predefined_specification
):
    client = usage_client(FakeGateway())

    response = client.post(
        "/ws/workflows/wf",
        json={"inputs": {"x": 3}, "api_key": "k", "workflow_id": "other"},
    )

    assert response.status_code == 200, response.text
    assert _only_row(usage_collector)["resource_id"] == "wf"


def test_resource_id_falls_back_to_the_specification_hash(
    usage_client, usage_collector
):
    client = usage_client(FakeGateway())

    response = client.post(
        "/workflows/run",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text
    assert _only_row(usage_collector)["resource_id"] == PASSTHROUGH_WF_SHA


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
    assert _only_row(usage_collector)["resource_details"]["steps"] == [
        "roboflow_core/roboflow_object_detection_model@v2:det0",
        "roboflow_core/roboflow_object_detection_model@v2:det1",
    ]


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
    assert _only_row(usage_collector)["resource_details"]["steps"] == [
        "roboflow_core/roboflow_object_detection_model@v2:det0"
    ]


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
    row = _only_row(usage_collector)
    assert row["is_preview"] is is_preview
    assert row["resource_details"]["is_preview"] is is_preview
    assert row["billable"] is True


def test_failing_step_records_an_error_row_with_the_models_collected(
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
    row = _only_row(usage_collector)
    assert row["category"] == "request"
    assert row["error_type"] == "StepExecutionError"
    assert row["resource_details"]["error_type"] == "StepExecutionError"
    assert row["resource_details"]["error"].startswith("StepExecutionError: ")
    assert [entry["model_id"] for entry in row["resource_details"]["models"]] == [
        "ds/1"
    ]


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

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.request_hook"):
        response = client.post(
            "/workflows/run",
            json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["outputs"] == [{"y": 3}]
    assert usage_collector.rows == []
    assert "Usage of the request was not recorded: RuntimeError" in caplog.text


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
    row = _only_row(usage_collector)
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
    assert _only_row(usage_collector)["billable"] is True


def test_workflow_run_without_a_collector_records_nothing(legacy_client):
    import inference_server.app as app_mod

    client = legacy_client(FakeGateway())
    assert app_mod.app.state.usage_collector is None

    response = client.post(
        "/workflows/run",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )

    assert response.status_code == 200, response.text


def _custom_python_block(step_type, step_name, block_kind="custom_python"):
    return SimpleNamespace(
        _usage_block_kind=block_kind,
        _usage_block_type="Snippet",
        _workflow_step_type=step_type,
        _workflow_step_name=step_name,
    )


def _observer():
    models, custom_python = [], []
    observer = UsageExecutionObserver(models=models, custom_python=custom_python)

    return observer, models, custom_python


def test_step_and_run_hooks_leave_the_holder_unbound():
    observer, _, _ = _observer()
    seen = []

    def _in_fresh_thread():
        context = observer.capture_step_context()
        with observer.step_scope(context=context, step_name="det"):
            seen.append(MODEL_INVOCATIONS.get())
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
    observer, _, custom_python = _observer()

    def _run():
        record_block_duration(duration=0.25, source="local_runtime")
        return {"value": 1}

    result = observer.observe_block_run(
        block=_custom_python_block("Echo", "echo"),
        block_args=(),
        block_kwargs={"value": 1},
        run=_run,
    )

    assert result == {"value": 1}
    assert custom_python == [
        {"block_type": "Echo", "step_name": "echo", "execution_duration": 0.25}
    ]


def test_observe_block_run_falls_back_to_its_own_clock():
    observer, _, custom_python = _observer()

    def _run():
        time.sleep(0.02)
        return {"value": 1}

    observer.observe_block_run(
        block=_custom_python_block("Echo", "echo"),
        block_args=(),
        block_kwargs={},
        run=_run,
    )

    assert len(custom_python) == 1
    assert custom_python[0]["execution_duration"] >= 0.02


def test_observe_block_run_records_a_failing_custom_block():
    observer, _, custom_python = _observer()

    def _run():
        record_block_duration(duration=0.5, source="local_runtime")
        raise ValueError("bad code")

    with pytest.raises(ValueError):
        observer.observe_block_run(
            block=_custom_python_block("Echo", "echo"),
            block_args=(),
            block_kwargs={},
            run=_run,
        )

    assert custom_python == [
        {"block_type": "Echo", "step_name": "echo", "execution_duration": 0.5}
    ]


def test_observe_block_run_merges_repeated_runs_of_one_step():
    observer, _, custom_python = _observer()
    block = _custom_python_block("Echo", "echo")

    for duration in (0.25, 0.5):
        observer.observe_block_run(
            block=block,
            block_args=(),
            block_kwargs={},
            run=lambda duration=duration: record_block_duration(
                duration=duration, source="local_runtime"
            ),
        )

    assert custom_python == [
        {"block_type": "Echo", "step_name": "echo", "execution_duration": 0.75}
    ]


def test_observe_block_run_ignores_blocks_that_are_not_custom_python():
    observer, _, custom_python = _observer()

    result = observer.observe_block_run(
        block=_custom_python_block("Echo", "echo", block_kind=None),
        block_args=(),
        block_kwargs={},
        run=lambda: "ran",
    )

    assert result == "ran"
    assert custom_python == []


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
def test_observe_model_run_records_a_models_entry(images, frames):
    observer, models, _ = _observer()

    result = observer.observe_model_run(
        block=_custom_python_block("Sam2", "sam2"),
        model_id="sam2/hiera_small",
        images=images,
        run=lambda: "segments",
    )

    assert result == "segments"
    assert models == [
        {
            "model_id": "sam2/hiera_small",
            "frames": frames,
            "execution_duration": models[0]["execution_duration"],
        }
    ]
    assert models[0]["execution_duration"] >= 0


def test_observe_model_run_records_a_failing_model_call():
    observer, models, _ = _observer()

    def _run():
        raise RuntimeError("model broke")

    with pytest.raises(RuntimeError):
        observer.observe_model_run(
            block=_custom_python_block("Sam2", "sam2"),
            model_id="sam2/hiera_small",
            images=[object()],
            run=_run,
        )

    assert [entry["model_id"] for entry in models] == ["sam2/hiera_small"]


def test_observer_bookkeeping_failure_never_raises_into_the_run(caplog):
    observer, models, custom_python = _observer()

    class _Broken:
        _usage_block_kind = "custom_python"

        @property
        def _workflow_step_type(self):
            raise RuntimeError("no type")

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.observer"):
        result = observer.observe_block_run(
            block=_Broken(), block_args=(), block_kwargs={}, run=lambda: "ran"
        )

    assert result == "ran"
    assert custom_python == []
    assert models == []
    assert "RuntimeError" in caplog.text


def test_concurrent_appends_reach_the_same_lists():
    observer, models, custom_python = _observer()

    def _step(index):
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

    assert sorted(entry["model_id"] for entry in models) == ["m0", "m1", "m2", "m3"]
    assert all(entry["frames"] == 10 for entry in models)
    assert sorted(entry["step_name"] for entry in custom_python) == [
        "echo0",
        "echo1",
        "echo2",
        "echo3",
    ]
    assert all(entry["execution_duration"] == 10.0 for entry in custom_python)


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
    row = _only_row(usage_collector)
    assert row["error_type"] is not None
    assert row["resource_details"]["steps"] == []


def test_steps_extraction_skips_malformed_entries():
    assert _workflow_steps({"steps": [{"type": "a", "name": "b"}, 3, None]}) == ["a:b"]
    assert _workflow_steps({}) == []


def _run_block(observer, duration):
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
        observer, _, custom_python = _observer()
        _run_block(observer, 0.02)
    finally:
        apply_duration_minimum.reset(token)

    assert custom_python[0]["execution_duration"] == 0.1


def test_custom_python_duration_keeps_its_value_when_the_floor_is_disabled(
    monkeypatch,
):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    token = apply_duration_minimum.set(False)
    try:
        observer, _, custom_python = _observer()
        _run_block(observer, 0.02)
    finally:
        apply_duration_minimum.reset(token)

    assert custom_python[0]["execution_duration"] == 0.02


def test_custom_python_duration_is_unchanged_without_serverless(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", False)
    observer, _, custom_python = _observer()
    _run_block(observer, 0.02)

    assert custom_python[0]["execution_duration"] == 0.02


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


def test_model_run_in_a_thread_the_block_created_reaches_the_row(
    fake_stat, server_loop
):
    gateway = _detection_gateway(fake_stat, "ds/1")
    models = []
    token = MODEL_INVOCATIONS.set(models)
    try:
        provider = _request_provider(gateway, server_loop)
    finally:
        MODEL_INVOCATIONS.reset(token)

    _detect_in_own_thread(provider, "ds/1")

    assert [entry["model_id"] for entry in models] == ["ds/1"]


def test_concurrent_requests_keep_their_model_runs_apart(fake_stat, server_loop):
    gateway = _detection_gateway(fake_stat, "ds/1", "ds/2")
    holders = {"ds/1": [], "ds/2": []}
    providers = {}
    for model_id, holder in holders.items():
        token = MODEL_INVOCATIONS.set(holder)
        try:
            providers[model_id] = _request_provider(gateway, server_loop)
        finally:
            MODEL_INVOCATIONS.reset(token)

    threads = [
        threading.Thread(target=_detect_in_own_thread, args=(provider, model_id))
        for model_id, provider in providers.items()
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [entry["model_id"] for entry in holders["ds/1"]] == ["ds/1"]
    assert [entry["model_id"] for entry in holders["ds/2"]] == ["ds/2"]
    assert holders["ds/1"][0]["frames"] == 3
    assert holders["ds/2"][0]["frames"] == 3
