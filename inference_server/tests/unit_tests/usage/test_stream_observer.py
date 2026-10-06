import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    record_block_duration,
)
from roboflow_workflows.prototypes.observer import ExecutionObserver
from streamvision.stream.session import stream_session_id

from inference_server.usage.observer import (
    StreamUsageExecutionObserver,
    UsageExecutionObserver,
)
from inference_server.usage.request_hook import (
    CUSTOM_PYTHON_RUNS,
    MODEL_INVOCATIONS,
    _specification_resource_id,
    record_model_invocation,
)
from tests.unit_tests.usage import test_contract
from tests.unit_tests.usage.test_contract import ROW_KEYS, SYSTEM_INFO, _flush_and_stop

posts = test_contract.posts
real_collector = test_contract.real_collector

SPECIFICATION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {"type": "ObjectDetectionModel", "name": "det", "model_id": "ds/1"},
        {"type": "Echo", "name": "echo"},
    ],
    "outputs": [],
}
RUN_KEYS = {
    "api_key",
    "category",
    "resource_id",
    "resource_details",
    "frames",
    "execution_duration",
    "fps",
    "source_duration",
    "billable",
    "is_preview",
    "error_type",
    "error_status_code",
    "exec_session_id",
}
BATCH = [object(), object(), object()]


def _workflow(api_key="key-1", specification=SPECIFICATION):
    return SimpleNamespace(
        workflow_json=specification,
        init_parameters={"workflows_core.api_key": api_key},
    )


def _custom_python_block(step_name):
    return SimpleNamespace(
        _usage_block_kind="custom_python",
        _usage_block_type="Snippet",
        _workflow_step_type="Echo",
        _workflow_step_name=step_name,
    )


@pytest.fixture
def session():
    token = stream_session_id.set("sess-1")
    yield "sess-1"
    stream_session_id.reset(token)


def _run(observer, run=lambda: "result", **overrides):
    arguments = {
        "workflow": _workflow(),
        "runtime_parameters": {"image": list(BATCH)},
        "workflow_id": "wf-1",
        "fps": 25.0,
        "is_preview": False,
        "run": run,
    }
    arguments.update(overrides)

    return observer.observe_workflow_run(**arguments)


def test_stream_observer_implements_the_execution_observer_protocol(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    assert isinstance(observer, ExecutionObserver)
    assert isinstance(observer, UsageExecutionObserver)


def test_run_records_one_request_row(usage_collector, session):
    observer = StreamUsageExecutionObserver(usage_collector, workflow_id="wf-1")

    def _slow():
        time.sleep(0.01)
        return "result"

    result = _run(observer, run=_slow)

    assert result == "result"
    assert len(usage_collector.rows) == 1
    row = usage_collector.rows[0]
    assert set(row) == RUN_KEYS
    assert row["api_key"] == "key-1"
    assert row["category"] == "request"
    assert row["resource_id"] == "wf-1"
    assert row["resource_details"] == {
        "steps": ["ObjectDetectionModel:det", "Echo:echo"],
        "is_preview": False,
        "models": [],
        "custom_python": [],
    }
    assert row["frames"] == 1
    assert row["fps"] == 25.0
    assert row["source_duration"] == 1 / 25.0
    assert row["execution_duration"] >= 0.01
    assert row["billable"] is True
    assert row["is_preview"] is False
    assert row["error_type"] is None
    assert row["error_status_code"] is None
    assert row["exec_session_id"] == session


def test_frames_stay_one_per_run_regardless_of_the_batch(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, runtime_parameters={"image": list(BATCH)}, fps=10.0)
    _run(observer, runtime_parameters={"image": BATCH[0]}, fps=10.0)

    assert [row["frames"] for row in usage_collector.rows] == [1, 1]
    assert [row["source_duration"] for row in usage_collector.rows] == [0.1, 0.1]


@pytest.mark.parametrize("fps", [0, 0.0, -5.0, None])
def test_source_duration_is_zero_without_a_positive_fps(usage_collector, fps):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, fps=fps)

    row = usage_collector.rows[0]
    assert row["fps"] == 0.0
    assert row["source_duration"] == 0.0


def test_negative_fps_is_stored_as_zero_by_the_real_collector(real_collector, posts):
    observer = StreamUsageExecutionObserver(real_collector)

    _run(observer, fps=-5.0)
    _flush_and_stop(real_collector)

    row = posts.only_row()
    assert row["source_duration"] == 0
    assert row["fps"] == 0


def test_exec_session_id_is_unset_outside_a_stream_session(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer)

    assert usage_collector.rows[0]["exec_session_id"] is None


def test_preview_runs_are_flagged(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, is_preview=True)

    row = usage_collector.rows[0]
    assert row["is_preview"] is True
    assert row["resource_details"]["is_preview"] is True


def test_resource_id_falls_back_to_the_specification_hash_the_http_row_uses(
    usage_collector,
):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow_id=None)

    assert usage_collector.rows[0]["resource_id"] == _specification_resource_id(
        SPECIFICATION
    )
    assert usage_collector.rows[0]["resource_id"].startswith("sha:")


def test_inline_specification_is_hashed_whatever_its_own_id(usage_collector):
    specification = dict(SPECIFICATION, id="internal-123")
    observer = StreamUsageExecutionObserver(
        usage_collector, specification=specification
    )

    _run(
        observer,
        workflow_id="internal-123",
        workflow=_workflow(specification=specification),
    )

    resource_id = usage_collector.rows[0]["resource_id"]
    assert resource_id == _specification_resource_id(specification)
    assert resource_id.startswith("sha:")


def test_named_workflow_is_attributed_by_the_requested_name(usage_collector):
    specification = dict(SPECIFICATION, id="other-id")
    observer = StreamUsageExecutionObserver(
        usage_collector, workflow_id="my-workflow", specification=specification
    )

    _run(
        observer,
        workflow_id="other-id",
        workflow=_workflow(specification=specification),
    )

    assert usage_collector.rows[0]["resource_id"] == "my-workflow"


def test_resource_id_is_unknown_without_a_specification(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow_id=None, workflow=_workflow(specification=None))

    row = usage_collector.rows[0]
    assert row["resource_id"] == "unknown"
    assert "steps" not in row["resource_details"]


def test_api_key_comes_from_the_workflow_init_parameters(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow=_workflow(api_key=None))
    _run(observer, workflow=SimpleNamespace(workflow_json=SPECIFICATION))

    assert [row["api_key"] for row in usage_collector.rows] == ["", ""]


def test_failed_run_records_an_error_row_and_reraises(usage_collector, session):
    observer = StreamUsageExecutionObserver(usage_collector)

    def _failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run(observer, run=_failing)

    assert len(usage_collector.rows) == 1
    row = usage_collector.rows[0]
    assert row["error_type"] == "ValueError"
    assert row["error_status_code"] is None
    assert row["resource_details"]["error"] == "ValueError: boom"
    assert row["resource_details"]["error_type"] == "ValueError"
    assert row["exec_session_id"] == session


def test_error_status_code_is_taken_from_the_error(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    class _NotFound(Exception):
        status_code = 404

    def _failing():
        raise _NotFound("gone")

    with pytest.raises(_NotFound):
        _run(observer, run=_failing)

    row = usage_collector.rows[0]
    assert row["error_type"] == "_NotFound"
    assert row["error_status_code"] == 404


def test_error_message_is_redacted_of_the_api_key(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    def _failing():
        raise RuntimeError("secret-key-1 leaked")

    with pytest.raises(RuntimeError):
        _run(observer, run=_failing, workflow=_workflow(api_key="secret-key-1"))

    error = usage_collector.rows[0]["resource_details"]["error"]
    assert "secret-key-1" not in error
    assert error == "RuntimeError: *** leaked"


def test_model_and_custom_python_runs_land_in_the_row(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    def _workflow_body():
        observer.observe_model_run(
            block=None,
            model_id="sam2/hiera_small",
            images=[object(), object()],
            run=lambda: "segments",
        )
        observer.observe_block_run(
            block=_custom_python_block("echo"),
            block_args=(),
            block_kwargs={},
            run=lambda: record_block_duration(duration=0.25, source="local_runtime"),
        )
        return "done"

    assert _run(observer, run=_workflow_body) == "done"

    details = usage_collector.rows[0]["resource_details"]
    assert [entry["model_id"] for entry in details["models"]] == ["sam2/hiera_small"]
    assert details["models"][0]["frames"] == 2
    assert details["custom_python"] == [
        {"block_type": "Echo", "step_name": "echo", "execution_duration": 0.25}
    ]


def test_each_run_starts_with_empty_lists(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    def _with_model():
        observer.observe_model_run(
            block=None, model_id="m1", images=[object()], run=lambda: None
        )

    _run(observer, run=_with_model)
    _run(observer)

    models = [row["resource_details"]["models"] for row in usage_collector.rows]
    assert [len(entries) for entries in models] == [1, 0]
    assert models[0] is not models[1]


def test_invocations_recorded_through_the_holder_land_in_the_row(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)
    entry = {"model_id": "ds/1", "frames": 1, "execution_duration": 0.5}

    _run(observer, run=lambda: record_model_invocation(dict(entry)))

    assert usage_collector.rows[0]["resource_details"]["models"] == [entry]
    assert MODEL_INVOCATIONS.get() is None
    assert CUSTOM_PYTHON_RUNS.get() is None


def test_holders_scope_binds_the_lists_a_bridge_captures(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)
    entry = {"model_id": "ds/1", "frames": 1, "execution_duration": 0.5}

    with observer.holders_scope():
        captured = MODEL_INVOCATIONS.get()
        assert CUSTOM_PYTHON_RUNS.get() is not None
    assert MODEL_INVOCATIONS.get() is None

    def _bridge_call():
        token = MODEL_INVOCATIONS.set(captured)
        try:
            record_model_invocation(dict(entry))
        finally:
            MODEL_INVOCATIONS.reset(token)

    _run(observer, run=_bridge_call)

    assert usage_collector.rows[0]["resource_details"]["models"] == [entry]


def test_step_context_carries_the_session_and_holders_into_a_worker_thread(
    usage_collector,
):
    observer = StreamUsageExecutionObserver(usage_collector)
    seen = {}

    def _worker(context):
        seen["before"] = (
            stream_session_id.get(),
            MODEL_INVOCATIONS.get(),
            CUSTOM_PYTHON_RUNS.get(),
        )
        with observer.step_scope(context=context, step_name="det"):
            seen["inside"] = (
                stream_session_id.get(),
                MODEL_INVOCATIONS.get(),
                CUSTOM_PYTHON_RUNS.get(),
            )
            record_model_invocation(
                {"model_id": "ds/1", "frames": 1, "execution_duration": 0.1}
            )
        seen["after"] = (
            stream_session_id.get(),
            MODEL_INVOCATIONS.get(),
            CUSTOM_PYTHON_RUNS.get(),
        )

    def _workflow_body():
        context = observer.capture_step_context()
        thread = threading.Thread(target=_worker, args=(context,))
        thread.start()
        thread.join()
        seen["holders"] = (MODEL_INVOCATIONS.get(), CUSTOM_PYTHON_RUNS.get())

    token = stream_session_id.set("sess-7")
    try:
        _run(observer, run=_workflow_body)
    finally:
        stream_session_id.reset(token)

    models, custom_python = seen["holders"]
    assert seen["before"] == (None, None, None)
    assert seen["inside"] == ("sess-7", models, custom_python)
    assert seen["inside"][1] is models
    assert seen["inside"][2] is custom_python
    assert seen["after"] == (None, None, None)
    assert [
        entry["model_id"]
        for entry in usage_collector.rows[0]["resource_details"]["models"]
    ] == ["ds/1"]


def test_step_scope_without_a_context_binds_nothing():
    observer = StreamUsageExecutionObserver(SimpleNamespace())

    with observer.step_scope(context=None, step_name="det"):
        assert stream_session_id.get() is None
        assert MODEL_INVOCATIONS.get() is None


def test_recording_failure_never_reaches_the_run(usage_collector, caplog):
    usage_collector.error = RuntimeError("collector down")
    observer = StreamUsageExecutionObserver(usage_collector)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.observer"):
        result = _run(observer)

    assert result == "result"
    assert usage_collector.rows == []
    assert "RuntimeError" in caplog.text


def test_pipeline_row_passes_the_http_row_contract(real_collector, posts, session):
    observer = StreamUsageExecutionObserver(real_collector, workflow_id="wf-1")

    before = time.time_ns()
    _run(observer, run=lambda: record_model_invocation(_gateway_entry()))
    after = time.time_ns()
    _flush_and_stop(real_collector)

    row = posts.only_row()
    assert set(row) == ROW_KEYS
    assert row["api_key"] == "key-1"
    assert row["category"] == "request"
    assert type(row["processed_frames"]) is int
    assert row["processed_frames"] == 1
    for key in ("timestamp_start", "timestamp_stop"):
        assert type(row[key]) is int
        assert before <= row[key] <= after
    assert type(row["execution_duration"]) is float
    assert row["fps"] == 25.0
    assert row["source_duration"] == 1 / 25.0
    assert row["megapixel_buckets"] == {}
    assert row["hostname"] == SYSTEM_INFO["hostname"]
    assert row["ip_address_hash"] == SYSTEM_INFO["ip_address_hash"]
    assert row["exec_session_id"] == session
    assert row["resource_id"] == "wf-1"
    assert "api_key_hash" not in row
    assert "stream_session_id" not in row
    details = json.loads(row["resource_details"])
    assert details["billable"] is True
    assert details["is_preview"] is False
    assert details["steps"] == ["ObjectDetectionModel:det", "Echo:echo"]
    assert details["custom_python"] == []
    assert len(details["models"]) == 1
    assert details["models"][0]["model_id"] == "ds/1"


def _gateway_entry():
    return {
        "model_id": "ds/1",
        "model_architecture": "yolov8",
        "model_variant": "yolov8-n",
        "task_type": "object-detection",
        "model_input_height": 640,
        "model_input_width": 640,
        "execution_duration": 0.01,
        "frames": 1,
    }
