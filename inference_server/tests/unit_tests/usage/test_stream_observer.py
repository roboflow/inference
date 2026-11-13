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
from inference_server.usage.rows import (
    USAGE_SCOPE,
    record_model_usage,
    steps_resource_id,
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
STEPS = ["ObjectDetectionModel:det", "Echo:echo"]
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
    "roboflow_service_name",
    "roboflow_internal_secret",
    "megapixel_buckets",
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
        _usage_resource_id="custom_python/abc123",
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


def _gateway_row(scope):
    record_model_usage(
        scope,
        model_id="ds/1",
        api_key="key-1",
        frames=1,
        duration=0.01,
        details={
            "model_architecture": "yolov8",
            "model_variant": "yolov8-n",
            "task_type": "object-detection",
            "model_input_height": 640,
            "model_input_width": 640,
        },
        megapixel_buckets={
            "0.25-0.5": {"processed_frames": 1, "execution_duration": 0.01}
        },
    )


def test_stream_observer_implements_the_execution_observer_protocol(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    assert isinstance(observer, ExecutionObserver)
    assert isinstance(observer, UsageExecutionObserver)


def test_run_records_one_workflows_row(usage_collector, session):
    observer = StreamUsageExecutionObserver(usage_collector, api_key="pipeline-key")

    def _slow():
        time.sleep(0.01)
        return "result"

    result = _run(observer, run=_slow)

    assert result == "result"
    assert len(usage_collector.rows) == 1
    row = usage_collector.rows[0]
    assert set(row) == RUN_KEYS
    assert row["api_key"] == "key-1"
    assert row["category"] == "workflows"
    assert row["resource_id"] == "wf-1"
    assert row["resource_details"] == {"steps": STEPS, "is_preview": False}
    assert row["frames"] == 1
    assert row["fps"] == 25.0
    assert row["source_duration"] == 1 / 25.0
    assert row["execution_duration"] >= 0.01
    assert row["billable"] is True
    assert row["is_preview"] is False
    assert row["error_type"] is None
    assert row["error_status_code"] is None
    assert row["megapixel_buckets"] is None
    assert observer.scope.stream_session_id == session


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


def test_preview_runs_are_flagged(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, is_preview=True)

    row = usage_collector.rows[0]
    assert row["is_preview"] is True
    assert row["resource_details"]["is_preview"] is True


def test_resource_id_falls_back_to_the_hash_of_the_step_list(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow_id=None)

    assert usage_collector.rows[0]["resource_id"] == steps_resource_id(STEPS)


def test_engine_workflow_id_attributes_the_row(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow_id="internal-123")

    assert usage_collector.rows[0]["resource_id"] == "internal-123"


def test_resource_id_is_unknown_without_a_specification(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow_id=None, workflow=_workflow(specification=None))

    row = usage_collector.rows[0]
    assert row["resource_id"] == "unknown"
    assert "steps" not in row["resource_details"]


def test_api_key_falls_back_to_the_pipeline_key(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector, api_key="pipeline-key")

    _run(observer, workflow=_workflow(api_key=None))
    _run(observer, workflow=SimpleNamespace(workflow_json=SPECIFICATION))

    assert [row["api_key"] for row in usage_collector.rows] == ["pipeline-key"] * 2


def test_api_key_is_empty_without_any_key(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, workflow=_workflow(api_key=None))

    assert usage_collector.rows[0]["api_key"] == ""


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


def test_wrapped_error_reports_the_inner_error_type_and_status(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    class _NotFound(Exception):
        status_code = 404

    class _Wrapped(Exception):
        def __init__(self, inner):
            super().__init__("wrapped")
            self.inner_error = inner
            self.inner_error_type = type(inner).__name__

    def _failing():
        raise _Wrapped(_NotFound("gone"))

    with pytest.raises(_Wrapped):
        _run(observer, run=_failing)

    row = usage_collector.rows[0]
    assert row["error_type"] == "_NotFound"
    assert row["error_status_code"] == 404
    assert row["resource_details"]["error"] == "_NotFound: wrapped"


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


def test_model_and_custom_python_runs_record_rows_of_their_own(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector, api_key="pipeline-key")

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

    assert [(row["category"], row["resource_id"]) for row in usage_collector.rows] == [
        ("model", "sam2/hiera_small"),
        ("workflow_block", "custom_python/abc123"),
        ("workflows", "wf-1"),
    ]
    model, block, _ = usage_collector.rows
    assert model["frames"] == 2
    assert model["api_key"] == "pipeline-key"
    assert block["execution_duration"] == 0.25
    assert block["resource_details"]["step_name"] == "echo"
    assert block["api_key"] == "pipeline-key"


def test_rows_recorded_through_the_scope_land_on_the_collector(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    _run(observer, run=lambda: _gateway_row(USAGE_SCOPE.get()))

    assert [row["category"] for row in usage_collector.rows] == [
        "model",
        "workflows",
    ]
    assert USAGE_SCOPE.get() is None


def test_scope_binding_binds_the_scope_a_bridge_captures(usage_collector):
    observer = StreamUsageExecutionObserver(usage_collector)

    with observer.scope_binding():
        captured = USAGE_SCOPE.get()
    assert captured is observer.scope
    assert USAGE_SCOPE.get() is None

    _run(observer, run=lambda: _gateway_row(captured))

    assert [row["category"] for row in usage_collector.rows] == [
        "model",
        "workflows",
    ]


def test_step_context_carries_the_session_and_scope_into_a_worker_thread(
    usage_collector,
):
    observer = StreamUsageExecutionObserver(usage_collector)
    seen = {}

    def _worker(context):
        seen["before"] = (stream_session_id.get(), USAGE_SCOPE.get())
        with observer.step_scope(context=context, step_name="det"):
            seen["inside"] = (stream_session_id.get(), USAGE_SCOPE.get())
            _gateway_row(USAGE_SCOPE.get())
        seen["after"] = (stream_session_id.get(), USAGE_SCOPE.get())

    def _workflow_body():
        context = observer.capture_step_context()
        thread = threading.Thread(target=_worker, args=(context,))
        thread.start()
        thread.join()

    token = stream_session_id.set("sess-7")
    try:
        _run(observer, run=_workflow_body)
    finally:
        stream_session_id.reset(token)

    assert seen["before"] == (None, None)
    assert seen["inside"] == ("sess-7", observer.scope)
    assert seen["after"] == (None, None)
    assert [row["category"] for row in usage_collector.rows] == [
        "model",
        "workflows",
    ]


def test_step_scope_without_a_context_binds_nothing():
    observer = StreamUsageExecutionObserver(SimpleNamespace())

    with observer.step_scope(context=None, step_name="det"):
        assert stream_session_id.get() is None
        assert USAGE_SCOPE.get() is None


def test_recording_failure_never_reaches_the_run(usage_collector, caplog):
    usage_collector.error = RuntimeError("collector down")
    observer = StreamUsageExecutionObserver(usage_collector)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage.observer"):
        result = _run(observer)

    assert result == "result"
    assert usage_collector.rows == []
    assert "RuntimeError" in caplog.text


class _SessionRecordingCollector:
    def __init__(self):
        self.sessions = []

    def record_usage(self, **row):
        self.sessions.append((row["category"], stream_session_id.get()))


def test_rows_recorded_outside_the_session_thread_are_bound_to_the_run_session(
    session,
):
    collector = _SessionRecordingCollector()
    observer = StreamUsageExecutionObserver(collector)
    seen = []

    def _loop_thread():
        with observer.scope_binding():
            _gateway_row(USAGE_SCOPE.get())
        seen.append(stream_session_id.get())

    _run(observer)
    thread = threading.Thread(target=_loop_thread)
    thread.start()
    thread.join()

    assert collector.sessions == [("workflows", session), ("model", session)]
    assert seen == [None]


def test_pipeline_rows_pass_the_http_row_contract(real_collector, posts, session):
    observer = StreamUsageExecutionObserver(real_collector)

    before = time.time_ns()
    _run(observer, run=lambda: _gateway_row(USAGE_SCOPE.get()))
    after = time.time_ns()
    _flush_and_stop(real_collector)

    assert len(posts.calls) == 2
    rows = {row["category"]: row for call in posts.calls for row in call.json}
    assert set(rows) == {"workflows", "model"}
    assert [len(call.json) for call in posts.calls] == [1, 1]
    for row in rows.values():
        assert set(row) == ROW_KEYS
        assert row["api_key"] == "key-1"
        assert type(row["processed_frames"]) is int
        assert row["processed_frames"] == 1
        for key in ("timestamp_start", "timestamp_stop"):
            assert type(row[key]) is int
            assert before <= row[key] <= after
        assert type(row["execution_duration"]) is float
        assert row["hostname"] == SYSTEM_INFO["hostname"]
        assert row["ip_address_hash"] == SYSTEM_INFO["ip_address_hash"]
        assert row["exec_session_id"] == session
        assert "api_key_hash" not in row
        assert "stream_session_id" not in row
        assert json.loads(row["resource_details"])["billable"] is True
    workflow = rows["workflows"]
    assert workflow["fps"] == 25.0
    assert workflow["source_duration"] == 1 / 25.0
    assert workflow["megapixel_buckets"] == {}
    assert workflow["resource_id"] == "wf-1"
    details = json.loads(workflow["resource_details"])
    assert details["is_preview"] is False
    assert details["steps"] == STEPS
    model = rows["model"]
    assert model["resource_id"] == "ds/1"
    assert model["fps"] == 0.0
    assert model["megapixel_buckets"] == {
        "0.25-0.5": {"processed_frames": 1, "execution_duration": 0.01}
    }
    assert json.loads(model["resource_details"])["model_architecture"] == "yolov8"
