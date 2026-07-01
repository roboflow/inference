"""Contract tests for `python_blocks_output_streams` and `python_blocks_debug_traces` exposure on the workflow run endpoint.

These tests pin the over-the-wire JSON shape that clients (e.g. the Roboflow
editor's Debug Mode) depend on:
- `debug=True` + printing block -> 200 with `python_blocks_output_streams` populated,
- `debug=True` + block using `debug_traces.append()` -> 200 with `python_blocks_debug_traces` populated,
- `debug=True` + failing block  -> 400 with `python_blocks_output_streams` (logs of the
  steps executed before the failure, plus the failing step itself),
  `python_blocks_debug_traces` (entries appended before the failure) and
  `block_traceback` carrying the failing step's streams,
- no `debug` flag               -> `python_blocks_output_streams` is null,
- `debug=True` + silent block   -> `python_blocks_output_streams` is null.
"""

from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_test_client(monkeypatch) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    interface = http_api.HttpInterface(model_manager=model_manager)
    return TestClient(interface.app)


def _dynamic_block_definition(block_type: str, code: str) -> dict:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter", "step_output"],
                },
            },
            "outputs": {"result": {"type": "DynamicOutputDefinition", "kind": []}},
        },
        "code": {"type": "PythonCode", "run_function_code": code},
    }


PRINTING_BLOCK_CODE = """
import sys

def run(self, value) -> BlockResult:
    print("stdout of printing block:", value)
    print("stderr of printing block", file=sys.stderr)
    return {"result": value}
"""

SILENT_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    return {"result": value}
"""

FAILING_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    debug_traces.append({"about_to_fail": value})
    print("printed right before failure")
    raise RuntimeError("boom")
"""

DEBUG_TRACE_BLOCK_CODE = """
def run(self, value) -> BlockResult:
    debug_traces.append({"received": value, "doubled": value * 2}, add_timestamp=True)
    return {"result": value * 2}
"""


SUCCESS_SPECIFICATION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        _dynamic_block_definition("PrintingBlock", PRINTING_BLOCK_CODE),
    ],
    "steps": [
        {"type": "PrintingBlock", "name": "printing_step", "value": "$inputs.value"},
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.printing_step.result",
        },
    ],
}

SILENT_SPECIFICATION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        _dynamic_block_definition("SilentBlock", SILENT_BLOCK_CODE),
    ],
    "steps": [
        {"type": "SilentBlock", "name": "silent_step", "value": "$inputs.value"},
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.silent_step.result",
        },
    ],
}

DEBUG_TRACE_SPECIFICATION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        _dynamic_block_definition("DebugTraceBlock", DEBUG_TRACE_BLOCK_CODE),
    ],
    "steps": [
        {
            "type": "DebugTraceBlock",
            "name": "debug_trace_step",
            "value": "$inputs.value",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.debug_trace_step.result",
        },
    ],
}

# failing_step consumes printing_step's output, so the printing step is
# guaranteed to complete before the failure happens.
FAILING_SPECIFICATION = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        _dynamic_block_definition("PrintingBlock", PRINTING_BLOCK_CODE),
        _dynamic_block_definition("FailingBlock", FAILING_BLOCK_CODE),
    ],
    "steps": [
        {"type": "PrintingBlock", "name": "printing_step", "value": "$inputs.value"},
        {
            "type": "FailingBlock",
            "name": "failing_step",
            "value": "$steps.printing_step.result",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.failing_step.result",
        },
    ],
}


def test_workflow_run_with_debug_returns_python_blocks_debug_traces(
    monkeypatch,
) -> None:
    # given
    client = _build_test_client(monkeypatch)

    # when
    response = client.post(
        "/workflows/run",
        json={
            "specification": DEBUG_TRACE_SPECIFICATION,
            "inputs": {"value": 7},
            "debug": True,
        },
    )

    # then
    assert response.status_code == 200
    body = response.json()
    assert body["outputs"] == [{"result": 14}]
    trace = body["python_blocks_debug_traces"]
    assert len(trace) == 1
    assert trace[0]["step"] == "debug_trace_step"
    assert trace[0]["value"] == {"received": 7, "doubled": 14}
    assert "timestamp" in trace[0]
    assert trace[0]["timestamp_timezone"] == "UTC"
    assert trace[0]["timestamp"].endswith("+00:00")


def test_workflow_run_with_debug_returns_python_blocks_output_streams(
    monkeypatch,
) -> None:
    # given
    client = _build_test_client(monkeypatch)

    # when
    response = client.post(
        "/workflows/run",
        json={
            "specification": SUCCESS_SPECIFICATION,
            "inputs": {"value": 7},
            "debug": True,
        },
    )

    # then
    assert response.status_code == 200
    body = response.json()
    assert body["outputs"] == [{"result": 7}]
    logs = body["python_blocks_output_streams"]
    assert list(logs.keys()) == ["printing_step"]
    assert len(logs["printing_step"]) == 1
    entry = logs["printing_step"][0]
    assert "stdout of printing block: 7" in entry["stdout"]
    assert "stderr of printing block" in entry["stderr"]


def test_workflow_run_failure_with_debug_returns_partial_logs_in_error_response(
    monkeypatch,
) -> None:
    # given
    client = _build_test_client(monkeypatch)

    # when
    response = client.post(
        "/workflows/run",
        json={
            "specification": FAILING_SPECIFICATION,
            "inputs": {"value": 7},
            "debug": True,
        },
    )

    # then - 400 with logs of both the completed and the failing step
    assert response.status_code == 400
    body = response.json()
    logs = body["python_blocks_output_streams"]
    assert "stdout of printing block: 7" in logs["printing_step"][0]["stdout"]
    assert "printed right before failure" in logs["failing_step"][0]["stdout"]
    # the failing step's streams also keep riding the block traceback
    block_error = body["blocks_errors"][0]
    assert "printed right before failure" in block_error["block_traceback"]["stdout"]
    # partial debug_traces appended before the failure must also survive on the
    # error response - this guards the exception-identity propagation (http_api
    # sets error.python_blocks_debug_traces; error_handlers reads it via getattr).
    # A future change that re-wraps the exception would silently drop these.
    traces = body["python_blocks_debug_traces"]
    assert any(
        entry["step"] == "failing_step" and entry["value"] == {"about_to_fail": 7}
        for entry in traces
    )


def test_workflow_run_without_debug_returns_null_python_blocks_output_streams(
    monkeypatch,
) -> None:
    # given
    client = _build_test_client(monkeypatch)

    # when - block prints, but debug flag is not set
    response = client.post(
        "/workflows/run",
        json={
            "specification": SUCCESS_SPECIFICATION,
            "inputs": {"value": 7},
        },
    )

    # then
    assert response.status_code == 200
    body = response.json()
    assert body["outputs"] == [{"result": 7}]
    assert body.get("python_blocks_output_streams") is None


def test_workflow_run_with_debug_and_silent_blocks_returns_null_python_blocks_output_streams(
    monkeypatch,
) -> None:
    # given
    client = _build_test_client(monkeypatch)

    # when - debug requested, but no block writes to stdout/stderr
    response = client.post(
        "/workflows/run",
        json={
            "specification": SILENT_SPECIFICATION,
            "inputs": {"value": 7},
            "debug": True,
        },
    )

    # then - empty capture serializes as null, not {}
    assert response.status_code == 200
    body = response.json()
    assert body["outputs"] == [{"result": 7}]
    assert body.get("python_blocks_output_streams") is None


def test_workflow_run_without_debug_returns_null_python_blocks_debug_traces(
    monkeypatch,
) -> None:
    # given
    client = _build_test_client(monkeypatch)

    # when
    response = client.post(
        "/workflows/run",
        json={
            "specification": DEBUG_TRACE_SPECIFICATION,
            "inputs": {"value": 7},
        },
    )

    # then
    assert response.status_code == 200
    assert response.json().get("python_blocks_debug_traces") is None
