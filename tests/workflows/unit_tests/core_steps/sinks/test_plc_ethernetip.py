from unittest.mock import MagicMock, patch

import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

_TRIGGER_STEP = {
    "type": "roboflow_core/json_parser@v1",
    "name": "trigger",
    "raw_json": '{"ok": true}',
    "expected_fields": ["ok"],
}

WORKFLOW_PLC_WRITE = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceParameter", "name": "plc_ip"},
        {"type": "InferenceParameter", "name": "tags_to_write"},
    ],
    "steps": [
        _TRIGGER_STEP,
        {
            "type": "roboflow_core/sinks@v1",
            "name": "plc_step",
            "plc_ip": "$inputs.plc_ip",
            "mode": "write",
            "tags_to_write": "$inputs.tags_to_write",
            "depends_on": "$steps.trigger.ok",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "plc_results",
            "selector": "$steps.plc_step.plc_results",
        }
    ],
}

WORKFLOW_PLC_READ = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceParameter", "name": "plc_ip"},
        {"type": "InferenceParameter", "name": "tags_to_read"},
    ],
    "steps": [
        _TRIGGER_STEP,
        {
            "type": "roboflow_core/sinks@v1",
            "name": "plc_step",
            "plc_ip": "$inputs.plc_ip",
            "mode": "read",
            "tags_to_read": "$inputs.tags_to_read",
            "depends_on": "$steps.trigger.ok",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "plc_results",
            "selector": "$steps.plc_step.plc_results",
        }
    ],
}

WORKFLOW_PLC_READ_AND_WRITE = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceParameter", "name": "plc_ip"},
        {"type": "InferenceParameter", "name": "tags_to_read"},
        {"type": "InferenceParameter", "name": "tags_to_write"},
    ],
    "steps": [
        _TRIGGER_STEP,
        {
            "type": "roboflow_core/sinks@v1",
            "name": "plc_step",
            "plc_ip": "$inputs.plc_ip",
            "mode": "read_and_write",
            "tags_to_read": "$inputs.tags_to_read",
            "tags_to_write": "$inputs.tags_to_write",
            "depends_on": "$steps.trigger.ok",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "plc_results",
            "selector": "$steps.plc_step.plc_results",
        }
    ],
}


def _make_mock_response(status="Success", value=None):
    resp = MagicMock()
    resp.Status = status
    resp.Value = value
    return resp


def _setup_mock_comm(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)
    return mock_comm


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_write(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)
    mock_comm.Write.return_value = _make_mock_response(status="Success")

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_WRITE,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_write": {"camera_fault": True, "defect_count": 5},
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert len(plc_results) == 1
    assert "write" in plc_results[0]
    assert plc_results[0]["write"]["camera_fault"] == "WriteSuccess"
    assert plc_results[0]["write"]["defect_count"] == "WriteSuccess"


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_read(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)

    def read_side_effect(tag):
        values = {"camera_msg": "OK", "sku_number": 42}
        return _make_mock_response(status="Success", value=values.get(tag))

    mock_comm.Read.side_effect = read_side_effect

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_READ,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_read": ["camera_msg", "sku_number"],
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert len(plc_results) == 1
    assert "read" in plc_results[0]
    assert plc_results[0]["read"]["camera_msg"] == "OK"
    assert plc_results[0]["read"]["sku_number"] == 42


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_read_and_write(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)

    def read_side_effect(tag):
        values = {"sensor_val": 3.14}
        return _make_mock_response(status="Success", value=values.get(tag))

    mock_comm.Read.side_effect = read_side_effect
    mock_comm.Write.return_value = _make_mock_response(status="Success")

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_READ_AND_WRITE,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_read": ["sensor_val"],
            "tags_to_write": {"output_flag": 1},
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert len(plc_results) == 1
    assert "read" in plc_results[0]
    assert "write" in plc_results[0]
    assert plc_results[0]["read"]["sensor_val"] == 3.14
    assert plc_results[0]["write"]["output_flag"] == "WriteSuccess"


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_read_status_failure(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)
    mock_comm.Read.return_value = _make_mock_response(status="Connection lost")

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_READ,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_read": ["bad_tag"],
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert plc_results[0]["read"]["bad_tag"] == "ReadFailure"


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_write_status_failure(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)
    mock_comm.Write.return_value = _make_mock_response(status="Connection lost")

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_WRITE,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_write": {"bad_tag": 99},
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert plc_results[0]["write"]["bad_tag"] == "WriteFailure"


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_read_exception(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)
    mock_comm.Read.side_effect = RuntimeError("PLC timeout")

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_READ,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_read": ["unreachable_tag"],
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert plc_results[0]["read"]["unreachable_tag"] == "ReadFailure"


@pytest.mark.timeout(10)
@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_workflow_plc_mixed_results(mock_pylogix) -> None:
    mock_comm = _setup_mock_comm(mock_pylogix)

    def read_side_effect(tag):
        if tag == "good_read":
            return _make_mock_response(status="Success", value=100)
        return _make_mock_response(status="Path segment error")

    mock_comm.Read.side_effect = read_side_effect

    call_count = 0

    def write_side_effect(tag, value):
        nonlocal call_count
        call_count += 1
        if tag == "good_write":
            return _make_mock_response(status="Success")
        raise RuntimeError("Write timeout")

    mock_comm.Write.side_effect = write_side_effect

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PLC_READ_AND_WRITE,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={
            "plc_ip": "192.168.1.10",
            "tags_to_read": ["good_read", "bad_read"],
            "tags_to_write": {"good_write": 1, "bad_write": 2},
        }
    )

    assert len(result) == 1
    plc_results = result[0]["plc_results"]
    assert plc_results[0]["read"]["good_read"] == 100
    assert plc_results[0]["read"]["bad_read"] == "ReadFailure"
    assert plc_results[0]["write"]["good_write"] == "WriteSuccess"
    assert plc_results[0]["write"]["bad_write"] == "WriteFailure"
