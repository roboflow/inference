from unittest.mock import MagicMock, patch

import pytest
import requests

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

V1 = "inference.enterprise.workflows.enterprise_blocks.sinks.plc.v1"
CLIENT = "inference.enterprise.workflows.enterprise_blocks.sinks.plc.client"

# A reader feeds a writer: the writer's `value` is a step-output selector
# (`$steps.plc_reader.error_status`) and its `tag` is an input selector. This exercises the
# step->step graph dependency, per-value selector resolution, both blocks' output
# registration, and workflow-output materialization end to end.
WORKFLOW_PLC_READER_TO_WRITER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "tag_to_write"},
    ],
    "steps": [
        {
            "type": "roboflow_core/plc_reader@v1",
            "name": "plc_reader",
            "tags_to_read": ["camera_ready"],
        },
        {
            "type": "roboflow_core/plc_writer@v1",
            "name": "plc_writer",
            "tag": "$inputs.tag_to_write",
            "value": "$steps.plc_reader.error_status",
            "depends_on": "$steps.plc_reader.tag_values",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "reader", "selector": "$steps.plc_reader.*"},
        {"type": "JsonField", "name": "writer", "selector": "$steps.plc_writer.*"},
    ],
}


def _http_response(json_body):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = json_body
    resp.text = ""
    return resp


def _relay_post(url, **kwargs):
    """Stand in for the PLC Relay: a successful batch read and batch write."""
    if url.endswith("/read_batch"):
        return _http_response(
            {"tags": [{"name": "camera_ready", "value": False}], "count": 1}
        )
    return _http_response(
        {"results": [{"name": "camera_fault", "success": True}], "success_count": 1}
    )


@pytest.mark.timeout(30)
def test_workflow_with_plc_reader_feeding_writer() -> None:
    # given - mock the relay HTTP layer so the workflow compiles/executes without a PLC
    session = MagicMock()
    session.post.side_effect = _relay_post
    with patch(f"{V1}.requests") as v1_requests, patch(
        f"{CLIENT}.requests"
    ) as client_requests:
        v1_requests.Session.return_value = session
        client_requests.exceptions = requests.exceptions

        execution_engine = ExecutionEngine.init(
            workflow_definition=WORKFLOW_PLC_READER_TO_WRITER,
            init_parameters={},
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )

        # when
        result = execution_engine.run(
            runtime_parameters={"tag_to_write": "camera_fault"}
        )

    # then - both declared outputs are materialized with the expected shapes
    assert len(result) == 1
    assert set(result[0].keys()) == {"reader", "writer"}
    assert result[0]["reader"] == {
        "tag_values": {"camera_ready": False},
        "error_status": False,
    }
    assert result[0]["writer"] == {
        "write_result": "WriteSuccess",
        "error_status": False,
    }

    # the writer's value selector resolved to the reader's error_status (False) and was sent
    write_call = [
        c for c in session.post.call_args_list if c.args[0].endswith("/write_batch")
    ][0]
    assert write_call.kwargs["json"] == {
        "writes": [{"name": "camera_fault", "value": False}]
    }
