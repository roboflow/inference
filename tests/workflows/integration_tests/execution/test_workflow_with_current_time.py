import datetime

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_CURRENT_TIME = {
    "version": "1.0",
    "inputs": [
        {
            "type": "WorkflowParameter",
            "name": "timezone",
            "default_value": "UTC",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/current_time@v1",
            "name": "now",
            "timezone": "$inputs.timezone",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "timestamp", "selector": "$steps.now.timestamp"},
        {
            "type": "JsonField",
            "name": "iso_string",
            "selector": "$steps.now.iso_string",
        },
        {"type": "JsonField", "name": "date", "selector": "$steps.now.date"},
        {"type": "JsonField", "name": "time", "selector": "$steps.now.time"},
    ],
}


def test_current_time_workflow(model_manager: ModelManager) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CURRENT_TIME,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"timezone": "America/New_York"})

    # then
    assert len(result) == 1, "Single image/parameter batch expected"
    row = result[0]
    assert set(row.keys()) == {"timestamp", "iso_string", "date", "time"}
    assert isinstance(row["timestamp"], datetime.datetime)
    assert row["timestamp"].tzinfo is not None, "Timestamp must be timezone-aware"
    assert str(row["timestamp"].tzinfo) == "America/New_York"
    # derived strings agree with the datetime object
    assert row["iso_string"] == row["timestamp"].isoformat()
    assert row["iso_string"].startswith(row["date"])
    assert row["date"] == row["timestamp"].strftime("%Y-%m-%d")
    assert row["time"] == row["timestamp"].strftime("%H:%M:%S")
