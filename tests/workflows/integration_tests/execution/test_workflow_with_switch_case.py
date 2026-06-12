import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

SWITCH_CASE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "routing_value"},
    ],
    "steps": [
        {
            "type": "roboflow_core/switch_case@v1",
            "name": "switch",
            "value": "$inputs.routing_value",
            "cases": {
                "a": "$steps.on_a",
                "b": "$steps.on_b",
            },
            "default_next_steps": ["$steps.on_default"],
        },
        {
            "type": "Expression",
            "name": "on_a",
            "data": {"value": "$inputs.routing_value"},
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {"type": "StaticCaseResult", "value": "EXECUTED A!"},
            },
        },
        {
            "type": "Expression",
            "name": "on_b",
            "data": {"value": "$inputs.routing_value"},
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {"type": "StaticCaseResult", "value": "EXECUTED B!"},
            },
        },
        {
            "type": "Expression",
            "name": "on_default",
            "data": {"value": "$inputs.routing_value"},
            "switch": {
                "type": "CasesDefinition",
                "cases": [],
                "default": {"type": "StaticCaseResult", "value": "EXECUTED DEFAULT!"},
            },
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "on_a", "selector": "$steps.on_a.output"},
        {"type": "JsonField", "name": "on_b", "selector": "$steps.on_b.output"},
        {
            "type": "JsonField",
            "name": "on_default",
            "selector": "$steps.on_default.output",
        },
    ],
}


def _init_engine(workflow_definition: dict) -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={
            "workflows_core.model_manager": None,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )


@pytest.mark.parametrize(
    "routing_value, expected",
    [
        ("a", {"on_a": "EXECUTED A!", "on_b": None, "on_default": None}),
        ("b", {"on_a": None, "on_b": "EXECUTED B!", "on_default": None}),
        ("c", {"on_a": None, "on_b": None, "on_default": "EXECUTED DEFAULT!"}),
    ],
    ids=["case_a_matches", "case_b_matches", "no_match_routes_to_default"],
)
def test_switch_case_routes_to_expected_branch(
    routing_value: str, expected: dict
) -> None:
    # given
    execution_engine = _init_engine(SWITCH_CASE_WORKFLOW)

    # when
    result = execution_engine.run(
        runtime_parameters={"routing_value": routing_value},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single (non-batch) result expected"
    assert result[0] == expected, (
        f"Expected only the branch matching '{routing_value}' to execute"
    )


def test_switch_case_terminates_branch_when_no_match_and_no_default() -> None:
    # given - on_default step removed entirely: without a flow-control edge it would
    # execute unconditionally as an independent step
    workflow_definition = {
        **SWITCH_CASE_WORKFLOW,
        "steps": [
            (
                {**step, "default_next_steps": []}
                if step["name"] == "switch"
                else step
            )
            for step in SWITCH_CASE_WORKFLOW["steps"]
            if step["name"] != "on_default"
        ],
        "outputs": [
            output
            for output in SWITCH_CASE_WORKFLOW["outputs"]
            if output["name"] != "on_default"
        ],
    }
    execution_engine = _init_engine(workflow_definition)

    # when
    result = execution_engine.run(
        runtime_parameters={"routing_value": "unmatched"},
    )

    # then
    assert result == [
        {"on_a": None, "on_b": None}
    ], "Expected all case branches terminated when nothing matches and no default is set"


def test_switch_case_compilation_fails_on_duplicated_targets() -> None:
    # given
    workflow_definition = {
        **SWITCH_CASE_WORKFLOW,
        "steps": [
            (
                {**step, "cases": {"a": "$steps.on_a", "alias_of_a": "$steps.on_a"}}
                if step["name"] == "switch"
                else step
            )
            for step in SWITCH_CASE_WORKFLOW["steps"]
        ],
    }

    # when / then
    with pytest.raises(WorkflowSyntaxError) as error:
        _init_engine(workflow_definition)
    assert "targeted at most once" in str(error.value.inner_error)
