from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_STRING_TEMPLATE = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "prompt_template"},
        {"type": "WorkflowParameter", "name": "expected_skus"},
    ],
    "steps": [
        {
            "type": "roboflow_core/string_template@v1",
            "name": "prompt_builder",
            "template": "$inputs.prompt_template",
            "data": {
                "sku_list": "$inputs.expected_skus",
                "fallback_answer": "NONE",
            },
            "data_operations": {
                "sku_list": [{"type": "SequenceJoin", "separator": ", "}]
            },
        },
        {
            "type": "roboflow_core/string_template@v1",
            "name": "prompt_wrapper",
            "template": "PROMPT START. {prompt} PROMPT END.",
            "data": {"prompt": "$steps.prompt_builder.output"},
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "prompt",
            "selector": "$steps.prompt_builder.output",
        },
        {
            "type": "JsonField",
            "name": "wrapped_prompt",
            "selector": "$steps.prompt_wrapper.output",
        },
    ],
}


def test_workflow_with_string_template(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_STRING_TEMPLATE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "prompt_template": "This facing contains one of: {sku_list}. "
            "Answer with the product name or {fallback_answer}.",
            "expected_skus": ["SKU-1", "SKU-2", "SKU-3"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output"
    assert set(result[0].keys()) == {
        "prompt",
        "wrapped_prompt",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["prompt"] == (
        "This facing contains one of: SKU-1, SKU-2, SKU-3. "
        "Answer with the product name or NONE."
    ), "Expected template rendered with joined SKU list and literal fallback"
    assert result[0]["wrapped_prompt"] == (
        "PROMPT START. This facing contains one of: SKU-1, SKU-2, SKU-3. "
        "Answer with the product name or NONE. PROMPT END."
    ), "Expected downstream string field to consume the STRING_KIND output"
