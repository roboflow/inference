import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_SIFT = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceImage", "name": "template"},
    ],
    "steps": [
        {
            "type": "roboflow_core/sift@v1",
            "name": "image_sift",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/sift@v1",
            "name": "template_sift",
            "image": "$inputs.template",
        },
        {
            "type": "roboflow_core/sift_comparison@v1",
            "name": "sift_comparison",
            "descriptor_1": "$steps.image_sift.descriptors",
            "descriptor_2": "$steps.template_sift.descriptors",
            "good_matches_threshold": 50,
        },
        {
            "type": "roboflow_core/first_non_empty_or_default@v1",
            "name": "empty_values_replacement",
            "data": ["$steps.sift_comparison.images_match"],
            "default": False,
        },
        {
            "type": "roboflow_core/expression@v1",
            "name": "is_match_expression",
            "data": {
                "images_match": "$steps.empty_values_replacement.output",
            },
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": {
                            "type": "StatementGroup",
                            "statements": [
                                {
                                    "type": "UnaryStatement",
                                    "operand": {
                                        "type": "DynamicOperand",
                                        "operand_name": "images_match",
                                    },
                                    "operator": {"type": "(Boolean) is True"},
                                }
                            ],
                        },
                        "result": {"type": "StaticCaseResult", "value": "MATCH"},
                    },
                ],
                "default": {"type": "StaticCaseResult", "value": "NO MATCH"},
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "coordinates_system": "own",
            "selector": "$steps.is_match_expression.output",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="SIFT in Workflows",
    use_case_description="""
In this example we check how SIFT-based pattern matching works in cooperation
with expression block.

The Workflow first calculates SIFT features for input image and reference template, 
then image features are compared to template features. At the end - switch-case 
statement is built with Expression block to produce output. 

Important detail: If there is empty output from SIFT descriptors calculation
for (which is a valid output if no feature gets recognised) the sift comparison won't 
execute - hence First Non Empty Or Default block is used to provide default outcome 
for `images_match` output of SIFT comparison block.

Please note that a single image can be passed as template, and batch of images
are passed as images to look for template. This workflow does also validate
Execution Engine capabilities to broadcast batch-oriented inputs properly.
    """,
    workflow_definition=WORKFLOW_WITH_SIFT,
    workflow_name_in_app="sift",
)
def test_workflow_with_classical_pattern_matching(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    template = np.ascontiguousarray(dogs_image[::-1, ::-1], dtype=np.uint8)
    empty_image_without_descriptors = np.zeros((256, 256, 3), dtype=np.uint8)
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SIFT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, dogs_image, empty_image_without_descriptors],
            "template": template,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 3, "Three images provided, so three outputs expected"
    assert set(result[0].keys()) == {
        "result",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "result",
    }, "Expected all declared outputs to be delivered"
    assert set(result[2].keys()) == {
        "result",
    }, "Expected all declared outputs to be delivered"
    assert (
        result[0]["result"] == "NO MATCH"
    ), "Expected first image not to match with template"
    assert (
        result[1]["result"] == "MATCH"
    ), "Expected second image not to match with template"
    assert (
        result[2]["result"] == "NO MATCH"
    ), "Expected third image not to match with template"
