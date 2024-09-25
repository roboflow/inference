import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

CLIP_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ClipComparison",
            "name": "comparison",
            "images": "$inputs.image",
            "texts": "$inputs.reference",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "similarity",
            "selector": "$steps.comparison.similarity",
        },
    ],
}


@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow with CLIP model",
    use_case_description="""
This is the basic workflow that only contains a single CLIP model block. 

Please take a look at how batch-oriented WorkflowImage data is plugged to 
detection step via input selector (`$inputs.image`) and how non-batch parameters 
(reference set of texts that the each image in batch will be compared to)
is dynamically specified - via `$inputs.reference` selector.
    """,
    workflow_definition=CLIP_WORKFLOW,
    workflow_name_in_app="clip",
)
def test_clip_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "reference": ["car", "crowd"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "similarity",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "similarity",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["similarity"]) == 2
    ), "Expected 2 elements of similarity comparison list for first image"
    assert (
        result[0]["similarity"][0] > result[0]["similarity"][1]
    ), "Expected to predict `car` class for first image"
    assert (
        len(result[1]["similarity"]) == 2
    ), "Expected 2 elements of similarity comparison list for second image"
    assert (
        result[1]["similarity"][0] < result[1]["similarity"][1]
    ), "Expected to predict `crowd` class for second image"


WORKFLOW_WITH_CLIP_COMPARISON_V2 = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
        {"type": "WorkflowParameter", "name": "version", "default_value": "ViT-B-16"},
    ],
    "steps": [
        {
            "type": "roboflow_core/clip_comparison@v2",
            "name": "comparison",
            "images": "$inputs.image",
            "classes": "$inputs.reference",
            "version": "$inputs.version",
        },
        {
            "type": "PropertyDefinition",
            "name": "property_extraction",
            "data": "$steps.comparison.classification_predictions",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "clip_output",
            "selector": "$steps.comparison.*",
        },
        {
            "type": "JsonField",
            "name": "class_name",
            "selector": "$steps.property_extraction.output",
        },
    ],
}


def test_workflow_with_clip_comparison_v2_and_property_definition_with_valid_input(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLIP_COMPARISON_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "reference": ["car", "crowd"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "clip_output",
        "class_name",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "clip_output",
        "class_name",
    }, "Expected all declared outputs to be delivered"
    assert np.allclose(
        result[0]["clip_output"]["similarities"],
        [0.23334351181983948, 0.17259158194065094],
        atol=1e-4,
    ), "Expected predicted similarities to match values verified at test creation"
    assert (
        abs(
            result[0]["clip_output"]["similarities"][0]
            - result[0]["clip_output"]["max_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        abs(
            result[0]["clip_output"]["similarities"][1]
            - result[0]["clip_output"]["min_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        result[0]["clip_output"]["most_similar_class"] == "car"
    ), "Expected most similar class to be extracted properly"
    assert (
        result[0]["clip_output"]["least_similar_class"] == "crowd"
    ), "Expected least similar class to be extracted properly"
    assert (
        result[0]["clip_output"]["classification_predictions"]["top"] == "car"
    ), "Expected classifier output to be shaped correctly"
    assert (
        result[0]["class_name"] == "car"
    ), "Expected property definition step to cooperate nicely with clip output"
    assert np.allclose(
        result[1]["clip_output"]["similarities"],
        [0.18426208198070526, 0.207647442817688],
        atol=1e-4,
    ), "Expected predicted similarities to match values verified at test creation"
    assert (
        abs(
            result[1]["clip_output"]["similarities"][1]
            - result[1]["clip_output"]["max_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        abs(
            result[1]["clip_output"]["similarities"][0]
            - result[1]["clip_output"]["min_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        result[1]["clip_output"]["most_similar_class"] == "crowd"
    ), "Expected most similar class to be extracted properly"
    assert (
        result[1]["clip_output"]["least_similar_class"] == "car"
    ), "Expected least similar class to be extracted properly"
    assert (
        result[1]["clip_output"]["classification_predictions"]["top"] == "crowd"
    ), "Expected classifier output to be shaped correctly"
    assert (
        result[1]["class_name"] == "crowd"
    ), "Expected property definition step to cooperate nicely with clip output"


def test_workflow_with_clip_comparison_v2_and_property_definition_with_empty_class_list(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLIP_COMPARISON_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": [license_plate_image, crowd_image],
                "reference": [],
            }
        )


def test_workflow_with_clip_comparison_v2_and_property_definition_with_invalid_model_version(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLIP_COMPARISON_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": [license_plate_image, crowd_image],
                "reference": ["car", "crowd"],
                "version": "invalid",
            }
        )
