import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.query_language.errors import (
    EvaluationEngineError,
)
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

FILTERING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "DetectionsTransformation",
            "name": "filtering",
            "predictions": "$steps.detection.predictions",
            "operations": [
                {
                    "type": "DetectionsFilter",
                    "filter_operation": {
                        "type": "StatementGroup",
                        "operator": "and",
                        "statements": [
                            {
                                "type": "BinaryStatement",
                                "left_operand": {
                                    "type": "DynamicOperand",
                                    "operations": [
                                        {
                                            "type": "ExtractDetectionProperty",
                                            "property_name": "class_name",
                                        }
                                    ],
                                },
                                "comparator": {"type": "in (Sequence)"},
                                "right_operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "classes",
                                },
                            },
                            {
                                "type": "BinaryStatement",
                                "left_operand": {
                                    "type": "DynamicOperand",
                                    "operations": [
                                        {
                                            "type": "ExtractDetectionProperty",
                                            "property_name": "size",
                                        },
                                    ],
                                },
                                "comparator": {"type": "(Number) >="},
                                "right_operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "image",
                                    "operations": [
                                        {
                                            "type": "ExtractImageProperty",
                                            "property_name": "size",
                                        },
                                        {"type": "Multiply", "other": 0.02},
                                    ],
                                },
                            },
                        ],
                    },
                }
            ],
            "operations_parameters": {
                "image": "$inputs.image",
                "classes": "$inputs.classes",
            },
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.filtering.*"}
    ],
}

EXPECTED_OBJECT_DETECTION_BBOXES = np.array(
    [
        [180, 273, 244, 383],
        [271, 266, 328, 383],
    ]
)
EXPECTED_OBJECT_DETECTION_CONFIDENCES = np.array(
    [
        0.84284,
        0.83957,
    ]
)


@add_to_workflows_gallery(
    category="Workflows with data transformations",
    use_case_title="Workflow with detections filtering",
    use_case_description="""
This example presents how to use Detections Transformation block to build workflow
that is going to filter predictions based on:

- predicted classes

- size of predicted bounding box relative to size of input image 
    """,
    workflow_definition=FILTERING_WORKFLOW,
    workflow_name_in_app="detections-filtering",
)
def test_filtering_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FILTERING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "classes": {"person"},
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    detections: sv.Detections = result[0]["result"]["predictions"]
    assert np.allclose(
        detections.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES,
        atol=1,
    ), "Expected bboxes to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES,
        atol=0.01,
    ), "Expected confidences to match what was validated manually as workflow outcome"


def test_filtering_workflow_when_batch_input_provided(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FILTERING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, crowd_image],
            "model_id": "yolov8n-640",
            "classes": {"person"},
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided - two outputs expected"
    detections_1: sv.Detections = result[0]["result"]["predictions"]
    detections_2: sv.Detections = result[1]["result"]["predictions"]
    assert np.allclose(
        detections_1.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES,
        atol=1,
    ), "Expected bboxes for first image to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections_1.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES,
        atol=0.01,
    ), "Expected confidences for first image to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections_2.xyxy,
        EXPECTED_OBJECT_DETECTION_BBOXES,
        atol=1,
    ), "Expected bboxes for 2nd image to match what was validated manually as workflow outcome"
    assert np.allclose(
        detections_2.confidence,
        EXPECTED_OBJECT_DETECTION_CONFIDENCES,
        atol=0.01,
    ), "Expected confidences for 2nd image to match what was validated manually as workflow outcome"


def test_filtering_workflow_when_model_id_not_provided_in_input(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FILTERING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
            }
        )


def test_filtering_workflow_when_image_not_provided_in_input(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FILTERING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "model_id": "yolov8n-640",
            }
        )


def test_filtering_workflow_when_classes_not_provided(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FILTERING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(EvaluationEngineError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "yolov8n-640",
            }
        )


def test_filtering_workflow_when_model_id_cannot_be_resolved_to_valid_model(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FILTERING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(StepExecutionError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": crowd_image,
                "model_id": "invalid",
            }
        )
