from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import (
    ControlFlowDefinitionError,
    StepInputLineageError,
)
from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow

WORKFLOW_WITH_LINEAGE_CONFLICT_IN_FLOW_CONTROL = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_1",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "Crop",
            "name": "crop_1",
            "image": "$inputs.image",
            "predictions": "$steps.detection_1.predictions",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_2",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "Crop",
            "name": "crop_2",
            "image": "$inputs.image",
            "predictions": "$steps.detection_2.predictions",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_3",
            "image": "$steps.crop_2.crops",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "ContinueIf",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "prediction",
                            "operations": [{"type": "SequenceLength"}],
                        },
                        "comparator": {"type": "(Number) =="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 1,
                        },
                    }
                ],
            },
            "evaluation_parameters": {"prediction": "$steps.detection_3.predictions"},
            "next_steps": ["$steps.breds_classification"],
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.crop_1.crops",
            "model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "breds_classification",
            "selector": "$steps.breds_classification.predictions",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_workflow_where_control_flow_block_causes_lineage_issue(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    """
    In this test case we have
            - detection - crop - (*) classification
    image
            - detection - crop - detection - continue-if - (*)
    making batch-oriented control-flow decision in "continue-if" step based on data with different
    lineage that the forward-link in "continue-if" points into.
    That situation is prevented by Execution error as would end up in never executing classification
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_dimensionality_manipulation_blocks"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(ControlFlowDefinitionError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_LINEAGE_CONFLICT_IN_FLOW_CONTROL,
            init_parameters=workflow_init_parameters,
        )


WORKFLOW_WITH_FUSION_BLOCK_COLLAPSING_DIFFERENT_LINEAGES = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_1",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "Crop",
            "name": "crop_1",
            "image": "$inputs.image",
            "predictions": "$steps.detection_1.predictions",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_2",
            "image": "$steps.crop_1.crops",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_3",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "Crop",
            "name": "crop_2",
            "image": "$inputs.image",
            "predictions": "$steps.detection_3.predictions",
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_4",
            "image": "$steps.crop_2.crops",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "DetectionsConsensus",
            "name": "consensus",
            "predictions_batches": [
                "$steps.detection_2.predictions",
                "$steps.detection_4.predictions",
            ],
            "required_votes": 1,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "consensus",
            "selector": "$steps.consensus.predictions",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_compilation_of_workflow_where_fusion_block_collapses_different_lineages(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    """
    In this test case we have
            - detection - crop - detection - (*) \
    image                                           (*) - detection-consensus
            - detection - crop - detection - (*) /
    making detection-consensus operating on two dynamic crops which could
    end up in ambiguous results, as there is no guarantee on the size of crops batch
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.compilation.stub_plugins.plugin_with_dimensionality_manipulation_blocks"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # when
    with pytest.raises(StepInputLineageError):
        _ = compile_workflow(
            workflow_definition=WORKFLOW_WITH_FUSION_BLOCK_COLLAPSING_DIFFERENT_LINEAGES,
            init_parameters=workflow_init_parameters,
        )
