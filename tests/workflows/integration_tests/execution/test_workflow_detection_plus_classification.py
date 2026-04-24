import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

LEGACY_DETECTION_PLUS_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
            "confidence": 0.09,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.breds_classification.predictions",
        },
    ],
}


def test_legacy_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=LEGACY_DETECTION_PLUS_CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
    assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"


# Matched pairs only — OD and classification blocks are bumped together.
DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS = [
    (
        "roboflow_core/roboflow_object_detection_model@v2",
        "roboflow_core/roboflow_classification_model@v2",
    ),
    (
        "roboflow_core/roboflow_object_detection_model@v3",
        "roboflow_core/roboflow_classification_model@v3",
    ),
]


def _build_detection_plus_classification_workflow(
    od_block_type: str, cls_block_type: str
) -> dict:
    if cls_block_type.endswith("@v3"):
        cls_step_confidence = {
            "confidence_mode": "custom",
            "custom_confidence": 0.09,
        }
    else:
        cls_step_confidence = {"confidence": 0.09}
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": od_block_type,
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "cropping",
                "image": "$inputs.image",
                "predictions": "$steps.general_detection.predictions",
            },
            {
                "type": cls_block_type,
                "name": "breds_classification",
                "image": "$steps.cropping.crops",
                "model_id": "dog-breed-xpaq6/1",
                **cls_step_confidence,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.breds_classification.predictions",
            },
        ],
    }


# Gallery registration uses the latest block pair.
DETECTION_PLUS_CLASSIFICATION_WORKFLOW_V2_BLOCKS = (
    _build_detection_plus_classification_workflow(
        *DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS[-1]
    )
)


@add_to_workflows_gallery(
    category="Workflows with multiple models",
    use_case_title="Workflow detection model followed by classifier",
    use_case_description="""
This example showcases how to stack models on top of each other - in this particular
case, we detect objects using object detection models, requesting only "dogs" bounding boxes
in the output of prediction.

Based on the model predictions, we take each bounding box with dog and apply dynamic cropping
to be able to run classification model for each and every instance of dog separately.
Please note that for each inserted image we will have nested batch of crops (with size
dynamically determined in runtime, based on first model predictions) and for each crop
we apply secondary model.

Secondary model is supposed to make prediction from dogs breed classifier model
to assign detailed class for each dog instance.
    """,
    workflow_definition=DETECTION_PLUS_CLASSIFICATION_WORKFLOW_V2_BLOCKS,
    workflow_name_in_app="detection-plus-classification",
)
@pytest.mark.parametrize(
    "od_block_type, cls_block_type", DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS
)
def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
    od_block_type: str,
    cls_block_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_detection_plus_classification_workflow(
            od_block_type, cls_block_type
        ),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
    assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"


@pytest.mark.parametrize(
    "od_block_type, cls_block_type", DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS
)
def test_detection_plus_classification_workflow_when_minimal_valid_input_provided_and_serialization_requested(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
    od_block_type: str,
    cls_block_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_detection_plus_classification_workflow(
            od_block_type, cls_block_type
        ),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        },
        serialize_results=True,
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
    assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"


@pytest.mark.parametrize(
    "od_block_type, cls_block_type", DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS
)
def test_detection_plus_classification_workflow_when_nothing_gets_predicted(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    roboflow_api_key: str,
    od_block_type: str,
    cls_block_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_detection_plus_classification_workflow(
            od_block_type, cls_block_type
        ),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 0
    ), "Expected no prediction from 2nd model, as no dogs detected"


def _build_detection_plus_classification_plus_consensus_workflow(
    od_block_type: str, cls_block_type: str
) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": od_block_type,
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "class_filter": ["dog"],
            },
            {
                "type": "roboflow_core/detections_consensus@v1",
                "name": "detections_consensus",
                "predictions_batches": [
                    "$steps.general_detection.predictions",
                ],
                "required_votes": 1,
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "cropping",
                "image": "$inputs.image",
                "predictions": "$steps.detections_consensus.predictions",
            },
            {
                "type": cls_block_type,
                "name": "breds_classification",
                "image": "$steps.cropping.crops",
                "model_id": "dog-breed-xpaq6/1",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.breds_classification.predictions",
            },
        ],
    }


@pytest.mark.parametrize(
    "od_block_type, cls_block_type", DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS
)
def test_detection_plus_classification_workflow_when_nothing_gets_predicted_and_empty_sv_detections_produced_without_metadata(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    roboflow_api_key: str,
    od_block_type: str,
    cls_block_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_detection_plus_classification_plus_consensus_workflow(
            od_block_type, cls_block_type
        ),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 0
    ), "Expected no prediction from 2nd model, as no dogs detected"


@pytest.mark.parametrize(
    "od_block_type, cls_block_type", DETECTION_PLUS_CLASSIFICATION_BLOCK_PAIRS
)
def test_detection_plus_classification_workflow_when_nothing_gets_predicted_and_serialization_requested(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    roboflow_api_key: str,
    od_block_type: str,
    cls_block_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_detection_plus_classification_plus_consensus_workflow(
            od_block_type, cls_block_type
        ),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
        },
        serialize_results=True,
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 0
    ), "Expected no prediction from 2nd model, as no dogs detected"
