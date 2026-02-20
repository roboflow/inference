import numpy as np
import pytest
import supervision as sv

from inference.core.env import USE_INFERENCE_MODELS, WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsSortProperties,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    EvaluationEngineError,
)
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)


def build_sorting_workflow_definition(
    sort_operation_mode: DetectionsSortProperties,
    ascending: bool,
) -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.75},
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
                "name": "sorting",
                "predictions": "$steps.detection.predictions",
                "operations": [
                    {
                        "type": "SortDetections",
                        "mode": sort_operation_mode.value,
                        "ascending": ascending,
                    }
                ],
                "operations_parameters": {
                    "image": "$inputs.image",
                    "classes": "$inputs.classes",
                },
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.sorting.*"}
        ],
    }


def test_sorting_workflow_for_when_nothing_to_sort(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.CONFIDENCE,
        ascending=True,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "classes": {"person"},
            "confidence": 0.999,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    detections: sv.Detections = result[0]["result"]["predictions"]
    assert len(detections) == 0, "Expected nothing to pass confidence threshold"


@add_to_workflows_gallery(
    category="Workflows with data transformations",
    use_case_title="Workflow with detections sorting",
    use_case_description="""
This workflow presents how to use Detections Transformation block that is going to 
align predictions from object detection model such that results are sorted 
ascending regarding confidence.
    """,
    workflow_definition=build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.CONFIDENCE,
        ascending=True,
    ),
    workflow_name_in_app="detections-sorting",
)
def test_sorting_workflow_for_confidence_ascending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.CONFIDENCE,
        ascending=True,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.confidence,
        np.array([0.75794, 0.75804, 0.80455, 0.81555, 0.83957, 0.84284]),
        atol=0.01,
    ), "Expected alignment of confidences to be as requested"


def test_sorting_workflow_for_x_min_descending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.X_MIN,
        ascending=False,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.xyxy[:, 0],
        np.array([552, 521, 416, 271, 180, 113]),
        atol=1,
    ), "Expected alignment of boxes min_x to be as requested"


def test_sorting_workflow_for_x_max_ascending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.X_MAX,
        ascending=True,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.xyxy[:, 2],
        np.array([145, 244, 328, 457, 555, 598]),
        atol=1,
    ), "Expected alignment of boxes max_x to be as requested"


def test_sorting_workflow_for_y_min_ascending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.Y_MIN,
        ascending=True,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.xyxy[:, 1],
        np.array([257, 258, 259, 266, 269, 273]),
        atol=1,
    ), "Expected alignment of boxes min_y to be as requested"


def test_sorting_workflow_for_y_max_descending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.Y_MAX,
        ascending=False,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.xyxy[:, 3],
        np.array([383, 383, 365, 365, 360, 347]),
        atol=1,
    ), "Expected alignment of boxes max_y to be as requested"


def test_sorting_workflow_for_size_descending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.SIZE,
        ascending=False,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
    if not USE_INFERENCE_MODELS:
        assert np.allclose(
            detections.box_area,
            np.array([7040, 6669, 4876, 4387, 3502, 2496]),
            atol=1,
        ), "Expected alignment of boxes size to be as requested"
    else:
        assert np.allclose(
            detections.box_area,
            np.array([7104, 6669, 4830, 4346, 3502, 2496]),
            atol=1,
        ), "Expected alignment of boxes size to be as requested"


def test_sorting_workflow_for_center_x_descending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.CENTER_X,
        ascending=False,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.xyxy[:, 0] + (detections.xyxy[:, 2] - detections.xyxy[:, 0]) / 2,
        np.array([575, 538, 436.5, 299.5, 212, 129]),
        atol=1,
    ), "Expected alignment of boxes centers to be as requested"


def test_sorting_workflow_for_center_y_ascending(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = build_sorting_workflow_definition(
        sort_operation_mode=DetectionsSortProperties.CENTER_Y,
        ascending=True,
    )
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
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
        detections.xyxy[:, 1] + (detections.xyxy[:, 3] - detections.xyxy[:, 1]) / 2,
        np.array([308, 308.5, 311.5, 312, 324.5, 328]),
        atol=1,
    ), "Expected alignment of boxes centers to be as requested"
