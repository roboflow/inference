import numpy as np
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


PER_CLASS_CONFIDENCE_FILTER_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
        {"type": "WorkflowParameter", "name": "class_thresholds"},
        {"type": "WorkflowParameter", "name": "default_threshold", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "roboflow_core/per_class_confidence_filter@v1",
            "name": "filter",
            "predictions": "$inputs.predictions",
            "class_thresholds": "$inputs.class_thresholds",
            "default_threshold": "$inputs.default_threshold",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "filtered",
            "selector": "$steps.filter.predictions",
        }
    ],
}


def _make_detections(
    class_names: list[str], confidences: list[float]
) -> sv.Detections:
    n = len(class_names)
    return sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * n, dtype=np.float64),
        class_id=np.arange(n),
        confidence=np.array(confidences, dtype=np.float64),
        data={
            "class_name": np.array(class_names),
            "detection_id": np.array([f"d{i}" for i in range(n)]),
        },
    )


def test_per_class_confidence_filter_end_to_end(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PER_CLASS_CONFIDENCE_FILTER_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    predictions = _make_detections(
        class_names=["person", "person", "car", "dog"],
        confidences=[0.99, 0.7, 0.6, 0.4],
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "predictions": [predictions],
            "class_thresholds": {"person": 0.98, "car": 0.5},
            "default_threshold": 0.5,
        }
    )

    # then
    assert isinstance(result, list)
    assert len(result) == 1
    filtered: sv.Detections = result[0]["filtered"]
    assert list(filtered.data["class_name"]) == ["person", "car"]
    assert list(filtered.confidence) == [0.99, 0.6]


def test_per_class_confidence_filter_default_threshold_filters_unknown_class(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PER_CLASS_CONFIDENCE_FILTER_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    predictions = _make_detections(
        class_names=["cat", "cat"],
        confidences=[0.2, 0.8],
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "predictions": [predictions],
            "class_thresholds": {"person": 0.98},
            "default_threshold": 0.5,
        }
    )

    # then
    filtered: sv.Detections = result[0]["filtered"]
    assert list(filtered.data["class_name"]) == ["cat"]
    assert list(filtered.confidence) == [0.8]


def test_per_class_confidence_filter_handles_batch_of_images(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PER_CLASS_CONFIDENCE_FILTER_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    predictions_image_1 = _make_detections(
        class_names=["person", "car"], confidences=[0.99, 0.4]
    )
    predictions_image_2 = _make_detections(
        class_names=["car"], confidences=[0.55]
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "predictions": [predictions_image_1, predictions_image_2],
            "class_thresholds": {"person": 0.98, "car": 0.5},
            "default_threshold": 0.3,
        }
    )

    # then
    assert len(result) == 2
    assert list(result[0]["filtered"].data["class_name"]) == ["person"]
    assert list(result[1]["filtered"].data["class_name"]) == ["car"]
