import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import Detections as NativeDetections

# Under ENABLE_TENSOR_DATA_REPRESENTATION the object_detection_prediction deserializer
# accepts only a native inference_models.Detections (or a serialised dict), not a raw
# sv.Detections, and the per_class_confidence_filter block emits a native Detections
# instead of sv.Detections. The numpy-shaped tests below (which feed sv.Detections and
# assert sv attributes like .data / .confidence) run only with the flag off; an
# equivalent *_tensor_native test asserts the same semantic filtering result against the
# native carrier with the flag on.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections input/output; native under ENABLE_TENSOR_DATA_REPRESENTATION "
    "— see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

PER_CLASS_CONFIDENCE_FILTER_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
        {"type": "WorkflowParameter", "name": "class_thresholds"},
        {
            "type": "WorkflowParameter",
            "name": "default_threshold",
            "default_value": 0.3,
        },
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


def _make_detections(class_names: list[str], confidences: list[float]) -> sv.Detections:
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


def _make_native_detections(
    class_names: list[str], confidences: list[float]
) -> NativeDetections:
    """Native inference_models.Detections equivalent of ``_make_detections``.

    Per-box class name is carried both via the canonical
    ``image_metadata[CLASS_NAMES_KEY]`` (class_id -> name) map and via
    ``bboxes_metadata[i][CLASS_NAME_KEY]`` — the two sources the block's
    ``_resolve_class_names`` reads (native equivalents of sv.Detections.data).
    """
    n = len(class_names)
    name_to_id: dict[str, int] = {}
    for name in class_names:
        name_to_id.setdefault(name, len(name_to_id))
    class_id = torch.tensor(
        [name_to_id[name] for name in class_names], dtype=torch.int64
    )
    name_by_class_id = {class_id: name for name, class_id in name_to_id.items()}
    return NativeDetections(
        xyxy=torch.tensor([[0, 0, 10, 10]] * n, dtype=torch.float32),
        class_id=class_id,
        confidence=torch.tensor(confidences, dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: name_by_class_id},
        bboxes_metadata=[
            {CLASS_NAME_KEY: class_names[i], DETECTION_ID_KEY: f"d{i}"}
            for i in range(n)
        ],
    )


def _native_class_names(detections: NativeDetections) -> list[str]:
    name_by_class_id = detections.image_metadata.get(CLASS_NAMES_KEY) or {}
    return [
        name_by_class_id.get(int(detections.class_id[i]))
        for i in range(int(detections.confidence.shape[0]))
    ]


@_NUMPY_ONLY
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


@_NUMPY_ONLY
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


@_NUMPY_ONLY
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
    predictions_image_2 = _make_detections(class_names=["car"], confidences=[0.55])

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


@_TENSOR_ONLY
def test_per_class_confidence_filter_end_to_end_tensor_native(
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
    predictions = _make_native_detections(
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
    filtered = result[0]["filtered"]
    assert isinstance(filtered, NativeDetections)
    assert _native_class_names(filtered) == ["person", "car"]
    assert filtered.confidence.tolist() == pytest.approx([0.99, 0.6])


@_TENSOR_ONLY
def test_per_class_confidence_filter_default_threshold_filters_unknown_class_tensor_native(
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
    predictions = _make_native_detections(
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
    filtered = result[0]["filtered"]
    assert isinstance(filtered, NativeDetections)
    assert _native_class_names(filtered) == ["cat"]
    assert filtered.confidence.tolist() == pytest.approx([0.8])


@_TENSOR_ONLY
def test_per_class_confidence_filter_handles_batch_of_images_tensor_native(
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
    predictions_image_1 = _make_native_detections(
        class_names=["person", "car"], confidences=[0.99, 0.4]
    )
    predictions_image_2 = _make_native_detections(
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
    assert _native_class_names(result[0]["filtered"]) == ["person"]
    assert _native_class_names(result[1]["filtered"]) == ["car"]
