import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    USE_INFERENCE_MODELS,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the overlap block returns a native
# inference_models.Detections (a plain @dataclass of torch tensors) which has no
# sv `.data["class_name"]`. The sv-shaped test below is skipped when the flag is
# on; the `*_tensor_native` parity test (skipped when the flag is off) asserts the
# SAME semantic result by resolving class names from `image_metadata[class_names]`.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="overlap block returns native Detections (no .data) under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def _native_class_names(predictions) -> list:
    """Resolve a native Detections' per-detection class names from its
    `image_metadata[class_names]` map keyed by each `class_id` (the same
    resolution the overlap block performs), mirroring sv `.data["class_name"]`."""
    class_names_map = (predictions.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    class_id = predictions.class_id.detach().to("cpu").numpy()
    return [class_names_map.get(int(cid), f"class_{int(cid)}") for cid in class_id]


# check both overlap types
OVERLAP_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/overlap@v1",
            "name": "any_overlap",
            "predictions": "$steps.model.predictions",
            "overlap_class_name": "banana",
            "overlap_type": "Any Overlap",
        },
        {
            "type": "roboflow_core/overlap@v1",
            "name": "center_overlap",
            "predictions": "$steps.model.predictions",
            "overlap_class_name": "banana",
            "overlap_type": "Center Overlap",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "center_predictions",
            "coordinates_system": "own",
            "selector": "$steps.center_overlap.overlaps",
        },
        {
            "type": "JsonField",
            "name": "any_predictions",
            "coordinates_system": "own",
            "selector": "$steps.any_overlap.overlaps",
        },
    ],
}

'''
@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow with dynamic zone and perspective converter",
    use_case_description="""
In this example dynamic zone with 4 vertices is calculated from detected segmentations.
Perspective correction is applied to the input image as well as to detected segmentations based on this zone.
    """,
    workflow_definition=OVERLAP_WORKFLOW,
    workflow_name_in_app="dynamic_zone_and_perspective_converter",
)
'''


@_NUMPY_ONLY
def test_workflow_with_overlap_all(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OVERLAP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
        }
    )

    assert len(result) == 1, "One set of images provided, so one output expected"

    if not USE_INFERENCE_MODELS:
        # if overlap_type is "Any Overlap", both the apples and orange will overlap the banana
        any_redictions = result[0]["any_predictions"]
        assert len(any_redictions.class_id) == 4
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" in class_names
        assert "orange" in class_names

        # if overlap_type is "Center Overlap" only the orange will overlap the banana
        any_redictions = result[0]["center_predictions"]
        assert len(any_redictions.class_id) == 1
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" not in class_names
        assert "orange" in class_names
    else:
        # if overlap_type is "Any Overlap", both the apples and orange will overlap the banana
        any_redictions = result[0]["any_predictions"]
        assert len(any_redictions.class_id) == 5
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" in class_names
        assert "orange" in class_names

        # if overlap_type is "Center Overlap" only the orange will overlap the apple and orange
        any_redictions = result[0]["center_predictions"]
        assert len(any_redictions.class_id) == 2
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" in class_names
        assert "orange" in class_names


@_TENSOR_ONLY
def test_workflow_with_overlap_all_tensor_native(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OVERLAP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
        }
    )

    # then
    assert len(result) == 1, "One set of images provided, so one output expected"

    # The overlap block returns a native inference_models.Detections (torch tensors,
    # no sv `.data`); class names are resolved from `image_metadata[class_names]`.
    # Same semantic result as the sv `test_workflow_with_overlap_all` else-branch.

    # if overlap_type is "Any Overlap", both the apples and orange will overlap the banana
    any_redictions = result[0]["any_predictions"]
    assert len(any_redictions.class_id) == 5
    class_names = _native_class_names(any_redictions)
    assert "banana" not in class_names
    assert "apple" in class_names
    assert "orange" in class_names

    # if overlap_type is "Center Overlap" the apple and orange will overlap
    any_redictions = result[0]["center_predictions"]
    assert len(any_redictions.class_id) == 2
    class_names = _native_class_names(any_redictions)
    assert "banana" not in class_names
    assert "apple" in class_names
    assert "orange" in class_names


@_TENSOR_ONLY
def test_workflow_with_overlap_all_with_tensor_input(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # Same as test_workflow_with_overlap_all_tensor_native, but the image arrives ALREADY
    # materialised as a CHW RGB device tensor (is_tensor_materialised() == True), so the
    # OD block runs its on-device tensor path.
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OVERLAP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": numpy_image_as_tensor(fruit_image),
        }
    )

    # then
    assert len(result) == 1, "One set of images provided, so one output expected"

    # The overlap block returns a native inference_models.Detections (torch tensors,
    # no sv `.data`); class names are resolved from `image_metadata[class_names]`.
    # Same semantic result as the sv `test_workflow_with_overlap_all` else-branch.

    # if overlap_type is "Any Overlap", both the apples and orange will overlap the banana
    any_redictions = result[0]["any_predictions"]
    assert len(any_redictions.class_id) == 5
    class_names = _native_class_names(any_redictions)
    assert "banana" not in class_names
    assert "apple" in class_names
    assert "orange" in class_names

    # if overlap_type is "Center Overlap" the apple and orange will overlap
    any_redictions = result[0]["center_predictions"]
    assert len(any_redictions.class_id) == 2
    class_names = _native_class_names(any_redictions)
    assert "banana" not in class_names
    assert "apple" in class_names
    assert "orange" in class_names
