import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    BlockManifest,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

RESIZE_IMAGE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "resize_image",
            "image": "$inputs.image",
            "task_type": "resize",
            "width": 1000,
            "height": 800,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "resized_image",
            "coordinates_system": "own",
            "selector": "$steps.resize_image.image",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow resizing the input image",
    use_case_description="""
This example shows how the Image Preprocessing block can be used to resize an input image. 
    """,
    workflow_definition=RESIZE_IMAGE_WORKFLOW,
    workflow_name_in_app="resize-image",
)
def test_resize_image_workflow_when_valid_input_provided(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=RESIZE_IMAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": license_plate_image,  # Original dimensions are (1920 × 1280)
        }
    )
    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "resized_image",
    }, "Expected all declared outputs to be delivered"
    np_image = result[0]["resized_image"].numpy_image
    assert (
        np_image.shape[0] == 1000,
        "Expected image to be resized to a width of 1000",
    )
    assert (np_image.shape[1] == 800, "Expected image to be resized to a height of 800")


def test_upsize_image_workflow_when_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=RESIZE_IMAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )
    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "resized_image",
    }, "Expected all declared outputs to be delivered"
    np_image = result[0]["resized_image"].numpy_image
    assert (
        np_image.shape[0] == 1000,
        "Expected image to be upsized to a width of 1000",
    )
    assert (np_image.shape[1] == 800, "Expected image to be upsized to a height of 800")


def test_resize_image_workflow_when_missing_required_parameters() -> None:
    # given
    data = {
        "type": "roboflow_core/image_preprocessing@v1",
        "name": "resize",
        "image": "$inputs.image",
        "task_type": "resize",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


ROTATE_IMAGE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "rotate",
            "image": "$inputs.image",
            "task_type": "rotate",
            "rotation_degrees": 90,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "rotated_image",
            "coordinates_system": "own",
            "selector": "$steps.rotate.image",
        }
    ],
}


def test_rotate_image_workflow_when_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=ROTATE_IMAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    original_image_height, original_image_width, _ = dogs_image.shape
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "rotated_image",
    }, "Expected all declared outputs to be delivered"
    np_image = result[0]["rotated_image"].numpy_image
    assert (
        np_image.shape[0] == original_image_width
    ), "Expected image to be rotated 90 degrees"
    assert (
        np_image.shape[1] == original_image_height
    ), "Expected image to be rotated 90 degrees"


def test_rotate_image_workflow_with_dynamic_rotation_degrees(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    """Test that rotation_degrees can accept dynamic parameter references."""
    # given
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {
                "type": "WorkflowParameter",
                "name": "rotation_angle",
                "default_value": 90,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/image_preprocessing@v1",
                "name": "rotate",
                "image": "$inputs.image",
                "task_type": "rotate",
                "rotation_degrees": "$inputs.rotation_angle",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "rotated_image",
                "coordinates_system": "own",
                "selector": "$steps.rotate.image",
            }
        ],
    }

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "rotation_angle": 180,
        }
    )

    # then
    original_image_height, original_image_width, _ = dogs_image.shape
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "rotated_image",
    }, "Expected all declared outputs to be delivered"
    np_image = result[0]["rotated_image"].numpy_image
    # 180 degree rotation should preserve dimensions
    assert (
        np_image.shape[0] == original_image_height
    ), "Expected image height to be preserved after 180 degree rotation"
    assert (
        np_image.shape[1] == original_image_width
    ), "Expected image width to be preserved after 180 degree rotation"


def test_rotate_image_workflow_when_missing_required_parameters() -> None:
    # given
    data = {
        "type": "roboflow_core/image_preprocessing@v1",
        "name": "rotate",
        "image": "$inputs.image",
        "task_type": "rotate",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_rotate_image_workflow_when_invalid_task_type_provided() -> None:
    # given
    data = {
        "type": "MaskVisualization",
        "name": "mask1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "opacity": 0.5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


VERTICAL_FLIP_IMAGE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "flip",
            "image": "$inputs.image",
            "task_type": "flip",
            "flip_type": "vertical",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "flipped_image",
            "coordinates_system": "own",
            "selector": "$steps.flip.image",
        }
    ],
}


HORIZONTAL_FLIP_IMAGE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_preprocessing@v1",
            "name": "flip",
            "image": "$inputs.image",
            "task_type": "flip",
            "flip_type": "horizontal",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "flipped_image",
            "coordinates_system": "own",
            "selector": "$steps.flip.image",
        }
    ],
}


def test_flip_image_workflow_vertical_flip(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=VERTICAL_FLIP_IMAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert "flipped_image" in result[0], "Expected 'flipped_image' in the output"

    original_image = dogs_image
    flipped_image = result[0]["flipped_image"].numpy_image

    # Check dimensions
    assert (
        original_image.shape == flipped_image.shape
    ), "Image dimensions should remain the same after flipping"

    # Check a few pixels to ensure they've moved to the opposite side (vertically)
    height, width = original_image.shape[:2]
    for x in [0, width // 4, width // 2]:
        for y in [0, height // 2, height - 1]:
            np.testing.assert_array_equal(
                original_image[y, x],
                flipped_image[height - 1 - y, x],
                err_msg=f"Pixel at ({y}, {x}) did not flip correctly vertically",
            )


def test_flip_image_workflow_horizontal_flip(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=HORIZONTAL_FLIP_IMAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "flip_type": "horizontal",
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert "flipped_image" in result[0], "Expected 'flipped_image' in the output"

    original_image = dogs_image
    flipped_image = result[0]["flipped_image"].numpy_image

    # Check dimensions
    assert (
        original_image.shape == flipped_image.shape
    ), "Image dimensions should remain the same after flipping"

    # Check a few pixels to ensure they've moved to the opposite side (horizontally)
    height, width = original_image.shape[:2]
    for y in [0, height // 2, height - 1]:
        for x in [0, width // 4, width // 2]:
            np.testing.assert_array_equal(
                original_image[y, x],
                flipped_image[y, width - 1 - x],
                err_msg=f"Pixel at ({y}, {x}) did not flip correctly",
            )


def test_flip_image_workflow_when_missing_required_parameters() -> None:
    # given
    data = {
        "type": "roboflow_core/image_preprocessing@v1",
        "name": "rotate",
        "image": "$inputs.image",
        "task_type": "flip",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_flip_image_workflow_when_invalid_task_type_provided() -> None:
    # given
    data = {
        "type": "MaskVisualization",
        "name": "mask1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "opacity": 0.5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)
