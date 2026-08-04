import cv2
import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.classical_cv.auto_rotate_on_edges.v1 import (
    build_auto_rotate_matrix,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

AUTO_ROTATE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "auto_rotate",
            "image": "$inputs.image",
            "target_orientation": "vertical",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "rotated_image",
            "coordinates_system": "own",
            "selector": "$steps.auto_rotate.image",
        },
        {
            "type": "JsonField",
            "name": "angle",
            "coordinates_system": "own",
            "selector": "$steps.auto_rotate.angle",
        },
    ],
}


def _make_vertical_stripes(
    width: int = 800, height: int = 600, spacing: int = 40, thickness: int = 3
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(0, width, spacing):
        cv2.rectangle(image, (x, 0), (x + thickness, height), (255, 255, 255), -1)
    return image


def _pre_rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    rotation_matrix = build_auto_rotate_matrix(width, height, angle_degrees)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow automatically straightening a skewed image",
    use_case_description="""
This example shows how the Auto Rotate on Edges block detects the rotation
angle at which the image's dominant straight lines become vertical and applies
that rotation, outputting both the straightened image and the applied angle.
    """,
    workflow_definition=AUTO_ROTATE_WORKFLOW,
    workflow_name_in_app="auto-rotate-on-edges",
)
def test_auto_rotate_workflow_when_skewed_input_provided(
    model_manager: ModelManager,
) -> None:
    # given - vertical stripes pre-rotated by +7 degrees
    skewed_image = _pre_rotate(_make_vertical_stripes(), 7.0)
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=AUTO_ROTATE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": skewed_image})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "rotated_image",
        "angle",
    }, "Expected all declared outputs to be delivered"
    angle = result[0]["angle"]
    assert isinstance(angle, float), "Expected angle output to be a float"
    assert abs(angle + 7.0) <= 0.5, "Expected recovered angle to cancel the +7 skew"
    np_image = result[0]["rotated_image"].numpy_image
    in_height, in_width = skewed_image.shape[:2]
    matrix = build_auto_rotate_matrix(in_width, in_height, angle)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    expected_width = int((in_height * sin) + (in_width * cos))
    expected_height = int((in_height * cos) + (in_width * sin))
    assert np_image.shape[1] == expected_width, "Expected canvas-expanded width"
    assert np_image.shape[0] == expected_height, "Expected canvas-expanded height"


def test_auto_rotate_workflow_when_straight_input_provided(
    model_manager: ModelManager,
) -> None:
    # given - already-straight stripes: identity passthrough expected
    straight_image = _make_vertical_stripes()
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=AUTO_ROTATE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": straight_image})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "rotated_image",
        "angle",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["angle"] == 0.0, "Expected identity passthrough angle"
    np_image = result[0]["rotated_image"].numpy_image
    assert (
        np_image.shape[:2] == straight_image.shape[:2]
    ), "Expected unchanged dimensions on identity passthrough"
