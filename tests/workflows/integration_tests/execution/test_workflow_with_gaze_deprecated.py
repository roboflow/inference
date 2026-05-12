"""End-to-end deprecation contract for the gaze workflow.

Replaces the prior `test_workflow_with_gaze.py` integration test: the
underlying L2CS-Net + MediaPipe path is gone, so the only behavioural
guarantee left is that a workflow that references `roboflow_core/gaze@v1`
still compiles and that executing it propagates `FeatureDeprecatedError`
through the executor middleware as `ClientCausedStepExecutionError`
with `status_code=410`.
"""

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.exceptions import FeatureDeprecatedError
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import ClientCausedStepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine

GAZE_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "do_run_face_detection",
            "default_value": True,
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/gaze@v1",
            "name": "gaze",
            "images": "$inputs.image",
            "do_run_face_detection": "$inputs.do_run_face_detection",
        },
        {
            "type": "roboflow_core/keypoint_visualization@v1",
            "name": "visualization",
            "predictions": "$steps.gaze.face_predictions",
            "image": "$inputs.image",
            "annotator_type": "vertex",
            "color": "#A351FB",
            "text_color": "black",
            "text_scale": 0.5,
            "text_thickness": 1,
            "text_padding": 10,
            "thickness": 2,
            "radius": 10,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "face_predictions",
            "selector": "$steps.gaze.face_predictions",
        },
        {
            "type": "JsonField",
            "name": "yaw_degrees",
            "selector": "$steps.gaze.yaw_degrees",
        },
        {
            "type": "JsonField",
            "name": "pitch_degrees",
            "selector": "$steps.gaze.pitch_degrees",
        },
        {
            "type": "JsonField",
            "name": "visualization",
            "selector": "$steps.visualization.image",
        },
    ],
}


def test_gaze_workflow_compiles_and_raises_410_via_executor_middleware(
    model_manager: ModelManager,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=GAZE_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    # when / then
    with pytest.raises(ClientCausedStepExecutionError) as captured:
        execution_engine.run(
            runtime_parameters={
                "image": [image],
                "do_run_face_detection": True,
            }
        )

    assert captured.value.status_code == 410
    assert isinstance(captured.value.inner_error, FeatureDeprecatedError)
    assert captured.value.inner_error.feature == "roboflow_core/gaze@v1"
