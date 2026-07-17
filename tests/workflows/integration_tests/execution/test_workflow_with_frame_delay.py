from datetime import datetime

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata

TIME_TRAVEL_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "frame_value",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractFrameMetadata", "property_name": "frame_number"}
            ],
        },
        {
            "type": "roboflow_core/frame_delay@v1",
            "name": "past",
            "image": "$inputs.image",
            "data": "$steps.frame_value.output",
            "offset": -1,
        },
        {
            "type": "roboflow_core/frame_delay@v1",
            "name": "past_two",
            "image": "$inputs.image",
            "data": "$steps.frame_value.output",
            "offset": -2,
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "past_output", "selector": "$steps.past.output"},
        {
            "type": "JsonField",
            "name": "past_available",
            "selector": "$steps.past.is_available",
        },
        {
            "type": "JsonField",
            "name": "past_ref",
            "selector": "$steps.past.reference_frame_number",
        },
        {
            "type": "JsonField",
            "name": "past_two_output",
            "selector": "$steps.past_two.output",
        },
        {
            "type": "JsonField",
            "name": "past_two_ref",
            "selector": "$steps.past_two.reference_frame_number",
        },
    ],
}


def _frame_input(frame_number: int) -> dict:
    return {
        "type": "numpy_object",
        "value": np.zeros((2, 2, 3), dtype=np.uint8),
        "video_metadata": VideoMetadata(
            video_identifier="stream",
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
            fps=30,
            comes_from_video_file=True,
        ),
    }


def test_workflow_with_frame_delay_past_offsets(
    model_manager: ModelManager,
) -> None:
    # given - a workflow with two past time-travel steps (-1 and -2), reused across a
    # sequence of frames (block state persists between engine.run() calls).
    execution_engine = ExecutionEngine.init(
        workflow_definition=TIME_TRAVEL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when - process frames 0..3; each step's `data` is that frame's frame_number
    results = [
        execution_engine.run(
            runtime_parameters={"image": [_frame_input(frame_number=n)]}
        )[0]
        for n in range(4)
    ]

    # then - past offset (-1) references the previous frame's value (its frame_number)
    assert results[0]["past_available"] is False
    assert results[0]["past_output"] is None
    assert results[0]["past_ref"] == 0
    assert results[1]["past_output"] == 0
    assert results[1]["past_available"] is True
    assert results[1]["past_ref"] == 1
    assert results[2]["past_output"] == 1
    assert results[3]["past_output"] == 2

    # then - offset (-2) references the value two frames earlier; aligned to frame N
    assert results[0]["past_two_output"] is None
    assert results[1]["past_two_output"] is None
    assert results[2]["past_two_output"] == 0
    assert results[3]["past_two_output"] == 1
    for n in range(4):
        assert results[n]["past_two_ref"] == n
