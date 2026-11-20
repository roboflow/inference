"""Engine-level test of the oriented-object-detection task of the VLM blocks.

The per-block suites call ``run()`` directly. Two things only break in the real
``ExecutionEngine``: the compiler validating that ``predictions`` narrows to
``oriented_object_detection_prediction`` once a consumer of that kind is wired
onto it, and the serializer of that kind turning the carrier into the native
oriented response.
"""

import math
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.execution_engine.core import ExecutionEngine

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 100
VLM_ANSWER = (
    '{"oriented_detections": [{"label": "forklift", "cx": 100, "cy": 50, '
    '"length": 80, "width": 20, "angle_deg": 90}]}'
)
CLASS_HEADING = {"forklift": True, "scissor-lift": False}

BLOCKS = {
    "openai": {
        "step": {
            "type": "roboflow_core/open_ai@v7",
            "model_version": "gpt-6-astra",
        },
        "seam": "roboflow_workflows.core_steps.models.foundation.openai.v7"
        ".execute_openai_requests",
    },
    "gemini": {
        "step": {
            "type": "roboflow_core/google_gemini@v6",
            "model_version": "gemini-2.5-pro",
        },
        "seam": "roboflow_workflows.core_steps.models.foundation.google_gemini.v6"
        ".execute_gemini_requests",
    },
}


def _workflow_definition(block: str) -> Dict[str, Any]:
    vlm_step = {
        **BLOCKS[block]["step"],
        "name": "vlm",
        "images": "$inputs.image",
        "task_type": "oriented-object-detection",
        "classes": ["forklift", "scissor-lift"],
        "class_heading": CLASS_HEADING,
        "api_key": "$inputs.api_key",
    }
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "api_key"},
        ],
        "steps": [
            vlm_step,
            {
                "type": "roboflow_core/oriented_box_visualization@v1",
                "name": "visualization",
                "image": "$inputs.image",
                "predictions": "$steps.vlm.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.vlm.predictions",
            },
            {
                "type": "JsonField",
                "name": "error_status",
                "selector": "$steps.vlm.error_status",
            },
            {
                "type": "JsonField",
                "name": "visualization",
                "selector": "$steps.visualization.image",
            },
        ],
    }


@pytest.mark.parametrize("block", sorted(BLOCKS))
def test_oriented_predictions_feed_an_oriented_consumer_and_serialize_natively(
    block: str,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=_workflow_definition(block),
        init_parameters={
            "workflows_core.model_manager": MagicMock(),
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    # when
    with patch(BLOCKS[block]["seam"]) as mock_execute:
        mock_execute.return_value = [(VLM_ANSWER, 11, 3)]
        result = execution_engine.run(
            runtime_parameters={"image": [image], "api_key": "sk-test"},
            serialize_results=True,
        )

    # then
    assert len(result) == 1
    assert result[0]["error_status"] is False
    predictions = result[0]["predictions"]
    assert predictions["task_type"] == "oriented-object-detection"
    assert predictions["class_heading"] == CLASS_HEADING
    assert predictions["image"] == {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT}
    (row,) = predictions["predictions"]
    assert row["obb"] == pytest.approx([100.0, 50.0, 80.0, 20.0, math.pi / 2])
    assert row["class"] == "forklift"
    assert row["class_id"] == 0
    assert row["heading"] is True
    assert row["confidence"] == 1.0
    assert result[0]["visualization"] is not None
