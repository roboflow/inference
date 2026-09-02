"""Engine-level smoke test for in-block VLM decoding.

The per-block unit suites (``test_anthropic_claude_v5.py`` and friends)
exercise ``run()`` directly. This module runs one family - Claude v5 - through
the real ``ExecutionEngine`` instead, because two things can only break there:

* ``get_actual_outputs`` narrows ``predictions`` to
  ``OBJECT_DETECTION_PREDICTION_KIND`` for detection tasks, and the compiler
  only validates that narrowing when a *detection consumer* (here, a
  bounding-box visualization) is wired onto the output.
* the engine must accept exactly the keys ``run()`` returns.

The remaining ten VLM blocks share ``common/vlm_decoding``, so one family is
enough to cover the wiring.
"""

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.unit_tests.core_steps._vlm_prediction_readers import (
    classification_top_class,
    classification_top_confidence,
    detection_boxes,
    detection_class_names,
    is_classification_prediction,
    is_detection_prediction,
)

EXECUTE_REQUESTS_SEAM = (
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5"
    ".execute_claude_requests"
)

# Small enough that the Anthropic upload path does not resize it, so the
# `xyxy_absolute` box below lands on the original image pixels unchanged.
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 100
EXPECTED_XYXY = [20.0, 10.0, 100.0, 50.0]
DETECTION_OUTPUT = json.dumps(
    [{"box_2d": [20, 10, 100, 50], "label": "a"}],
)
CLASSIFICATION_OUTPUT = json.dumps({"class_name": "b", "confidence": 0.75})


def _workflow_definition(task_type: str, with_visualization: bool) -> Dict[str, Any]:
    claude_step = {
        "type": "roboflow_core/anthropic_claude@v5",
        "name": "claude",
        "images": "$inputs.image",
        "task_type": task_type,
        "classes": ["a", "b"],
        "api_key": "$inputs.api_key",
    }
    if task_type == "unconstrained":
        # `unconstrained` is validated as requiring a free-form prompt.
        claude_step["prompt"] = "what is on the image?"
    steps = [claude_step]
    outputs = [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.claude.predictions",
        },
        {
            "type": "JsonField",
            "name": "error_status",
            "selector": "$steps.claude.error_status",
        },
        {
            "type": "JsonField",
            "name": "inference_id",
            "selector": "$steps.claude.inference_id",
        },
    ]
    if with_visualization:
        steps.append(
            {
                "type": "roboflow_core/bounding_box_visualization@v1",
                "name": "visualization",
                "predictions": "$steps.claude.predictions",
                "image": "$inputs.image",
            }
        )
        outputs.append(
            {
                "type": "JsonField",
                "name": "visualization",
                "selector": "$steps.visualization.image",
            }
        )
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "api_key"},
        ],
        "steps": steps,
        "outputs": outputs,
    }


def _run_workflow(definition: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    execution_engine = ExecutionEngine.init(
        workflow_definition=definition,
        init_parameters={
            "workflows_core.model_manager": MagicMock(),
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    with patch(EXECUTE_REQUESTS_SEAM) as mock_execute:
        mock_execute.return_value = [(raw_output, 11, 3)]
        result = execution_engine.run(
            runtime_parameters={
                "image": [image],
                "api_key": "sk-ant-test",
            }
        )
    assert len(result) == 1, "Single image given, expected single output"
    return result[0]


def test_object_detection_predictions_feed_a_detection_consumer() -> None:
    # given
    definition = _workflow_definition(
        task_type="object-detection", with_visualization=True
    )

    # when
    result = _run_workflow(definition=definition, raw_output=DETECTION_OUTPUT)

    # then
    assert result["error_status"] is False
    predictions = result["predictions"]
    assert is_detection_prediction(
        predictions
    ), "Detection task must deliver the detection carrier of the active representation"
    assert detection_boxes(predictions) == [EXPECTED_XYXY]
    assert detection_class_names(predictions) == ["a"]
    assert result["inference_id"] is not None
    visualization = result["visualization"]
    assert visualization.numpy_image.shape[:2] == (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ), "Visualization must render on the original image"


def test_classification_predictions_reach_workflow_output() -> None:
    # given
    definition = _workflow_definition(
        task_type="classification", with_visualization=False
    )

    # when
    result = _run_workflow(definition=definition, raw_output=CLASSIFICATION_OUTPUT)

    # then
    assert result["error_status"] is False
    predictions = result["predictions"]
    assert is_classification_prediction(predictions)
    assert classification_top_class(predictions) == "b"
    assert classification_top_confidence(predictions) == pytest.approx(0.75)


def test_non_decoding_task_returns_no_predictions() -> None:
    # given
    definition = _workflow_definition(
        task_type="unconstrained", with_visualization=False
    )

    # when
    result = _run_workflow(definition=definition, raw_output="some free-form answer")

    # then
    assert result["predictions"] is None
    assert result["error_status"] is False
