from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

ROCK_PAPER_SCISSOR_IF_ELSE_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "gestures_detection",
            "image": "$inputs.image",
            "model_id": "rock-paper-scissors-sxsw/14",
        },
        {
            "type": "Condition",
            "name": "condition",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "prediction",
                            "operations": [
                                {"type": "SequenceLength"},
                            ],
                        },
                        "comparator": {"type": "(Number) =="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 2,
                        },
                    }
                ],
            },
            "evaluation_parameters": {
                "prediction": "$steps.gestures_detection.predictions",
            },
            "steps_if_true": [
                "$steps.take_leftmost_detection",
                "$steps.take_rightmost_detection",
            ],
            "steps_if_false": ["$steps.undefined_verdict_generator"],
        },
        {
            "type": "ExpressionTestBlock",
            "name": "undefined_verdict_generator",
            "data": {
                "image_only_for_dimension": "$inputs.image",
            },
            "output": "Gestures not detected",
        },
        {
            "type": "DetectionsTransformation",
            "name": "take_leftmost_detection",
            "predictions": "$steps.gestures_detection.predictions",
            "operations": [{"type": "DetectionsSelection", "mode": "left_most"}],
        },
        {
            "type": "DetectionsTransformation",
            "name": "take_rightmost_detection",
            "predictions": "$steps.gestures_detection.predictions",
            "operations": [{"type": "DetectionsSelection", "mode": "right_most"}],
        },
        {
            "type": "ExpressionTestBlock",
            "name": "defined_verdict_generator",
            "data": {
                "left_player_detections": "$steps.take_leftmost_detection.predictions",
                "right_player_detections": "$steps.take_rightmost_detection.predictions",
            },
            "output": {
                "type": "PythonBlock",
                "code": """
def function(left_player_detections, right_player_detections):
    left_gesture, right_gesture = (
        left_player_detections["class_name"][0], right_player_detections["class_name"][0]
    )                    
    if left_gesture == right_gesture:
        return "tie"
    if left_gesture == "Rock":
        if right_gesture == "Paper":
            return "right wins"
        else:
            return "left wins"
    if left_gesture == "Paper":
        if right_gesture == "Scissors":
            return "right wins"
        else:
            return "left wins"
    if left_gesture == "Scissors":
        if right_gesture == "Rock":
            return "right wins"
        else:
            return "left wins"
    return "undefined - should never happen"
                """,
            },
        },
        {
            "type": "TakeFirstNonEmpty",
            "name": "verdict",
            "inputs": [
                "$steps.defined_verdict_generator.output",
                "$steps.undefined_verdict_generator.output",
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "verdict",
            "selector": "$steps.verdict.output",
        },
    ],
}


@pytest.mark.skip(reason="Could not find requested Roboflow resource, maybe deleted?")
@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_rock_paper_scissors_workflow(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
    crowd_image: np.ndarray,
    left_scissors_right_paper: np.ndarray,
    left_rock_right_paper: np.ndarray,
    left_rock_right_rock: np.ndarray,
    left_scissors_right_scissors: np.ndarray,
    roboflow_api_key: str,
) -> None:
    """
    In this test we created rock-paper-scissor game relying on two
    blocks from rock_paper_scissor_plugin - namely Expression (capable of executing
    arbitrary Python function on data that is referred to block or provide static
    output - yet some batch input must be provided to deduce dimensionality) and
    TakeFirstNonEmpty which is capable of collapsing execution branches

    What happens is the following:
    * we detect instances of rock / paper / scissors gesture
    * we check if there are exactly 2 instances detected
        - if not we go into block producing "Gestures not detected" verdict
        - else - we go into blocks filtering detections - taking leftmost and rightmost
            ones. Those two detections are plugged into Expression block running
            Python function to turn classes of left- and right- most BBoxes into
            verdict
    * at the end, two branches created by if-else statement collapses to one output, for
    convenience - taking first non-empty result

    What is verified from EE standpoint:
    * Creating and collapsing execution branches
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.rock_paper_scissor_plugin",
        "tests.workflows.integration_tests.execution.stub_plugins.flow_control_plugin",
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=ROCK_PAPER_SCISSOR_IF_ELSE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [
                crowd_image,
                left_scissors_right_paper,
                left_rock_right_paper,
                left_rock_right_rock,
                left_scissors_right_scissors,
            ]
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 5, "5 images provided, so 5 output elements expected"
    for element in result:
        assert set(element.keys()) == {"verdict"}, "Expected output to be registered"
    assert (
        result[0]["verdict"] == "Gestures not detected"
    ), "Expected first image not detect gestures"
    assert result[1]["verdict"] == "left wins", "Expected scissors to win over paper"
    assert result[2]["verdict"] == "right wins", "Expected paper wins over rock"
    assert result[3]["verdict"] == "tie", "Expected tie for 4th image"
    assert result[4]["verdict"] == "tie", "Expected tie for 5th image"
