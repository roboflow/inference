"""End-to-end test of `roboflow_core/detections_difference@v1` running inside
a compiled workflow.

Instead of chaining two `RoboflowObjectDetectionModel` steps (which would need
model weights and network access), detections are injected directly through
`WorkflowBatchInput` entries of kind `object_detection_prediction` — supported
since EE `v1.3.0`, the block's declared compatibility floor. This keeps the
test hermetic while still exercising the full compile → deserialize → run →
(optionally serialize) path:

- the inference-format detection dicts are deserialized into `sv.Detections`
  (including `detection_id` propagation),
- the block's removed / persisted / new split and `verified` gating run on
  those,
- with `serialize_results=True`, the detection outputs are serialized back to
  the wire format, proving the outputs survive the REMOTE / API boundary.
"""

from typing import List

import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

DETECTIONS_DIFFERENCE_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowBatchInput",
            "name": "reference_predictions",
            "kind": ["object_detection_prediction"],
        },
        {
            "type": "WorkflowBatchInput",
            "name": "candidate_predictions",
            "kind": ["object_detection_prediction"],
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/detections_difference@v1",
            "name": "difference",
            "reference_predictions": "$inputs.reference_predictions",
            "candidate_predictions": "$inputs.candidate_predictions",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.difference.*"}
    ],
}


def _prediction(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    class_name: str,
    class_id: int,
    detection_id: str,
) -> dict:
    return {
        "x": (x_min + x_max) / 2,
        "y": (y_min + y_max) / 2,
        "width": x_max - x_min,
        "height": y_max - y_min,
        "confidence": 0.9,
        "class": class_name,
        "class_id": class_id,
        "detection_id": detection_id,
    }


def _detections(predictions: List[dict]) -> dict:
    return {"image": {"width": 640, "height": 480}, "predictions": predictions}


_BEFORE = _detections(
    [
        _prediction(0, 0, 100, 100, "bottle", 0, "r0"),
        _prediction(200, 0, 300, 100, "can", 1, "r1"),
        _prediction(0, 200, 100, 300, "bag", 2, "r2"),
    ]
)
# after the cleanup only the can remains, re-detected slightly shifted
_AFTER = _detections([_prediction(205, 0, 305, 100, "can", 1, "c0")])


def test_detections_difference_workflow_verifies_partial_cleanup() -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_DIFFERENCE_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "reference_predictions": _BEFORE,
            "candidate_predictions": _AFTER,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected one output element for one datapoint"
    assert set(result[0].keys()) == {"result"}
    difference = result[0]["result"]
    assert difference["removed_count"] == 2
    assert difference["new_count"] == 0
    assert difference["verified"] is True
    removed = difference["removed_detections"]
    assert isinstance(removed, sv.Detections)
    assert set(removed.data["detection_id"]) == {"r0", "r2"}, (
        "Expected the bottle and the bag from the before image to be reported "
        "removed, with their injected detection ids preserved"
    )
    persisted = difference["persisted_detections"]
    assert isinstance(persisted, sv.Detections)
    assert set(persisted.data["detection_id"]) == {"c0"}, (
        "Expected the shifted can re-detection to be reported persisted, in "
        "candidate-image coordinates under its own detection id"
    )
    assert len(difference["new_detections"]) == 0


def test_detections_difference_workflow_when_nothing_removed() -> None:
    # given: before == after — nothing to verify
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_DIFFERENCE_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "reference_predictions": _BEFORE,
            "candidate_predictions": _BEFORE,
        }
    )

    # then
    difference = result[0]["result"]
    assert difference["removed_count"] == 0
    assert difference["new_count"] == 0
    assert difference["verified"] is False
    assert len(difference["persisted_detections"]) == 3


def test_detections_difference_workflow_results_serialization() -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_DIFFERENCE_WORKFLOW,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "reference_predictions": _BEFORE,
            "candidate_predictions": _AFTER,
        },
        serialize_results=True,
    )

    # then: detection outputs come back in the wire format
    difference = result[0]["result"]
    for output_name in (
        "removed_detections",
        "persisted_detections",
        "new_detections",
    ):
        serialized = difference[output_name]
        assert isinstance(serialized, dict), f"`{output_name}` must serialize"
        assert "predictions" in serialized and "image" in serialized
    removed_classes = {
        prediction["class"]
        for prediction in difference["removed_detections"]["predictions"]
    }
    assert removed_classes == {"bottle", "bag"}
    assert difference["removed_count"] == 2
    assert difference["new_count"] == 0
    assert difference["verified"] is True
