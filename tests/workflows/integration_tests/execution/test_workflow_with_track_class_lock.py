import numpy as np
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

TRACK_CLASS_LOCK_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/track_class_lock@v1",
            "name": "class_lock",
            "image": "$inputs.image",
            "detections": "$inputs.predictions",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "tracked_detections",
            "selector": "$steps.class_lock.tracked_detections",
        }
    ],
}

CLASS_IDS = {"cat": 0, "dog": 1}


def _tracked_detections(class_name: str, confidence: float) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        confidence=np.array([confidence]),
        class_id=np.array([CLASS_IDS[class_name]]),
        tracker_id=np.array([7]),
        data={
            "class_name": np.array([class_name]),
            "detection_id": np.array(["d0"]),
        },
    )


def test_track_class_lock_state_persists_across_sequential_engine_runs(
    model_manager: ModelManager,
) -> None:
    """
    Feeds frames one by one through ExecutionEngine.run() - mirroring how the
    block is used on a video stream - and verifies that voting state persists
    across engine calls: the lock is acquired after min_votes frames and a
    contrary frame afterwards is relabelled to the locked class.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=TRACK_CLASS_LOCK_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # when - 10 confident "cat" frames, one engine call per frame
    results = []
    for _ in range(10):
        result = execution_engine.run(
            runtime_parameters={
                "image": [image],
                "predictions": [_tracked_detections("cat", 0.9)],
            }
        )
        results.append(result)

    # then - lock requires state accumulated across separate engine runs
    before_lock = results[8][0]["tracked_detections"]
    assert not before_lock.data["class_locked"][0]
    locked = results[9][0]["tracked_detections"]
    assert locked.data["class_locked"][0]
    assert locked.data["class_name"][0] == "cat"

    # when - a contrary "dog" frame arrives in a fresh engine call
    result = execution_engine.run(
        runtime_parameters={
            "image": [image],
            "predictions": [_tracked_detections("dog", 0.95)],
        }
    )

    # then - relabelled to the locked class, proving state persisted
    out = result[0]["tracked_detections"]
    assert out.data["class_locked"][0]
    assert out.data["class_name"][0] == "cat"
    assert out.class_id[0] == CLASS_IDS["cat"]
