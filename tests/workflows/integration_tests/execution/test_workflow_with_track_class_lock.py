import numpy as np
import pytest
import supervision as sv

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine

# Under ENABLE_TENSOR_DATA_REPRESENTATION the track_class_lock block is the
# tensor-native sibling (v1_tensor): it accepts/returns an
# `inference_models.Detections` (per-box info on bboxes_metadata, class map on
# image_metadata[CLASS_NAMES_KEY]) instead of an sv.Detections, and the tensor
# deserializer rejects a raw sv.Detections runtime parameter. The sv test below is
# skipped when the flag is on; the `*_tensor_native` parity test (skipped when the
# flag is off) exercises the SAME lock-persistence semantics with a native input.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections runtime input + sv .data output; track_class_lock is "
    "native-only under ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native "
    "parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

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


@_NUMPY_ONLY
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


def _native_tracked_detections(class_name: str, confidence: float):
    """Native `inference_models.Detections` equivalent of `_tracked_detections`:
    one tracked box with tracker_id + detection_id + class carried on
    bboxes_metadata, and the class_id -> name map on
    image_metadata[CLASS_NAMES_KEY]."""
    import torch

    from inference_models.models.base.object_detection import Detections

    return Detections(
        xyxy=torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
        class_id=torch.tensor([CLASS_IDS[class_name]], dtype=torch.long),
        confidence=torch.tensor([confidence], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "cat", 1: "dog"}},
        bboxes_metadata=[
            {
                TRACKER_ID_KEY: 7,
                DETECTION_ID_KEY: "d0",
                CLASS_NAME_KEY: class_name,
            }
        ],
    )


def _native_class_name(detections) -> str:
    """Read the per-box class name from a native Detections output: prefer the
    per-box `class` on bboxes_metadata, fall back to image_metadata[CLASS_NAMES_KEY]
    keyed by class_id."""
    entry = (detections.bboxes_metadata or [{}])[0] or {}
    if CLASS_NAME_KEY in entry:
        return str(entry[CLASS_NAME_KEY])
    class_names_map = (detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    return str(class_names_map[int(detections.class_id[0])])


def _native_class_locked(detections) -> bool:
    return bool((detections.bboxes_metadata or [{}])[0].get("class_locked"))


@_TENSOR_ONLY
def test_track_class_lock_state_persists_across_sequential_engine_runs_tensor_native(
    model_manager: ModelManager,
) -> None:
    """Tensor-native parity of the persistence test: feeds native
    `inference_models.Detections` frames one per engine call and asserts the SAME
    lock-persistence semantics (lock acquired after min_votes frames; a later
    contrary frame is relabelled to the locked class), proving the cross-run state
    persists in the tensor block too."""
    # given
    from inference_models.models.base.object_detection import (
        Detections as NativeDetections,
    )

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
                "predictions": [_native_tracked_detections("cat", 0.9)],
            }
        )
        results.append(result)

    # then - lock requires state accumulated across separate engine runs
    before_lock = results[8][0]["tracked_detections"]
    assert isinstance(before_lock, NativeDetections)
    assert not _native_class_locked(before_lock)
    locked = results[9][0]["tracked_detections"]
    assert _native_class_locked(locked)
    assert _native_class_name(locked) == "cat"

    # when - a contrary "dog" frame arrives in a fresh engine call
    result = execution_engine.run(
        runtime_parameters={
            "image": [image],
            "predictions": [_native_tracked_detections("dog", 0.95)],
        }
    )

    # then - relabelled to the locked class, proving state persisted
    out = result[0]["tracked_detections"]
    assert isinstance(out, NativeDetections)
    assert _native_class_locked(out)
    assert _native_class_name(out) == "cat"
    assert int(out.class_id[0]) == CLASS_IDS["cat"]
