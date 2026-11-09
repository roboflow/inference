import datetime

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("remove", [False, True], ids=["keep-outside", "drop-outside"])
def test_workflow_time_in_zone_reentry_preserves_first_entry(remove: bool) -> None:
    # given
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        torch = pytest.importorskip("torch")
        from inference_models.models.base.object_detection import Detections

    workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "detections",
                "kind": ["object_detection_prediction"],
                "dimensionality": 1,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/time_in_zone@v3",
                "name": "timer",
                "image": "$inputs.image",
                "detections": "$inputs.detections",
                "zone": [[10, 10], [10, 30], [30, 30], [30, 10]],
                "remove_out_of_zone_detections": remove,
                "reset_out_of_zone_detections": False,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "timed",
                "selector": "$steps.timer.timed_detections",
                "coordinates_system": "own",
            }
        ],
    }
    timestamp = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    engine = ExecutionEngine.init(workflow_definition=workflow, init_parameters={})

    # when
    elapsed = []
    try:
        for frame_number, center in enumerate([16, 16, 42, 16]):
            image = WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="test-frame"),
                numpy_image=pixels,
                video_metadata=VideoMetadata(
                    video_identifier="test-video",
                    frame_number=frame_number,
                    fps=1,
                    frame_timestamp=timestamp,
                    comes_from_video_file=True,
                ),
            )
            box = [[center - 2, center - 2, center + 2, center + 2]]
            if ENABLE_TENSOR_DATA_REPRESENTATION:
                detections = Detections(
                    xyxy=torch.tensor(box, dtype=torch.float32),
                    class_id=torch.tensor([0]),
                    confidence=torch.tensor([0.9]),
                    bboxes_metadata=[{"tracker_id": 7}],
                )
            else:
                detections = sv.Detections(
                    xyxy=np.array(box, dtype=float), tracker_id=np.array([7])
                )
            timed = engine.run(
                runtime_parameters={"image": image, "detections": [detections]},
            )[0]["timed"]
            if ENABLE_TENSOR_DATA_REPRESENTATION:
                elapsed.append(
                    timed.bboxes_metadata[0]["time_in_zone"]
                    if len(timed.xyxy)
                    else None
                )
            else:
                elapsed.append(timed["time_in_zone"][0] if len(timed) else None)
    finally:
        engine.close()

    # then
    assert elapsed == [0.0, 1.0, None if remove else 0.0, 3.0]
