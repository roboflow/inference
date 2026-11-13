import datetime
from importlib import import_module

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("tensor_native", [False, True], ids=["numpy", "tensor"])
@pytest.mark.parametrize(
    ("remove", "reset", "expected_exit", "expected_reentry"),
    [
        pytest.param(False, False, 0.0, 3.0, id="keep-continue"),
        pytest.param(True, False, None, 3.0, id="drop-continue"),
        pytest.param(False, True, 0.0, 0.0, id="keep-reset"),
        pytest.param(True, True, None, 0.0, id="drop-reset"),
    ],
)
def test_time_in_zone_reentry(
    version: int,
    tensor_native: bool,
    remove: bool,
    reset: bool,
    expected_exit: float | None,
    expected_reentry: float,
) -> None:
    # given
    if tensor_native:
        torch = pytest.importorskip("torch")
        from inference_models.models.base.object_detection import Detections

    suffix = "_tensor" if tensor_native else ""
    module = import_module(
        f"roboflow_workflows.core_steps.analytics.time_in_zone.v{version}{suffix}"
    )
    block = getattr(module, f"TimeInZoneBlockV{version}")()
    zone = [[10, 10], [10, 30], [30, 30], [30, 10]]
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    timestamp = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

    # when
    frames = []
    for frame_number in range(4):
        moving_box = [40, 40, 43, 43] if frame_number == 2 else [14, 14, 17, 17]
        boxes = np.array([moving_box, [20, 20, 23, 23]], dtype=float)
        metadata = VideoMetadata(
            video_identifier="test-video",
            frame_number=frame_number,
            fps=1,
            frame_timestamp=timestamp,
            comes_from_video_file=True,
        )
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test-frame"),
            numpy_image=pixels,
            video_metadata=metadata,
        )
        if tensor_native:
            detections = Detections(
                xyxy=torch.as_tensor(boxes, dtype=torch.float32),
                class_id=torch.zeros(2, dtype=torch.int64),
                confidence=torch.ones(2, dtype=torch.float32),
                bboxes_metadata=[{"tracker_id": 7}, {"tracker_id": 8}],
            )
        else:
            detections = sv.Detections(xyxy=boxes, tracker_id=np.array([7, 8]))

        result = block.run(
            image=image,
            detections=detections,
            zone=zone,
            triggering_anchor="CENTER",
            remove_out_of_zone_detections=remove,
            reset_out_of_zone_detections=reset,
            **({"metadata": metadata} if version == 1 else {}),
        )["timed_detections"]
        if tensor_native:
            tracks = [
                (row["tracker_id"], row["time_in_zone"])
                for row in result.bboxes_metadata
            ]
        else:
            tracks = list(zip(result.tracker_id, result["time_in_zone"]))
        frames.append(tracks)

    # then
    outside_frame = [(8, 2.0)]
    if expected_exit is not None:
        outside_frame.insert(0, (7, expected_exit))

    assert frames == [
        [(7, 0.0), (8, 0.0)],
        [(7, 1.0), (8, 1.0)],
        outside_frame,
        [(7, expected_reentry), (8, 3.0)],
    ]
