"""Every ByteTrack block initializes its tracker with a usable frame rate.

supervision removes a lost track after ``int(frame_rate / 30 * lost_track_buffer)``
frames. With a frame rate of 0 that is 0, so an object missed for three frames
comes back with a new tracker ID. Live sources that declare no frame rate reach
the blocks with ``fps`` 0 or None.
"""

import datetime
from typing import Callable, List, Optional
from unittest import mock

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.transformations.byte_tracker import _fps
from roboflow_workflows.core_steps.transformations.byte_tracker.v1 import (
    ByteTrackerBlockV1,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v1_tensor import (
    ByteTrackerBlockV1 as ByteTrackerBlockV1Tensor,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v2 import (
    ByteTrackerBlockV2,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v2_tensor import (
    ByteTrackerBlockV2 as ByteTrackerBlockV2Tensor,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v3 import (
    ByteTrackerBlockV3,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v3_tensor import (
    ByteTrackerBlockV3 as ByteTrackerBlockV3Tensor,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

from inference_models.models.base.object_detection import Detections


def _metadata(video_id: str, frame_number: int, fps: Optional[float]) -> VideoMetadata:
    return VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=False,
    )


def _image(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((128, 128, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _sv_detections(boxes: List[List[float]]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.full(len(boxes), 0.9),
        class_id=np.ones(len(boxes), dtype=int),
    )


def _tensor_detections(boxes: List[List[float]]) -> Detections:
    count = len(boxes)
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
        class_id=torch.ones(count, dtype=torch.long),
        confidence=torch.full((count,), 0.9, dtype=torch.float32),
        image_metadata={"class_names": {1: "car"}},
        bboxes_metadata=[{"detection_id": f"d{i}"} for i in range(count)],
    )


def _sv_tracker_ids(result: dict) -> List[int]:
    tracker_id = result["tracked_detections"].tracker_id
    return [] if tracker_id is None else tracker_id.tolist()


def _tensor_tracker_ids(result: dict) -> List[int]:
    metadata = result["tracked_detections"].bboxes_metadata or []
    return [int(entry["tracker_id"]) for entry in metadata]


def _runner(block_class, uses_image: bool, tensor: bool) -> Callable:
    def run(block, video_id: str, frame_number: int, fps, boxes) -> List[int]:
        metadata = _metadata(video_id, frame_number, fps)
        detections = (_tensor_detections if tensor else _sv_detections)(boxes)
        if uses_image:
            result = block.run(image=_image(metadata), detections=detections)
        else:
            result = block.run(metadata=metadata, detections=detections)
        return (_tensor_tracker_ids if tensor else _sv_tracker_ids)(result)

    return run


BLOCKS = [
    pytest.param(
        ByteTrackerBlockV1, _runner(ByteTrackerBlockV1, False, False), id="v1"
    ),
    pytest.param(ByteTrackerBlockV2, _runner(ByteTrackerBlockV2, True, False), id="v2"),
    pytest.param(ByteTrackerBlockV3, _runner(ByteTrackerBlockV3, True, False), id="v3"),
    pytest.param(
        ByteTrackerBlockV1Tensor,
        _runner(ByteTrackerBlockV1Tensor, False, True),
        id="v1_tensor",
    ),
    pytest.param(
        ByteTrackerBlockV2Tensor,
        _runner(ByteTrackerBlockV2Tensor, True, True),
        id="v2_tensor",
    ),
    pytest.param(
        ByteTrackerBlockV3Tensor,
        _runner(ByteTrackerBlockV3Tensor, True, True),
        id="v3_tensor",
    ),
]

BOX = [[10, 10, 20, 20]]
OTHER_BOX = [60, 60, 70, 70]


@pytest.mark.parametrize("block_class,run", BLOCKS)
@pytest.mark.parametrize("fps", [None, 0, -5, float("nan"), float("inf")])
def test_missing_fps_initializes_tracker_at_30_fps(block_class, run, fps) -> None:
    block = block_class()

    run(block, "camera", 0, fps, BOX)

    assert block._trackers["camera"].max_time_lost == 30


@pytest.mark.parametrize("block_class,run", BLOCKS)
def test_declared_fps_is_used_without_warning(block_class, run) -> None:
    block = block_class()

    with mock.patch.object(_fps.logger, "warning") as warning:
        run(block, "camera", 0, 10, BOX)

    assert block._trackers["camera"].max_time_lost == 10
    warning.assert_not_called()


@pytest.mark.parametrize("block_class,run", BLOCKS)
def test_missing_fps_warns_once_per_video(block_class, run) -> None:
    block = block_class()

    with mock.patch.object(_fps.logger, "warning") as warning:
        for frame_number in range(5):
            run(block, "front", frame_number, 0, BOX)
            run(block, "back", frame_number, None, BOX)

    assert warning.call_count == 2
    messages = [call.args[0] for call in warning.call_args_list]
    assert block_class.__name__ in messages[0]
    assert "'front'" in messages[0] and "'back'" in messages[1]
    assert "defaulting to 30 fps" in messages[0]


@pytest.mark.parametrize("block_class,run", BLOCKS)
def test_track_keeps_its_id_through_missed_frames_without_declared_fps(
    block_class, run
) -> None:
    block = block_class()
    both = BOX + [OTHER_BOX]

    first = run(block, "camera", 0, 0, both)
    run(block, "camera", 1, 0, both)
    for frame_number in range(2, 5):  # BOX is missed for three frames
        run(block, "camera", frame_number, 0, [OTHER_BOX])
    again = run(block, "camera", 5, 0, both)

    assert len(first) == 2
    assert again == first
