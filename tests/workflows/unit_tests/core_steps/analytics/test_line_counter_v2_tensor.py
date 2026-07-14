import datetime

import numpy as np
import pytest
import supervision as sv

torch = pytest.importorskip("torch")

from inference.core.workflows.core_steps.analytics.line_counter import (  # noqa: E402
    v2_tensor,
)
from inference.core.workflows.execution_engine.entities.base import (  # noqa: E402
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections  # noqa: E402

LineCounterBlockV2 = v2_tensor.LineCounterBlockV2


def _native_detections(xyxy: np.ndarray, tracker_ids: np.ndarray) -> Detections:
    n = len(xyxy)
    return Detections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32),
        class_id=torch.arange(n, dtype=torch.int64),
        confidence=torch.ones(n, dtype=torch.float32),
        bboxes_metadata=[{"tracker_id": int(value)} for value in tracker_ids],
    )


def test_line_counter_v2_matches_raw_supervision_across_frames() -> None:
    line_segment = [[15, 0], [15, 100]]
    frames = [
        (
            np.array([[5, 10, 7, 12], [20, 20, 22, 22], [5, 150, 7, 152]]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([[20, 10, 22, 12], [5, 20, 7, 22], [20, 150, 22, 152]]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([[22, 10, 24, 12], [3, 20, 5, 22], [10, 40, 20, 50]]),
            np.array([1, 2, 4]),
        ),
    ]
    metadata = VideoMetadata(
        video_identifier="tensor-line",
        frame_number=0,
        frame_timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((200, 200, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    block = LineCounterBlockV2()
    oracle = sv.LineZone(sv.Point(*line_segment[0]), sv.Point(*line_segment[1]))

    for xyxy, tracker_ids in frames:
        expected_in, expected_out = oracle.trigger(
            sv.Detections(xyxy=xyxy.astype(float), tracker_id=tracker_ids)
        )
        result = block.run(
            detections=_native_detections(xyxy, tracker_ids),
            image=image,
            line_segment=line_segment,
            triggering_anchor=None,
        )

        assert result["count_in"] == oracle.in_count
        assert result["count_out"] == oracle.out_count
        np.testing.assert_array_equal(
            result["detections_in"].xyxy.numpy(), xyxy[expected_in]
        )
        np.testing.assert_array_equal(
            result["detections_out"].xyxy.numpy(), xyxy[expected_out]
        )
        assert [
            item["tracker_id"] for item in result["detections_in"].bboxes_metadata
        ] == tracker_ids[expected_in].tolist()
        assert [
            item["tracker_id"] for item in result["detections_out"].bboxes_metadata
        ] == tracker_ids[expected_out].tolist()
