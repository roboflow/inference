import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.trackers.bytetrack.v1 import ByteTrackBlockV1
from roboflow_workflows.core_steps.trackers.bytetrack.v1_tensor import (
    ByteTrackBlockV1 as TensorByteTrackBlockV1,
)

from inference_models.models.base.object_detection import Detections


@pytest.mark.parametrize("tensor", [False, True])
def test_missing_fps_warns_only_when_initializing_a_video(tensor, caplog):
    block = TensorByteTrackBlockV1() if tensor else ByteTrackBlockV1()
    block._create_tracker = Mock(wraps=block._create_tracker)
    if tensor:
        detections = Detections(
            xyxy=torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
            confidence=torch.tensor([0.9]),
            class_id=torch.tensor([0]),
            bboxes_metadata=[{"detection_id": "box"}],
        )
    else:
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    with caplog.at_level(logging.WARNING):
        for video_id, fps in [
            ("one", None),
            ("one", None),
            ("two", 24),
            ("two", None),
            ("three", None),
        ]:
            block.run(
                image=SimpleNamespace(
                    video_metadata=SimpleNamespace(video_identifier=video_id, fps=fps)
                ),
                detections=detections,
            )

    assert [call.kwargs["fps"] for call in block._create_tracker.call_args_list] == [
        30,
        24,
        30,
    ]
    assert len([r for r in caplog.records if "fps not available" in r.message]) == 2
