from datetime import datetime

import numpy as np
import pytest

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

torch = pytest.importorskip("torch")


def test_tensor_frame_is_materialised_as_bgr_numpy_before_sink_call() -> None:
    tensor_image = torch.tensor(
        [
            [[10, 11], [12, 13]],
            [[20, 21], [22, 23]],
            [[30, 31], [32, 33]],
        ],
        dtype=torch.uint8,
    )
    video_frame = VideoFrame(
        image=tensor_image,
        frame_id=1,
        frame_timestamp=datetime.now(),
        source_id=0,
    )
    received = {}

    def sink(predictions, frame):
        received["predictions"] = predictions
        received["frame"] = frame
        received["copy"] = frame.image.copy()

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline._on_prediction = sink
    pipeline._status_update_handlers = []

    pipeline._use_sink({"result": "ok"}, video_frame)

    expected_bgr = np.array(
        [
            [[30, 20, 10], [31, 21, 11]],
            [[32, 22, 12], [33, 23, 13]],
        ],
        dtype=np.uint8,
    )
    assert received["predictions"] == {"result": "ok"}
    assert received["frame"] is not video_frame
    assert received["frame"].image.flags.c_contiguous
    np.testing.assert_array_equal(received["frame"].image, expected_bgr)
    np.testing.assert_array_equal(received["copy"], expected_bgr)
    assert video_frame.image is tensor_image


def test_tensor_frames_are_materialised_in_batch_sink_payload() -> None:
    video_frame = VideoFrame(
        image=torch.arange(4, dtype=torch.uint8).reshape(1, 2, 2),
        frame_id=1,
        frame_timestamp=datetime.now(),
        source_id=0,
    )
    received = {}

    def sink(predictions, frames):
        received["predictions"] = predictions
        received["frames"] = frames

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline._on_prediction = sink
    pipeline._status_update_handlers = []

    pipeline._use_sink([{"result": "ok"}, None], [video_frame, None])

    assert received["predictions"] == [{"result": "ok"}, None]
    assert received["frames"][1] is None
    np.testing.assert_array_equal(
        received["frames"][0].image,
        np.array([[0, 1], [2, 3]], dtype=np.uint8),
    )


def test_numpy_frame_is_passed_to_sink_without_copying() -> None:
    video_frame = VideoFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        frame_id=1,
        frame_timestamp=datetime.now(),
        source_id=0,
    )
    received = {}

    def sink(predictions, frame):
        received["frame"] = frame

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline._on_prediction = sink
    pipeline._status_update_handlers = []

    pipeline._use_sink({}, video_frame)

    assert received["frame"] is video_frame
