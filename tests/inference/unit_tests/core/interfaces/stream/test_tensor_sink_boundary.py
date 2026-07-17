from datetime import datetime

import numpy as np
import pytest

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.utils import materialise_video_frame_for_sink

torch = pytest.importorskip("torch")


def test_tensor_frame_reaches_sink_unmaterialised() -> None:
    # Under ENABLE_TENSOR_DATA_REPRESENTATION nothing materialises in dispatch:
    # the sink receives the ORIGINAL on-device tensor frame. Pixel-consuming
    # sinks convert at their own boundary via materialise_video_frame_for_sink.
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

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline._on_prediction = sink
    pipeline._status_update_handlers = []

    pipeline._use_sink({"result": "ok"}, video_frame)

    assert received["predictions"] == {"result": "ok"}
    assert received["frame"] is video_frame
    assert received["frame"].image is tensor_image


def test_tensor_frames_reach_batch_sink_unmaterialised() -> None:
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
    assert received["frames"][0] is video_frame
    assert received["frames"][1] is None


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


def test_materialise_helper_converts_3chw_tensor_to_bgr_hwc_numpy() -> None:
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

    materialised = materialise_video_frame_for_sink(video_frame)

    expected_bgr = np.array(
        [
            [[30, 20, 10], [31, 21, 11]],
            [[32, 22, 12], [33, 23, 13]],
        ],
        dtype=np.uint8,
    )
    assert materialised is not video_frame
    assert materialised.image.flags.c_contiguous
    np.testing.assert_array_equal(materialised.image, expected_bgr)
    # The original frame stays untouched (frozen-dataclass replace semantics).
    assert video_frame.image is tensor_image


def test_materialise_helper_passes_numpy_frame_through() -> None:
    video_frame = VideoFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        frame_id=1,
        frame_timestamp=datetime.now(),
        source_id=0,
    )

    assert materialise_video_frame_for_sink(video_frame) is video_frame
    assert materialise_video_frame_for_sink(None) is None
