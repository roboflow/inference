import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
from streamvision.webrtc_worker.webrtc import VideoFrameProcessor

from inference.core.interfaces.webrtc_worker.utils import (
    get_video_total_frames,
    process_frame,
)


def test_get_video_total_frames_reads_container_frame_count(
    local_video_path: str,
) -> None:
    # when
    result = get_video_total_frames(local_video_path)

    # then
    assert result == 431


def test_get_video_total_frames_returns_none_for_unreadable_file(
    tmp_path: Path,
) -> None:
    # when
    result = get_video_total_frames(str(tmp_path / "missing.mp4"))

    # then
    assert result is None


def test_process_frame_passes_total_frames_to_pipeline() -> None:
    # given
    frame = av.VideoFrame.from_ndarray(
        np.zeros((8, 8, 3), dtype=np.uint8), format="bgr24"
    )
    pipeline = MagicMock()
    pipeline._on_video_frame.return_value = [{}]

    # when
    process_frame(
        frame=frame,
        frame_id=3,
        declared_fps=30.0,
        measured_fps=30.0,
        comes_from_video_file=True,
        inference_pipeline=pipeline,
        render_output=False,
        total_frames=431,
    )

    # then
    [video_frames], _ = pipeline._on_video_frame.call_args
    assert video_frames[0].frame_id == 3
    assert video_frames[0].total_frames == 431


def test_get_video_total_frames_returns_none_when_container_reports_zero() -> None:
    # given
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [MagicMock(frames=0)]

    # when
    with patch("streamvision.webrtc_worker.utils.av.open", return_value=container):
        result = get_video_total_frames("video.mp4")

    # then
    assert result is None


def test_video_frame_processor_forwards_total_frames_to_process_frame() -> None:
    # given
    processor = object.__new__(VideoFrameProcessor)
    processor._rotation_code = None
    processor._declared_fps = 30.0
    processor._fps_monitor = MagicMock(all_timestamps=[])
    processor._file_processing = True
    processor._inference_pipeline = MagicMock()
    processor._total_frames = 431
    process_frame_mock = MagicMock(return_value=({}, None, []))

    # when
    with patch("streamvision.webrtc_worker.webrtc.process_frame", process_frame_mock):
        asyncio.run(processor._process_frame_async(frame=MagicMock(), frame_id=3))

    # then
    assert process_frame_mock.call_args.args[-1] == 431
