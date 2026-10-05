from pathlib import Path
from unittest.mock import MagicMock

import av
import numpy as np

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
