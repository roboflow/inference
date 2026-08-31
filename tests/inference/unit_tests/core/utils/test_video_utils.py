"""Tests for reading a clip's windows in one pass."""

import cv2
import numpy as np
import pytest

from inference.core.utils.video_utils import read_frame_windows, read_frames


@pytest.fixture
def clip(tmp_path) -> str:
    """A 30-frame clip whose blue channel equals the frame number."""
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (16, 16))
    for frame_number in range(30):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :, 0] = frame_number
        writer.write(frame)
    writer.release()
    return path


def _frame_numbers(frames) -> list:
    # _to_rgb reverses the channels, so the written blue lands in red.
    return [int(frame[0, 0, 2]) for frame in frames]


def test_each_window_gets_the_frames_it_asked_for(clip) -> None:
    windows = [[0, 2, 4], [10, 12], [20, 25, 29]]

    result = list(read_frame_windows(path=clip, windows=windows))

    assert [_frame_numbers(window) for window in result] == windows


def test_the_clip_is_walked_once_for_every_window(clip, monkeypatch) -> None:
    # Reading each window on its own capture re-decodes the frames before it,
    # so the cost grows with the square of the clip length.
    opened = []
    real_video_capture = cv2.VideoCapture

    def _counting_capture(*args, **kwargs):
        opened.append(args[0] if args else None)
        return real_video_capture(*args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", _counting_capture)

    list(read_frame_windows(path=clip, windows=[[0], [10], [20], [29]]))

    assert len(opened) == 1


def test_a_window_past_the_end_comes_back_short(clip) -> None:
    result = list(read_frame_windows(path=clip, windows=[[0, 1], [100, 200]]))

    assert _frame_numbers(result[0]) == [0, 1]
    assert result[1] == []


def test_frames_are_capped_only_when_a_side_is_given(clip) -> None:
    uncapped = list(read_frame_windows(path=clip, windows=[[0]]))[0]
    capped = list(read_frame_windows(path=clip, windows=[[0]], max_frame_side=8))[0]

    assert uncapped[0].shape == (16, 16, 3)
    assert capped[0].shape == (8, 8, 3)


def test_read_frames_reads_one_window(clip) -> None:
    assert _frame_numbers(read_frames(path=clip, frame_indices=[3, 7])) == [3, 7]


def test_no_windows_reads_nothing(clip) -> None:
    assert list(read_frame_windows(path=clip, windows=[])) == []
