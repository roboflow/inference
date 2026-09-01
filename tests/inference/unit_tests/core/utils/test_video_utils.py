"""Tests for reading a clip's windows in one pass."""

import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from inference.core.exceptions import PayloadTooLargeError
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


class _FakeResponse:
    """Enough of requests.Response for the drain to read."""

    def __init__(self, chunks, declared=None):
        self._chunks = chunks
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_the_drain_returns_bytes_when_no_sink_is_given() -> None:
    from inference.core.utils.url_input import _drain_response

    result = _drain_response(
        response=_FakeResponse([b"abc", b"def"]), sink=None, max_bytes=None
    )

    assert result == b"abcdef"


def test_the_drain_writes_to_a_sink_and_keeps_nothing() -> None:
    from inference.core.utils.url_input import _drain_response

    written = []

    result = _drain_response(
        response=_FakeResponse([b"abc", b"def"]), sink=written.append, max_bytes=None
    )

    assert result is None
    assert b"".join(written) == b"abcdef"


def test_a_declared_length_over_the_cap_is_rejected_before_reading() -> None:
    from inference.core.utils.url_input import _drain_response

    # Nothing is read: the chunks would raise if they were.
    def _explode(chunk_size):
        raise AssertionError("body must not be read")

    response = _FakeResponse([], declared=100)
    response.iter_content = _explode

    with pytest.raises(PayloadTooLargeError):
        _drain_response(response=response, sink=None, max_bytes=10)


def test_a_body_that_grows_past_the_cap_is_stopped_mid_stream() -> None:
    # The declared length is caller-controlled and can lie, so the running
    # total is what actually enforces the cap.
    from inference.core.utils.url_input import _drain_response

    written = []

    with pytest.raises(PayloadTooLargeError):
        _drain_response(
            response=_FakeResponse([b"x" * 8, b"x" * 8], declared=4),
            sink=written.append,
            max_bytes=10,
        )

    # It stopped rather than buffering the rest.
    assert sum(len(chunk) for chunk in written) <= 10


def test_a_body_under_the_cap_passes() -> None:
    from inference.core.utils.url_input import _drain_response

    result = _drain_response(
        response=_FakeResponse([b"x" * 8], declared=8), sink=None, max_bytes=10
    )

    assert result == b"x" * 8


def test_a_base64_clip_over_the_cap_is_rejected(monkeypatch) -> None:
    import inference.core.utils.video_utils as video_utils

    monkeypatch.setattr(video_utils, "MAX_VIDEO_DOWNLOAD_SIZE_MB", 0)
    oversized = base64.b64encode(b"x" * 32).decode("ascii")

    with pytest.raises(PayloadTooLargeError):
        with video_utils.video_source_path(video_type="base64", value=oversized):
            pass


def test_an_oversized_base64_clip_is_rejected_without_decoding(monkeypatch) -> None:
    # Decoding first would expand the clip into memory just to reject it.
    import inference.core.utils.video_utils as video_utils

    monkeypatch.setattr(video_utils, "MAX_VIDEO_DOWNLOAD_SIZE_MB", 0)
    decoded = MagicMock(side_effect=AssertionError("must not decode"))
    monkeypatch.setattr(video_utils.base64, "b64decode", decoded)

    with pytest.raises(PayloadTooLargeError):
        with video_utils.video_source_path(
            video_type="base64", value=base64.b64encode(b"x" * 32).decode("ascii")
        ):
            pass

    decoded.assert_not_called()


def test_an_uncapped_deployment_reads_any_size(monkeypatch) -> None:
    import inference.core.utils.video_utils as video_utils

    monkeypatch.setattr(video_utils, "MAX_VIDEO_DOWNLOAD_SIZE_MB", -1)

    assert video_utils._max_download_bytes() is None
