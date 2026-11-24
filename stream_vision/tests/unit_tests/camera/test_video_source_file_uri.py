from pathlib import Path

import cv2
import numpy as np
import pytest
from streamvision.camera.local_files import local_file_path
from streamvision.camera.video_source import CV2VideoFrameProducer, VideoSource

FRAME_COUNT = 10


@pytest.fixture
def clip_path(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for frame_index in range(FRAME_COUNT):
        writer.write(np.full((48, 64, 3), frame_index * 20, dtype=np.uint8))
    writer.release()

    return path


def test_file_uri_and_local_path_give_the_same_source_properties(
    clip_path: Path,
) -> None:
    # when
    from_path = CV2VideoFrameProducer(str(clip_path))
    from_uri = CV2VideoFrameProducer(clip_path.as_uri())
    try:
        path_properties = from_path.discover_source_properties()
        uri_properties = from_uri.discover_source_properties()
    finally:
        from_path.release()
        from_uri.release()

    # then
    assert path_properties.is_file is True
    assert path_properties.total_frames == FRAME_COUNT
    assert uri_properties == path_properties


def test_video_source_marks_frames_of_a_file_uri_as_video_file_frames(
    clip_path: Path,
) -> None:
    # given
    source = VideoSource.init(video_reference=clip_path.as_uri())

    # when
    source.start()
    try:
        frames = list(source)
    finally:
        source.terminate(wait_on_frames_consumption=False)

    # then
    assert len(frames) == FRAME_COUNT
    assert all(frame.comes_from_video_file is True for frame in frames)
    assert all(frame.total_frames == FRAME_COUNT for frame in frames)


@pytest.mark.parametrize(
    "video, expected",
    [
        ("file:///videos/my%20clip.mp4", "/videos/my clip.mp4"),
        ("file://localhost/videos/clip.mp4", "/videos/clip.mp4"),
        ("./videos/clip.mp4", "./videos/clip.mp4"),
        ("rtsp://camera.local/stream", None),
        (0, None),
    ],
)
def test_local_file_path(video, expected) -> None:
    # when
    result = local_file_path(video)

    # then
    assert result == expected
