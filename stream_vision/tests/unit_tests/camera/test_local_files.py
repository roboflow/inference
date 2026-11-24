from pathlib import Path

import pytest
from streamvision.camera.local_files import local_file_path, source_uri


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


@pytest.mark.parametrize(
    "video, expected",
    [
        ("/videos/my clip.mp4", "file:///videos/my%20clip.mp4"),
        ("file:///videos/my%20clip.mp4", "file:///videos/my%20clip.mp4"),
        ("rtsp://camera.local/stream", "rtsp://camera.local/stream"),
        (0, "0"),
    ],
)
def test_source_uri(video, expected) -> None:
    # when
    result = source_uri(video)

    # then
    assert result == expected


def test_source_uri_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    monkeypatch.chdir(tmp_path)

    # when
    result = source_uri("./clip.mp4")

    # then
    assert result == (tmp_path / "clip.mp4").resolve().as_uri()
