"""Unit tests for VideoFileSource http(s) URL support."""

import os
from unittest import mock

import pytest

from inference_sdk.webrtc import sources
from inference_sdk.webrtc.sources import VideoFileSource, _download_video

VIDEO_URL = "https://example.com/videos/cars.mp4"


def _fake_fetch(payload: bytes = b"video-bytes"):
    def fetch(url: str, destination_path: str, request_timeout=None) -> None:
        with open(destination_path, "wb") as f:
            f.write(payload)

    return fetch


class TestVideoFileSourceUrlDetection:
    def test_local_path_is_not_url(self) -> None:
        source = VideoFileSource("cars.mp4")
        assert source._is_url is False

    def test_http_and_https_paths_are_urls(self) -> None:
        assert VideoFileSource("http://example.com/a.mp4")._is_url is True
        assert VideoFileSource(VIDEO_URL)._is_url is True

    def test_use_cache_defaults_to_true(self) -> None:
        assert VideoFileSource(VIDEO_URL).use_cache is True


class TestDownloadVideo:
    def test_downloads_to_cache_dir_and_reuses_cached_file(
        self, tmp_path, monkeypatch
    ) -> None:
        cache_dir = str(tmp_path / "cache")
        monkeypatch.setattr(sources, "VIDEO_DOWNLOAD_CACHE_DIR", cache_dir)

        with mock.patch.object(
            sources, "fetch_url_to_file", side_effect=_fake_fetch()
        ) as fetch_mock:
            first = _download_video(VIDEO_URL, use_cache=True)
            second = _download_video(VIDEO_URL, use_cache=True)

        assert first == second
        assert first.startswith(cache_dir)
        assert first.endswith(".mp4")
        with open(first, "rb") as f:
            assert f.read() == b"video-bytes"
        # Second call served from cache - only one network request
        assert fetch_mock.call_count == 1

    def test_use_cache_false_downloads_to_temp_file(
        self, tmp_path, monkeypatch
    ) -> None:
        cache_dir = str(tmp_path / "cache")
        monkeypatch.setattr(sources, "VIDEO_DOWNLOAD_CACHE_DIR", cache_dir)

        with mock.patch.object(sources, "fetch_url_to_file", side_effect=_fake_fetch()):
            path = _download_video(VIDEO_URL, use_cache=False)

        try:
            assert not path.startswith(cache_dir)
            with open(path, "rb") as f:
                assert f.read() == b"video-bytes"
        finally:
            os.remove(path)

    def test_failed_download_leaves_no_cached_file(self, tmp_path, monkeypatch) -> None:
        cache_dir = str(tmp_path / "cache")
        monkeypatch.setattr(sources, "VIDEO_DOWNLOAD_CACHE_DIR", cache_dir)

        with mock.patch.object(
            sources, "fetch_url_to_file", side_effect=RuntimeError("HTTP 404")
        ):
            with pytest.raises(RuntimeError):
                _download_video(VIDEO_URL, use_cache=True)

        assert os.listdir(cache_dir) == []

    def test_url_without_extension_defaults_to_mp4(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(sources, "VIDEO_DOWNLOAD_CACHE_DIR", str(tmp_path))

        with mock.patch.object(sources, "fetch_url_to_file", side_effect=_fake_fetch()):
            path = _download_video("https://example.com/stream", use_cache=True)

        assert path.endswith(".mp4")


class TestVideoFileSourceCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_download(self, tmp_path) -> None:
        temp_file = tmp_path / "downloaded.mp4"
        temp_file.write_bytes(b"x")

        source = VideoFileSource(VIDEO_URL, use_cache=False)
        source._temp_download_path = str(temp_file)

        await source.cleanup()

        assert not temp_file.exists()
        assert source._temp_download_path is None

    @pytest.mark.asyncio
    async def test_cleanup_noop_for_local_path(self) -> None:
        source = VideoFileSource("cars.mp4")
        await source.cleanup()  # must not raise
