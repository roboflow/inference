import asyncio
import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import Response

from inference_server import configuration
from inference_server.errors import error_response
from inference_server.legacy import video
from inference_server.legacy.errors import LegacyHTTPError

_IMAGE_ERROR_PREFIX = "Could not load input image. Cause: "
_FRAME_COUNT = 12
_FPS = 10.0
_WIDTH, _HEIGHT = 64, 48


def _frame_colour_bgr(index: int) -> tuple[int, int, int]:
    return (200, 20 + index * 15, 60)


def _write_clip(path: Path, frame_count: int = _FRAME_COUNT) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_WIDTH, _HEIGHT)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        frame[:] = _frame_colour_bgr(index)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path) -> Path:
    return _write_clip(tmp_path / "clip.mp4")


@pytest.fixture
def temp_dir(tmp_path, monkeypatch) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


def _assert_image_error(error: LegacyHTTPError, cause: str) -> None:
    assert error.status_code == 400
    assert error.message == f"{_IMAGE_ERROR_PREFIX}{cause}"


def _channel_mean(frame: np.ndarray) -> np.ndarray:
    return frame.reshape(-1, 3).mean(axis=0)


class _DelegatingCapture:
    def __init__(self, inner):
        self._inner = inner

    def isOpened(self):
        return self._inner.isOpened()

    def get(self, prop):
        return self._inner.get(prop)

    def set(self, prop, value):
        return self._inner.set(prop, value)

    def read(self):
        return self._inner.read()

    def release(self):
        self._inner.release()


class TestProbeVideo:
    def test_reports_fps_and_frame_count(self, clip):
        assert video.probe_video(str(clip)) == (_FPS, _FRAME_COUNT)

    def test_undecodable_bytes_are_refused(self, tmp_path):
        bad = tmp_path / "bad.video"
        bad.write_bytes(os.urandom(4096))

        with pytest.raises(LegacyHTTPError) as error:
            video.probe_video(str(bad))

        _assert_image_error(error.value, "Video could not be decoded.")

    def test_still_image_is_refused(self, tmp_path):
        still = tmp_path / "still.jpg"
        cv2.imwrite(str(still), np.full((_HEIGHT, _WIDTH, 3), 90, dtype=np.uint8))

        with pytest.raises(LegacyHTTPError) as error:
            video.probe_video(str(still))

        _assert_image_error(
            error.value,
            "Video holds 1 frame(s). A clip needs at least two frames to hold an "
            "action. Send a video, not a still image.",
        )

    def test_unusable_frame_rate_is_refused(self, clip, monkeypatch):
        class _Capture:
            def __init__(self, path):
                pass

            def isOpened(self):
                return True

            def get(self, prop):
                return 0.0 if prop == cv2.CAP_PROP_FPS else 12.0

            def release(self):
                pass

        monkeypatch.setattr(video.cv2, "VideoCapture", _Capture)

        with pytest.raises(LegacyHTTPError) as error:
            video.probe_video(str(clip))

        _assert_image_error(error.value, "Video declares no usable frame rate.")

    def test_header_claim_over_file_size_is_recounted(self, clip, monkeypatch):
        real_capture = cv2.VideoCapture

        class _Capture(_DelegatingCapture):
            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return float(clip.stat().st_size + 1)
                return self._inner.get(prop)

        monkeypatch.setattr(
            video.cv2, "VideoCapture", lambda path: _Capture(real_capture(path))
        )

        assert video.probe_video(str(clip)) == (_FPS, _FRAME_COUNT)


class TestReadFrameWindows:
    def test_yields_rgb_frames_per_window_with_a_short_trailing_window(self, clip):
        windows = [[0, 2, 4], [6, 8, 10], [10, 11, 12, 13]]

        read = list(video.read_frame_windows(str(clip), windows=windows))

        assert [len(frames) for frames in read] == [3, 3, 2]
        for frames, window in zip(read, windows):
            for frame, index in zip(frames, window):
                assert frame.shape == (_HEIGHT, _WIDTH, 3)
                assert frame.dtype == np.uint8
                assert frame.flags["C_CONTIGUOUS"]
                blue, green, red = _frame_colour_bgr(index)
                assert np.allclose(_channel_mean(frame), (red, green, blue), atol=12)

    def test_downscales_the_longest_side_with_area_interpolation(self, clip):
        (frames,) = list(
            video.read_frame_windows(str(clip), windows=[[0, 1]], max_frame_side=32)
        )

        assert [frame.shape for frame in frames] == [(24, 32, 3), (24, 32, 3)]
        blue, green, red = _frame_colour_bgr(1)
        assert np.allclose(_channel_mean(frames[1]), (red, green, blue), atol=12)

    def test_small_frames_are_not_upscaled(self, clip):
        (frames,) = list(
            video.read_frame_windows(str(clip), windows=[[0]], max_frame_side=1000)
        )

        assert frames[0].shape == (_HEIGHT, _WIDTH, 3)

    def test_no_windows_yields_nothing(self, clip):
        assert list(video.read_frame_windows(str(clip), windows=[])) == []

    def test_undecodable_file_is_refused(self, tmp_path):
        bad = tmp_path / "bad.video"
        bad.write_bytes(os.urandom(4096))

        with pytest.raises(LegacyHTTPError) as error:
            list(video.read_frame_windows(str(bad), windows=[[0]]))

        _assert_image_error(error.value, "Video could not be decoded.")

    def test_reads_sequentially_without_seeking(self, clip, monkeypatch):
        real_capture = cv2.VideoCapture
        sets = []

        class _Capture(_DelegatingCapture):
            def set(self, prop, value):
                sets.append((prop, value))
                return self._inner.set(prop, value)

        monkeypatch.setattr(
            video.cv2, "VideoCapture", lambda path: _Capture(real_capture(path))
        )

        read = list(video.read_frame_windows(str(clip), windows=[[8, 9], [10, 11]]))

        assert sets == []
        assert [len(frames) for frames in read] == [2, 2]


class TestDurationCap:
    def test_clip_over_the_cap_is_413(self, monkeypatch):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DURATION_SECONDS", 60.0)

        with pytest.raises(LegacyHTTPError) as error:
            video.ensure_clip_fits_the_duration_cap(frame_count=61, fps=1.0)

        assert error.value.status_code == 413
        assert error.value.message == (
            "Video runs 61.0 s. This server classifies at most 60 s in one request. "
            "Send a shorter clip, or raise MAX_VIDEO_DURATION_SECONDS on the server."
        )

    def test_clip_at_the_cap_passes(self, monkeypatch):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DURATION_SECONDS", 60.0)

        video.ensure_clip_fits_the_duration_cap(frame_count=600, fps=10.0)

    def test_negative_cap_disables_the_check(self, monkeypatch):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DURATION_SECONDS", -1.0)

        video.ensure_clip_fits_the_duration_cap(frame_count=10**9, fps=1.0)


async def _enter(video_type: str, value):
    async with video.video_source_path(video_type, value) as path:
        return path


class TestVideoSourcePathBase64:
    @pytest.mark.asyncio
    async def test_yields_a_decodable_temp_file_removed_afterwards(
        self, clip, temp_dir
    ):
        encoded = base64.b64encode(clip.read_bytes()).decode("ascii")

        async with video.video_source_path("base64", encoded) as path:
            assert Path(path).parent == temp_dir
            assert path.endswith(".video")
            assert Path(path).read_bytes() == clip.read_bytes()
            assert video.probe_video(path) == (_FPS, _FRAME_COUNT)

        assert not Path(path).exists()
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_temp_file_is_removed_when_the_body_raises(self, clip, temp_dir):
        encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
        seen = {}

        with pytest.raises(RuntimeError):
            async with video.video_source_path("base64", encoded) as path:
                seen["path"] = path
                raise RuntimeError("boom")

        assert not Path(seen["path"]).exists()

    @pytest.mark.asyncio
    async def test_temp_file_is_removed_on_cancellation(self, clip, temp_dir):
        encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
        seen = {}
        entered = asyncio.Event()

        async def _hold():
            async with video.video_source_path("base64", encoded) as path:
                seen["path"] = path
                entered.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(_hold())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not Path(seen["path"]).exists()

    @pytest.mark.asyncio
    async def test_oversized_payload_is_refused_before_decoding(
        self, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DOWNLOAD_SIZE_MB", 0)
        decoded = []
        monkeypatch.setattr(
            video.base64, "b64decode", lambda value: decoded.append(value)
        )

        with pytest.raises(LegacyHTTPError) as error:
            await _enter("base64", "QUJDRA==")

        assert error.value.status_code == 413
        assert error.value.message == "Video is larger than this server accepts."
        assert decoded == []
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_negative_cap_accepts_any_size(self, clip, temp_dir, monkeypatch):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DOWNLOAD_SIZE_MB", -1)
        encoded = base64.b64encode(clip.read_bytes()).decode("ascii")

        async with video.video_source_path("base64", encoded) as path:
            assert Path(path).read_bytes() == clip.read_bytes()

    @pytest.mark.asyncio
    async def test_malformed_base64_is_refused(self, temp_dir):
        with pytest.raises(LegacyHTTPError) as error:
            await _enter("base64", "@@@not base64@@@")

        _assert_image_error(error.value, "Video could not be decoded from base64.")
        assert list(temp_dir.iterdir()) == []


class TestVideoSourcePathTypes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("video_type", ["file", "multipart", "numpy", "", "URL"])
    async def test_other_types_are_refused(self, video_type, temp_dir):
        with pytest.raises(LegacyHTTPError) as error:
            await _enter(video_type, "x")

        _assert_image_error(
            error.value,
            f"Video type '{video_type}' is not supported, expected one of 'url' or "
            "'base64'.",
        )


def _fake_fetch(body: bytes = b"", error: Response = None):
    calls = []

    async def fetch_to_sink(url, *, sink, max_bytes, timeout_s, destination_policy):
        calls.append(
            {
                "url": url,
                "max_bytes": max_bytes,
                "timeout_s": timeout_s,
                "destination_policy": destination_policy,
            }
        )
        if body:
            sink(body[: len(body) // 2])
            sink(body[len(body) // 2 :])
        return error

    return calls, fetch_to_sink


class TestVideoSourcePathUrl:
    @pytest.fixture(autouse=True)
    def _online(self, monkeypatch):
        monkeypatch.setattr(video, "LEGACY_OFFLINE_MODE", False)
        monkeypatch.setattr(video, "ALLOW_URL_INPUT", True)
        monkeypatch.setattr(configuration, "ALLOW_NON_HTTPS_URL_INPUT", False)
        monkeypatch.setattr(configuration, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
        monkeypatch.setattr(
            configuration, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", None
        )
        monkeypatch.setattr(
            configuration, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", None
        )
        monkeypatch.setattr(configuration, "VALIDATE_IMAGE_URL_REDIRECTS", False)

    @pytest.mark.asyncio
    async def test_streams_the_clip_into_the_temp_file(
        self, clip, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DOWNLOAD_SIZE_MB", 2)
        monkeypatch.setattr(configuration, "VIDEO_DOWNLOAD_TIMEOUT_SECONDS", 7.5)
        calls, fetch = _fake_fetch(body=clip.read_bytes())
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        async with video.video_source_path(
            "url", " HTTPS://example.com/clip.mp4"
        ) as path:
            assert Path(path).parent == temp_dir
            assert Path(path).read_bytes() == clip.read_bytes()
            assert video.probe_video(path) == (_FPS, _FRAME_COUNT)

        assert not Path(path).exists()
        assert len(calls) == 1
        assert calls[0]["url"] == "https://example.com/clip.mp4"
        assert calls[0]["max_bytes"] == 2 * 1024 * 1024
        assert calls[0]["timeout_s"] == 7.5
        assert calls[0]["destination_policy"].validate_redirect is None

    @pytest.mark.asyncio
    async def test_negative_limits_lift_the_cap_and_the_timeout(
        self, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(configuration, "MAX_VIDEO_DOWNLOAD_SIZE_MB", -1)
        monkeypatch.setattr(configuration, "VIDEO_DOWNLOAD_TIMEOUT_SECONDS", -1.0)
        calls, fetch = _fake_fetch(body=b"clip")
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        await _enter("url", "https://example.com/clip.mp4")

        assert calls[0]["max_bytes"] is None
        assert calls[0]["timeout_s"] is None

    @pytest.mark.asyncio
    async def test_redirects_are_checked_with_the_url_rules_when_validating(
        self, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(configuration, "VALIDATE_IMAGE_URL_REDIRECTS", True)
        calls, fetch = _fake_fetch(body=b"clip")
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        await _enter("url", "https://example.com/clip.mp4")

        validate_redirect = calls[0]["destination_policy"].validate_redirect
        assert validate_redirect("https://cdn.example.com/x") == (
            "https://cdn.example.com/x",
            None,
        )
        _, refusal = validate_redirect("http://cdn.example.com/x")
        assert refusal.status_code == 400

    @pytest.mark.asyncio
    async def test_offline_mode_refuses_before_any_fetch(self, temp_dir, monkeypatch):
        monkeypatch.setattr(video, "LEGACY_OFFLINE_MODE", True)
        calls, fetch = _fake_fetch()
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        with pytest.raises(LegacyHTTPError) as error:
            await _enter("url", "https://example.com/clip.mp4")

        _assert_image_error(
            error.value, "Cannot load a video from URL while OFFLINE_MODE is enabled."
        )
        assert calls == []
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_disabled_url_input_refuses_before_any_fetch(
        self, temp_dir, monkeypatch
    ):
        monkeypatch.setattr(video, "ALLOW_URL_INPUT", False)
        calls, fetch = _fake_fetch()
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        with pytest.raises(LegacyHTTPError) as error:
            await _enter("url", "https://example.com/clip.mp4")

        _assert_image_error(
            error.value,
            "Providing images via URL is not supported in this configuration of "
            "`inference`.",
        )
        assert calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url,cause",
        [
            (
                "http://example.com/clip.mp4",
                "Providing images via non https:// URL is not supported in this "
                "configuration of `inference`.",
            ),
            (
                "https://localhost/clip.mp4",
                "Providing images via URL without FQDN is not supported in this "
                "configuration of `inference`.",
            ),
            ("not a url", "Provided image URL is invalid"),
        ],
    )
    async def test_url_rules_refuse_before_any_fetch(
        self, url, cause, temp_dir, monkeypatch
    ):
        calls, fetch = _fake_fetch()
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        with pytest.raises(LegacyHTTPError) as error:
            await _enter("url", url)

        _assert_image_error(error.value, cause)
        assert calls == []
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fetch_error,status,message",
        [
            (
                error_response(403, "URL_DESTINATION_FORBIDDEN", "x"),
                400,
                f"{_IMAGE_ERROR_PREFIX}URL points to a network destination that is "
                "not allowed.",
            ),
            (
                error_response(502, "URL_FETCH_FAILED", "x"),
                400,
                f"{_IMAGE_ERROR_PREFIX}Video could not be fetched from the URL.",
            ),
            (
                error_response(504, "URL_FETCH_TIMEOUT", "x"),
                400,
                f"{_IMAGE_ERROR_PREFIX}Video could not be fetched from the URL.",
            ),
            (
                error_response(403, "URL_INPUT_DISABLED", "x"),
                400,
                f"{_IMAGE_ERROR_PREFIX}Providing images via URL is not supported in "
                "this configuration of `inference`.",
            ),
            (
                error_response(413, "URL_CONTENT_TOO_LARGE", "x"),
                413,
                "Content is larger than this server accepts.",
            ),
            (
                error_response(403, "URL_REFUSED", "refused by the hop rule"),
                400,
                f"{_IMAGE_ERROR_PREFIX}refused by the hop rule",
            ),
        ],
    )
    async def test_fetch_errors_answer_like_legacy_and_leave_no_file(
        self, fetch_error, status, message, temp_dir, monkeypatch
    ):
        calls, fetch = _fake_fetch(body=b"partial", error=fetch_error)
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        with pytest.raises(LegacyHTTPError) as error:
            await _enter("url", "https://example.com/clip.mp4")

        assert error.value.status_code == status
        assert error.value.message == message
        assert list(temp_dir.iterdir()) == []

    @pytest.mark.asyncio
    async def test_unknown_fetch_error_keeps_its_status_and_description(
        self, temp_dir, monkeypatch
    ):
        calls, fetch = _fake_fetch(
            error=error_response(418, "SOMETHING_ELSE", "the teapot said no")
        )
        monkeypatch.setattr(video, "fetch_to_sink", fetch)

        with pytest.raises(LegacyHTTPError) as error:
            await _enter("url", "https://example.com/clip.mp4")

        assert error.value.status_code == 418
        assert error.value.message == "the teapot said no"
