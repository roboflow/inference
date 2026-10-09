"""Video input of the legacy action-recognition route.

A clip arrives as a URL or base64 and lands in a temporary file, because
OpenCV reads containers from a path. The file is probed for its frame rate
and frame count, checked against the duration cap, and read window by window
in one sequential pass. Every OpenCV call here blocks; callers run the
decoding functions off the event loop.
"""

import asyncio
import base64
import binascii
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fastapi import Response

from inference_server import configuration
from inference_server.configuration import ALLOW_URL_INPUT, LEGACY_OFFLINE_MODE
from inference_server.framework.input_parsers.url_fetch import (
    DestinationPolicy,
    fetch_to_sink,
)
from inference_server.legacy.common import _check_url, image_load_error
from inference_server.legacy.errors import LegacyHTTPError

VIDEO_TYPE_URL = "url"
VIDEO_TYPE_BASE64 = "base64"
_VIDEO_OFFLINE_ERROR = "Cannot load a video from URL while OFFLINE_MODE is enabled."
_URL_INPUT_DISABLED_ERROR = (
    "Providing images via URL is not supported in this configuration of `inference`."
)
_URL_DESTINATION_ERROR = "URL points to a network destination that is not allowed."
_VIDEO_FETCH_ERROR = "Video could not be fetched from the URL."
_URL_CONTENT_TOO_LARGE_ERROR = "Content is larger than this server accepts."
_VIDEO_TOO_LARGE_ERROR = "Video is larger than this server accepts."
_BASE64_DECODE_ERROR = "Video could not be decoded from base64."
_VIDEO_DECODE_ERROR = "Video could not be decoded."
_NO_FRAME_RATE_ERROR = "Video declares no usable frame rate."
_URL_REFUSED_CODE = "URL_REFUSED"
_URL_FETCH_FAILED_CODES = frozenset({"URL_FETCH_FAILED", "URL_FETCH_TIMEOUT"})


@contextlib.asynccontextmanager
async def video_source_path(video_type: str, value: str) -> AsyncIterator[str]:
    """Put the clip on disk and yield its path for the length of the request.

    A URL runs the address rules of image input and streams into the file
    without a copy in memory; base64 is size-checked before it is decoded.
    The file is removed when the block ends, whatever ends it.

    Args:
        video_type: ``url`` or ``base64``.
        value: The URL, or the base64 text of the clip.

    Yields:
        Path of the temporary file holding the clip.

    Raises:
        LegacyHTTPError: 400 for an unsupported type, a refused or failed URL
            or undecodable base64; 413 for a clip over the download size cap.
    """
    if video_type not in (VIDEO_TYPE_URL, VIDEO_TYPE_BASE64):
        raise image_load_error(
            f"Video type '{video_type}' is not supported, expected one of "
            f"'{VIDEO_TYPE_URL}' or '{VIDEO_TYPE_BASE64}'."
        )
    if video_type == VIDEO_TYPE_URL:
        if LEGACY_OFFLINE_MODE:
            raise image_load_error(_VIDEO_OFFLINE_ERROR)
        if not ALLOW_URL_INPUT:
            raise image_load_error(_URL_INPUT_DISABLED_ERROR)
        prepared_url, refusal = _check_url(value)
        if refusal is not None:
            raise _video_fetch_error(refusal)
    else:
        payload = await asyncio.to_thread(_decode_base64_video, value)

    handle, path = tempfile.mkstemp(suffix=".video")
    try:
        if video_type == VIDEO_TYPE_URL:
            with os.fdopen(handle, "wb") as file:
                await _stream_url_into(prepared_url, sink=file.write)
        else:
            await asyncio.to_thread(_write_payload, handle, payload)
        yield path
    finally:
        with contextlib.suppress(OSError):
            Path(path).unlink()


def _write_payload(handle: int, payload: bytes) -> None:
    with os.fdopen(handle, "wb") as file:
        file.write(payload)


async def _stream_url_into(prepared_url: str, *, sink) -> None:
    validate_redirect = (
        _check_url if configuration.VALIDATE_IMAGE_URL_REDIRECTS else None
    )
    error = await fetch_to_sink(
        prepared_url,
        sink=sink,
        max_bytes=_max_download_bytes(),
        timeout_s=_download_timeout(),
        destination_policy=DestinationPolicy(validate_redirect=validate_redirect),
    )
    if error is not None:
        raise _video_fetch_error(error)


def _decode_base64_video(value: str) -> bytes:
    max_bytes = _max_download_bytes()
    if max_bytes is not None and len(value) // 4 * 3 > max_bytes:
        raise LegacyHTTPError(413, _VIDEO_TOO_LARGE_ERROR)
    try:
        payload = base64.b64decode(value)
    except (binascii.Error, TypeError, ValueError) as error:
        raise image_load_error(_BASE64_DECODE_ERROR) from error

    return payload


def _video_fetch_error(response: Response) -> LegacyHTTPError:
    try:
        body = json.loads(response.body)
    except Exception:
        body = {}
    error_code = body.get("error_code")
    if error_code == _URL_REFUSED_CODE:
        return image_load_error(body["description"])
    if error_code == "URL_INPUT_DISABLED":
        return image_load_error(_URL_INPUT_DISABLED_ERROR)
    if error_code == "URL_DESTINATION_FORBIDDEN":
        return image_load_error(_URL_DESTINATION_ERROR)
    if error_code in _URL_FETCH_FAILED_CODES:
        return image_load_error(_VIDEO_FETCH_ERROR)
    if error_code == "URL_CONTENT_TOO_LARGE":
        return LegacyHTTPError(413, _URL_CONTENT_TOO_LARGE_ERROR)

    return LegacyHTTPError(
        response.status_code, body.get("description") or _VIDEO_FETCH_ERROR
    )


def _download_timeout() -> Optional[float]:
    if configuration.VIDEO_DOWNLOAD_TIMEOUT_SECONDS < 0:
        return None
    return configuration.VIDEO_DOWNLOAD_TIMEOUT_SECONDS


def _max_download_bytes() -> Optional[int]:
    if configuration.MAX_VIDEO_DOWNLOAD_SIZE_MB < 0:
        return None
    return configuration.MAX_VIDEO_DOWNLOAD_SIZE_MB * 1024 * 1024


def ensure_clip_fits_the_duration_cap(frame_count: int, fps: float) -> None:
    """Refuse a clip longer than the deployment classifies in one request.

    Args:
        frame_count: Frames the clip holds.
        fps: Frame rate the clip declares.

    Raises:
        LegacyHTTPError: 413 when the clip runs past
            ``MAX_VIDEO_DURATION_SECONDS``; a negative setting disables it.
    """
    if configuration.MAX_VIDEO_DURATION_SECONDS < 0:
        return
    duration_seconds = frame_count / fps
    if duration_seconds <= configuration.MAX_VIDEO_DURATION_SECONDS:
        return

    message = (
        f"Video runs {duration_seconds:.1f} s. This server classifies at most "
        f"{configuration.MAX_VIDEO_DURATION_SECONDS:.0f} s in one request. Send a "
        f"shorter clip, or raise MAX_VIDEO_DURATION_SECONDS on the server."
    )
    raise LegacyHTTPError(413, message)


def probe_video(path: str) -> Tuple[float, int]:
    """Report the clip's frame rate and frame count without decoding it.

    The frame count comes from the container header, which nothing verifies.
    A frame cannot occupy less than one byte, so a header claiming more
    frames than the file has bytes is recounted by decoding the file.

    Args:
        path: File holding the clip.

    Returns:
        ``(fps, frame_count)`` as the container declares them.

    Raises:
        LegacyHTTPError: 400 when the file does not decode, declares no
            usable frame rate, or holds fewer than two frames.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise image_load_error(_VIDEO_DECODE_ERROR)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if source_fps <= 0 or not np.isfinite(source_fps):
        raise image_load_error(_NO_FRAME_RATE_ERROR)
    if frame_count <= 0 or frame_count > Path(path).stat().st_size:
        frame_count = _count_frames(path=path)
    if frame_count < 2:
        raise image_load_error(
            f"Video holds {frame_count} frame(s). A clip needs at least two frames "
            "to hold an action. Send a video, not a still image."
        )

    return source_fps, frame_count


def read_frame_windows(
    path: str,
    windows: Sequence[Sequence[int]],
    max_frame_side: Optional[int] = None,
) -> Iterator[List[np.ndarray]]:
    """Read every window's frames in one sequential pass, a window at a time.

    Frames are decoded in order, never sought, since a sought frame and a
    sequentially decoded one are not the same pixels for every codec. A
    window is yielded as soon as its last frame is read, and only frames a
    later window still asks for are held. A truncated clip yields short
    trailing windows.

    Args:
        path: File holding the clip.
        windows: Frame indices of each window, in clip order.
        max_frame_side: Longest side to downscale frames to; ``None`` keeps
            their own size.

    Yields:
        The RGB ``HWC`` ``uint8`` frames of each window, in window order.

    Raises:
        LegacyHTTPError: 400 when the file does not decode.
    """
    if not windows:
        return
    needed = {int(index) for window in windows for index in window}
    last_of = [max((int(i) for i in window), default=-1) for window in windows]
    by_index: Dict[int, np.ndarray] = {}
    emitted = 0

    def _window_frames(index: int) -> List[np.ndarray]:
        return [by_index[i] for i in windows[index] if i in by_index]

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise image_load_error(_VIDEO_DECODE_ERROR)
        position = 0
        stop = max(needed) if needed else -1
        while emitted < len(windows) and position <= stop:
            read_succeeded, frame = capture.read()
            if not read_succeeded:
                break
            if position in needed:
                by_index[position] = _to_rgb(frame=frame, max_side=max_frame_side)
            while emitted < len(windows) and last_of[emitted] <= position:
                yield _window_frames(emitted)
                emitted += 1
                still_wanted = {int(i) for window in windows[emitted:] for i in window}
                by_index = {i: f for i, f in by_index.items() if i in still_wanted}
            position += 1
    finally:
        capture.release()
    while emitted < len(windows):
        yield _window_frames(emitted)
        emitted += 1


def _to_rgb(frame: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = max_side / max(height, width) if max_side and max_side > 0 else 1.0
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(frame[:, :, ::-1])


def _count_frames(path: str) -> int:
    capture = cv2.VideoCapture(path)
    try:
        count = 0
        while capture.read()[0]:
            count += 1
    finally:
        capture.release()
    return count
