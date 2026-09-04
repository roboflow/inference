import base64
import binascii
import contextlib
import os
import tempfile
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from requests import RequestException

from inference.core.env import (
    MAX_VIDEO_DOWNLOAD_SIZE_MB,
    OFFLINE_MODE,
    VIDEO_DOWNLOAD_TIMEOUT_SECONDS,
)
from inference.core.exceptions import (
    InputImageLoadError,
    InvalidImageTypeDeclared,
    PayloadTooLargeError,
)
from inference.core.utils.image_utils import (
    _ensure_url_input_allowed,
    _fetch_image_bytes_from_url,
    _validate_url_destination,
)
from inference.core.utils.url_input import URLAddressNotAllowedError

VIDEO_TYPE_URL = "url"
VIDEO_TYPE_BASE64 = "base64"


@contextlib.contextmanager
def video_source_path(video_type: str, value: str) -> Iterator[str]:
    """Put the clip on disk and yield its path.

    OpenCV reads containers from a file, not from a buffer, so a clip that
    arrives in a request is written to a temporary file for the length of the
    request. URL fetching runs the same address policy as image input, and
    streams to that file rather than holding the clip in memory.

    A URL is the preferred transport. Base64 arrives whole in the parsed
    request and adds a third again on the wire, so it suits short clips only.
    """
    if video_type not in (VIDEO_TYPE_URL, VIDEO_TYPE_BASE64):
        message = (
            f"Video type '{video_type}' is not supported, expected one of "
            f"'{VIDEO_TYPE_URL}' or '{VIDEO_TYPE_BASE64}'."
        )
        raise InvalidImageTypeDeclared(message=message, public_message=message)
    if video_type == VIDEO_TYPE_URL:
        if OFFLINE_MODE:
            message = "Cannot load a video from URL while OFFLINE_MODE is enabled."
            raise InputImageLoadError(message=message, public_message=message)
        _ensure_url_input_allowed()
        prepared_url = _validate_url_destination(value=value)
    else:
        max_bytes = _max_download_bytes()
        # Four base64 characters carry three bytes, so the decoded size is
        # known before decoding. Checking it first keeps an oversized clip
        # from being expanded into memory just to be rejected.
        if max_bytes is not None and len(value) // 4 * 3 > max_bytes:
            message = "Video is larger than this server accepts."
            raise PayloadTooLargeError(message=message, public_message=message)
        try:
            payload = base64.b64decode(value)
        except (binascii.Error, TypeError, ValueError) as error:
            message = "Video could not be decoded from base64."
            raise InputImageLoadError(
                message=f"{message} Details: {error}", public_message=message
            ) from error

    handle, path = tempfile.mkstemp(suffix=".video")
    try:
        with os.fdopen(handle, "wb") as file:
            if video_type == VIDEO_TYPE_URL:
                try:
                    _fetch_image_bytes_from_url(
                        prepared_url=prepared_url,
                        sink=file.write,
                        max_bytes=_max_download_bytes(),
                        request_timeout=_download_timeout(),
                    )
                except URLAddressNotAllowedError as error:
                    message = "URL points to a network destination that is not allowed."
                    raise InputImageLoadError(
                        message=f"{message} Details: {error}", public_message=message
                    ) from error
                except (RequestException, ConnectionError) as error:
                    message = "Video could not be fetched from the URL."
                    raise InputImageLoadError(
                        message=f"{message} Details: {error}", public_message=message
                    ) from error
            else:
                file.write(payload)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _download_timeout() -> Optional[float]:
    if VIDEO_DOWNLOAD_TIMEOUT_SECONDS < 0:
        return None
    return VIDEO_DOWNLOAD_TIMEOUT_SECONDS


def _max_download_bytes() -> Optional[int]:
    """The clip size this deployment accepts, or ``None`` when uncapped."""
    if MAX_VIDEO_DOWNLOAD_SIZE_MB < 0:
        return None
    return MAX_VIDEO_DOWNLOAD_SIZE_MB * 1024 * 1024


def probe_video(path: str) -> Tuple[float, int]:
    """Report the clip's frame rate and frame count without decoding it.

    The frame count comes from the container header, which the sender writes
    and nothing verifies. Everything downstream sizes itself from it: the
    duration it implies decides how many windows get planned, and each window
    holds a frame index per sample. A header claiming far more frames than it
    has costs memory and time in proportion to the claim.

    A frame cannot occupy less than one byte, so the file size is a ceiling
    the header cannot honestly exceed. A claim above it gets counted instead,
    and counting reads the real file, whose size is already capped.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            message = "Video could not be decoded."
            raise InputImageLoadError(message=message, public_message=message)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if source_fps <= 0 or not np.isfinite(source_fps):
        message = "Video declares no usable frame rate."
        raise InputImageLoadError(message=message, public_message=message)
    if frame_count <= 0 or frame_count > os.path.getsize(path):
        frame_count = _count_frames(path=path)
    # OpenCV opens a still image as a one-frame video, so a JPEG reaches here
    # looking like a clip and would be classified as one, with a frame rate
    # nothing declared. An action is a change, and one frame cannot show one.
    if frame_count < 2:
        message = (
            f"Video holds {frame_count} frame(s). A clip needs at least two frames "
            "to hold an action. Send a video, not a still image."
        )
        raise InputImageLoadError(message=message, public_message=message)
    return source_fps, frame_count


def read_frame_windows(
    path: str,
    windows: Sequence[Sequence[int]],
    max_frame_side: Optional[int] = None,
) -> Iterator[List[np.ndarray]]:
    """Read every window's frames in one pass, yielding a window at a time.

    Windows tile forward, so one walk of the video serves all of them. Reading
    each window on its own capture re-decodes everything the windows before it
    already decoded, which grows with the square of the clip length.

    Frames are read in order rather than sought. A sought frame and a
    sequentially decoded one are not the same pixels for every codec, and the
    model's answer moves with them. ``max_frame_side`` of ``None`` reads the
    frames at their own size, which is what a model that never trained on a
    frame side needs.
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
            message = "Video could not be decoded."
            raise InputImageLoadError(message=message, public_message=message)
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
                # Hold only what a window still to come asks for.
                still_wanted = {int(i) for window in windows[emitted:] for i in window}
                by_index = {i: f for i, f in by_index.items() if i in still_wanted}
            position += 1
    finally:
        capture.release()
    # A truncated video leaves the trailing windows short, and the caller
    # drops the ones that fall under the model's minimum.
    while emitted < len(windows):
        yield _window_frames(emitted)
        emitted += 1


def read_frames(
    path: str, frame_indices: Sequence[int], max_frame_side: Optional[int] = None
) -> List[np.ndarray]:
    """Read one window's frames. See :func:`read_frame_windows`."""
    for frames in read_frame_windows(
        path=path, windows=[list(frame_indices)], max_frame_side=max_frame_side
    ):
        return frames
    return []


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
