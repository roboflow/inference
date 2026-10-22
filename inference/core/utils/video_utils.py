import base64
import binascii
import contextlib
import os
import tempfile
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

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
from inference_models.errors import VideoInputError
from inference_models.video import io as _video_io
from inference_models.video.io import VideoDecodeState

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
    """Probe a local file, translating package input errors to HTTP errors.

    Args:
        path (str): Local video path.

    Returns:
        Tuple[float, int]: Nominal source FPS and frame count.
    """
    try:
        metadata = _video_io.probe_video(path)
    except VideoInputError as error:
        raise InputImageLoadError(
            message=str(error), public_message=str(error)
        ) from error
    return metadata


def read_frame_windows(
    path: str,
    windows: Sequence[Sequence[int]],
    max_frame_side: Optional[int] = None,
    frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    decode_state: Optional[VideoDecodeState] = None,
    window_end_frames: Optional[Sequence[int]] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> Iterator[List[np.ndarray]]:
    """Delegate sequential local decoding while preserving server error types.

    Args:
        path (str): Local video path.
        windows (Sequence[Sequence[int]]): Requested source indices per window.
        max_frame_side (Optional[int]): Longest-side cap before retaining frames.
        frame_transform (Optional[Callable]): Model-owned RGB storage transform.
        decode_state (Optional[VideoDecodeState]): Mutable observed decode state.
        window_end_frames (Optional[Sequence[int]]): Exclusive decode endpoints.
        check_cancelled (Optional[Callable]): Cooperative request control.

    Yields:
        List[np.ndarray]: Prepared RGB frames for each requested window.
    """
    try:
        with contextlib.closing(
            _video_io.read_frame_windows(
                path=path,
                windows=windows,
                max_frame_side=max_frame_side,
                frame_transform=frame_transform,
                decode_state=decode_state,
                window_end_frames=window_end_frames,
                check_cancelled=check_cancelled,
            )
        ) as decoded:
            yield from decoded
    except VideoInputError as error:
        raise InputImageLoadError(
            message=str(error), public_message=str(error)
        ) from error


def read_frames(
    path: str, frame_indices: Sequence[int], max_frame_side: Optional[int] = None
) -> List[np.ndarray]:
    """Read one window using the shared local decoder.

    Args:
        path (str): Local video path.
        frame_indices (Sequence[int]): Requested source indices.
        max_frame_side (Optional[int]): Longest-side cap.

    Returns:
        List[np.ndarray]: Available RGB frames in requested order.
    """
    with contextlib.closing(
        read_frame_windows(
            path=path, windows=[frame_indices], max_frame_side=max_frame_side
        )
    ) as windows:
        frames = next(windows, [])
    return frames
