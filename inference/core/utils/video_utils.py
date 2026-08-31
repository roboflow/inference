import base64
import binascii
import contextlib
import os
import tempfile
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from inference.core.env import OFFLINE_MODE
from inference.core.exceptions import InputImageLoadError, InvalidImageTypeDeclared
from inference.core.utils.image_utils import (
    _ensure_url_input_allowed,
    _fetch_image_bytes_from_url,
    _validate_url_destination,
)

VIDEO_TYPE_URL = "url"
VIDEO_TYPE_BASE64 = "base64"


@contextlib.contextmanager
def video_source_path(video_type: str, value: str) -> Iterator[str]:
    """Put the clip on disk and yield its path.

    OpenCV reads containers from a file, not from a buffer, so a clip that
    arrives in a request is written to a temporary file for the length of the
    request. URL fetching runs the same address policy as image input.

    A URL is the preferred transport. Both branches hold the whole clip in
    memory before it reaches disk, and base64 adds a third again on the wire
    and another copy in the parsed request, so base64 suits short clips only.
    """
    if video_type == VIDEO_TYPE_URL:
        if OFFLINE_MODE:
            message = "Cannot load a video from URL while OFFLINE_MODE is enabled."
            raise InputImageLoadError(message=message, public_message=message)
        _ensure_url_input_allowed()
        prepared_url = _validate_url_destination(value=value)
        payload = _fetch_image_bytes_from_url(prepared_url=prepared_url)
    elif video_type == VIDEO_TYPE_BASE64:
        try:
            payload = base64.b64decode(value)
        except (binascii.Error, TypeError, ValueError) as error:
            message = "Video could not be decoded from base64."
            raise InputImageLoadError(
                message=f"{message} Details: {error}", public_message=message
            ) from error
    else:
        message = (
            f"Video type '{video_type}' is not supported, expected one of "
            f"'{VIDEO_TYPE_URL}' or '{VIDEO_TYPE_BASE64}'."
        )
        raise InvalidImageTypeDeclared(message=message, public_message=message)

    handle, path = tempfile.mkstemp(suffix=".video")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(payload)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def probe_video(path: str) -> Tuple[float, int]:
    """Report the clip's frame rate and frame count without decoding it."""
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
    if frame_count <= 0:
        frame_count = _count_frames(path=path)
    return source_fps, frame_count


def read_frames(
    path: str, frame_indices: Sequence[int], max_frame_side: Optional[int] = None
) -> List[np.ndarray]:
    """Read the named frames as RGB, longest side capped at ``max_frame_side``.

    ``max_frame_side`` of ``None`` reads the frames at their own size, which is
    what a model that never trained on a frame side needs.

    Frames are read in order rather than sought. A sought frame and a
    sequentially decoded one are not the same pixels for every codec, and the
    model's answer moves with them.
    """
    wanted = sorted(set(int(index) for index in frame_indices))
    if not wanted:
        return []
    by_index = {}
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            message = "Video could not be decoded."
            raise InputImageLoadError(message=message, public_message=message)
        last_wanted = wanted[-1]
        position = 0
        pending = set(wanted)
        while position <= last_wanted:
            read_succeeded, frame = capture.read()
            if not read_succeeded:
                break
            if position in pending:
                by_index[position] = _to_rgb(frame=frame, max_side=max_frame_side)
                pending.discard(position)
            position += 1
    finally:
        capture.release()
    return [by_index[index] for index in frame_indices if index in by_index]


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
