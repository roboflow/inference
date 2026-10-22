"""Probe and sequentially decode local video without server dependencies."""

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from inference_models.errors import VideoInputError, VideoProcessingLimitError


@dataclass
class VideoDecodeState:
    """Track decoded source-frame count and whether decoding reached EOF."""

    frame_count: int = 0
    reached_end: bool = False


def probe_video(
    path: Union[str, Path], *, check_cancelled: Optional[Callable[[], None]] = None
) -> Tuple[float, int]:
    """Report the clip's frame rate and frame count without decoding it.

    The frame count comes from the container header, which the sender writes
    and nothing verifies. Everything downstream sizes itself from it: the
    duration it implies decides how many windows get planned, and each window
    holds a frame index per sample. A header claiming far more frames than it
    has costs memory and time in proportion to the claim.

    An implausible header count is checked by sequential decoding. Timing
    uses nominal FPS; this does not provide variable-frame-rate timestamps.

    Args:
        path (Union[str, Path]): Local video file.
        check_cancelled (Optional[Callable]): Check during probing/counting.

    Returns:
        Tuple[float, int]: Nominal FPS and source-frame count.

    Raises:
        VideoInputError: Decoding, FPS or frame count is unusable.
        OSError: The path cannot be inspected.
    """
    if check_cancelled is not None:
        check_cancelled()
    if not Path(path).is_file():
        raise VideoInputError(f"Expected a local video file: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            message = "Video could not be decoded."
            raise VideoInputError(message)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(declared_count) if np.isfinite(declared_count) else 0
    finally:
        capture.release()
    if source_fps <= 0 or not np.isfinite(source_fps):
        message = "Video declares no usable frame rate."
        raise VideoInputError(message)
    if frame_count <= 0 or frame_count > Path(path).stat().st_size:
        frame_count = _count_frames(path=path, check_cancelled=check_cancelled)
    # OpenCV opens a still image as a one-frame video, so a JPEG reaches here
    # looking like a clip and would be classified as one, with a frame rate
    # nothing declared. An action is a change, and one frame cannot show one.
    if frame_count < 2:
        message = (
            f"Video holds {frame_count} frame(s). A clip needs at least two frames "
            "to hold an action. Send a video, not a still image."
        )
        raise VideoInputError(message)
    metadata = source_fps, frame_count
    return metadata


def read_frame_windows(
    path: Union[str, Path],
    *,
    windows: Sequence[Sequence[int]],
    max_frame_side: Optional[int] = None,
    frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    decode_state: Optional[VideoDecodeState] = None,
    window_end_frames: Optional[Sequence[int]] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
    max_retained_frame_bytes: Optional[int] = None,
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

    Args:
        path: Local video path.
        windows: Source-frame indices for each requested window.
        max_frame_side: Optional longest-side cap applied during RGB conversion.
        frame_transform: Optional model-owned transform applied once per decoded
            RGB frame, before storing it for overlapping windows.
        decode_state: Observed decoded count and whether decoding reached EOF.
        window_end_frames: Exclusive source endpoints to read before yielding.
            These include the real interval after a window's final sample.
        check_cancelled: Request control called during decoding and before yields.
        max_retained_frame_bytes: Optional prepared-pixel buffer limit.

    Yields:
        Lists of prepared RGB frames in each window's requested order.
    """
    if not windows:
        return

    needed = Counter(int(index) for window in windows for index in window)
    last_of = [max((int(i) for i in window), default=-1) for window in windows]
    if window_end_frames is not None:
        if len(window_end_frames) != len(windows):
            raise ValueError("Each window needs one exclusive source endpoint")
        last_of = [max(last, end - 1) for last, end in zip(last_of, window_end_frames)]
    state = decode_state if decode_state is not None else VideoDecodeState()
    by_index: Dict[int, np.ndarray] = {}
    emitted = 0
    retained_bytes = 0

    def _window_frames(index: int) -> List[np.ndarray]:
        frames = [by_index[i] for i in windows[index] if i in by_index]
        return frames

    if not Path(path).is_file():
        raise VideoInputError(f"Expected a local video file: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            message = "Video could not be decoded."
            raise VideoInputError(message)

        position = 0
        stop = max(last_of, default=-1)
        while emitted < len(windows) and position <= stop:
            if check_cancelled is not None and position % 32 == 0:
                check_cancelled()
            read_succeeded, frame = capture.read()
            if not read_succeeded:
                state.reached_end = True
                break
            state.frame_count = position + 1
            if position in needed:
                rgb_frame = _to_rgb(frame=frame, max_side=max_frame_side)
                if frame_transform is not None:
                    rgb_frame = frame_transform(rgb_frame)
                retained_bytes += rgb_frame.nbytes
                if (
                    max_retained_frame_bytes is not None
                    and retained_bytes > max_retained_frame_bytes
                ):
                    raise VideoProcessingLimitError(
                        "Prepared video frames exceed max_retained_frame_bytes."
                    )
                by_index[position] = rgb_frame
                del rgb_frame
            while emitted < len(windows) and last_of[emitted] <= position:
                if check_cancelled is not None:
                    check_cancelled()
                yield _window_frames(emitted)
                # Reference counts avoid rescanning every later window on every yield.
                for index in windows[emitted]:
                    needed[index] -= 1
                    if needed[index] == 0:
                        del needed[index]
                        released = by_index.pop(index, None)
                        if released is not None:
                            retained_bytes -= released.nbytes
                        del released
                emitted += 1
            position += 1
    finally:
        capture.release()
    # A truncated video leaves the trailing windows short, and the caller
    # drops the ones that fall under the model's minimum.
    while emitted < len(windows):
        if check_cancelled is not None:
            check_cancelled()
        yield _window_frames(emitted)
        emitted += 1


def read_frames(
    path: Union[str, Path],
    *,
    frame_indices: Sequence[int],
    max_frame_side: Optional[int] = None,
) -> List[np.ndarray]:
    """Read one window and release its decoder.

    Args:
        path (Union[str, Path]): Local video file.
        frame_indices (Sequence[int]): Requested source-frame indices.
        max_frame_side (Optional[int]): Longest-side cap before retaining RGB.

    Returns:
        List[np.ndarray]: Available frames in the requested order.
    """
    with closing(
        read_frame_windows(
            path=path, windows=[frame_indices], max_frame_side=max_frame_side
        )
    ) as windows:
        frames = next(windows, [])
    return frames


def _to_rgb(frame: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = max_side / max(height, width) if max_side and max_side > 0 else 1.0
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    return rgb_frame


def _count_frames(
    path: Union[str, Path], *, check_cancelled: Optional[Callable[[], None]] = None
) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = 0
        while True:
            if check_cancelled is not None and count % 32 == 0:
                check_cancelled()
            if not capture.read()[0]:
                break
            count += 1
    finally:
        capture.release()
    return count
