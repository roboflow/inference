"""Bounded, owned video input for the local/threaded SAM3 embedding path."""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from inference_models.configuration import (
    SAM3_VIDEO_DECODE_BUFFER_BYTES,
    SAM3_VIDEO_MAX_SOURCE_PIXELS,
)
from inference_models.errors import ModelInputError
from inference_models.utils.video import VideoDecodeError, VideoSizeError, open_video

# Covers all tracker instances in this process, including model replacement.
_EMBED_ADMISSION = threading.Lock()


class VideoInputLimitError(ModelInputError):
    """A clip exceeds a configured input limit (HTTP 413)."""


@contextmanager
def embedding_admission(check_cancelled=None):
    """Admit one embedding operation until its consumer actually finishes.

    Args:
        check_cancelled: Optional callback that raises when work is abandoned.

    Yields:
        None: The process-wide embedding slot.
    """
    while not _EMBED_ADMISSION.acquire(timeout=0.1):
        if check_cancelled is not None:
            check_cancelled()
    try:
        if check_cancelled is not None:
            check_cancelled()
        yield
    finally:
        _EMBED_ADMISSION.release()


class VideoFrameSource:
    """Own a downloaded file until its model consumer finishes.

    This object is an internal in-process input, never a public filesystem path
    or an IPC payload. Cancellation closes an idle handle immediately; an active
    consumer keeps its handle until it exits, even if the pathname is unlinked.

    Args:
        path: Server-created file containing the guarded download/upload.
        max_frames: Maximum observed frames permitted by the model cache.
        buffer_bytes: Maximum original RGB bytes retained in one batch.
        max_pixels: Maximum source pixels of each decoded frame.
    """

    def __init__(
        self,
        path: str,
        *,
        max_frames: int,
        buffer_bytes: int = SAM3_VIDEO_DECODE_BUFFER_BYTES,
        max_pixels: int = SAM3_VIDEO_MAX_SOURCE_PIXELS,
    ):
        if min(max_frames, buffer_bytes, max_pixels) <= 0:
            raise ValueError("Video input limits must be positive")
        self.max_frames = max_frames
        self.buffer_bytes = buffer_bytes
        self.max_pixels = max_pixels
        self._file = Path(path).open("rb")
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._active = False

    def cancel(self) -> None:
        """Cancel input consumption and close the handle when no consumer owns it."""
        with self._lock:
            self._cancelled.set()
            if not self._active:
                self._file.close()

    def check_cancelled(self) -> None:
        """Raise when the HTTP caller has abandoned this input."""
        if self._cancelled.is_set():
            raise ModelInputError(message="Video embedding was cancelled.")

    @contextmanager
    def consume(self):
        """Claim the input, closing it only after actual consumer completion.

        Yields:
            VideoFrameSource: The owned source, supporting bounded batch passes.
        """
        with self._lock:
            self.check_cancelled()
            if self._active or self._file.closed:
                raise ModelInputError(message="Video input has already been consumed.")
            self._active = True
        try:
            yield self
        finally:
            with self._lock:
                self._file.close()
                self._active = False

    def batches(self, *, max_batch_frames: int) -> Iterator[List[np.ndarray]]:
        """Decode bounded RGB batches; another pass is allowed within consumption.

        Args:
            max_batch_frames: Frame-count ceiling in addition to the byte budget.

        Yields:
            Lists of equally sized RGB uint8 frames. Release each batch before
            requesting the next; this iterator retains no previous batch.

        Raises:
            ModelInputError: Invalid, inconsistent, or cancelled input.
            VideoInputLimitError: A pixel, byte, or observed frame-count violation.
        """
        if not self._active:
            raise ModelInputError(
                message="Video input must be claimed before decoding."
            )
        if max_batch_frames <= 0:
            raise ValueError("max_batch_frames must be positive")
        self.check_cancelled()
        self._file.seek(0)
        count = 0
        size: Optional[tuple] = None
        batch = []
        used = 0
        frame = None
        try:
            with open_video(self._file, max_pixels=self.max_pixels) as reader:
                if reader.frame_count > self.max_frames:
                    raise VideoInputLimitError(
                        message=(
                            f"The video has {reader.frame_count} frames; the most a clip "
                            f"may have is {self.max_frames}, which is what SAM3_VIDEO_CACHE_MAX_GB holds."
                        )
                    )
                while True:
                    self.check_cancelled()
                    frame = reader.read(
                        rgb=True,
                        max_bytes=self.buffer_bytes - used,
                        expected_size=size,
                    )
                    if frame is None:
                        break
                    count += 1
                    if count > self.max_frames:
                        raise VideoInputLimitError(
                            message=(
                                f"The video has more than {self.max_frames} frames."
                            )
                        )
                    size = frame.shape[:2]
                    frame_bytes = frame.nbytes
                    batch.append(frame)
                    used += frame_bytes
                    del frame
                    if (
                        len(batch) >= max_batch_frames
                        or used + frame_bytes > self.buffer_bytes
                    ):
                        yield batch
                        batch = []
                        used = 0
                if count < 2:
                    raise ModelInputError(message="A video needs at least two frames.")
                if batch:
                    yield batch
        except VideoSizeError as error:
            raise VideoInputLimitError(message=str(error)) from error
        except VideoDecodeError as error:
            raise ModelInputError(message=str(error)) from error
        finally:
            # Do not keep originals alive in an exception traceback.
            batch.clear()
            frame = None
