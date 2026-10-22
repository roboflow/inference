"""Explicit resource limits and cooperative control for local video processing."""

import math
from dataclasses import dataclass, fields
from time import monotonic
from typing import Callable, Optional

from inference_models.errors import (
    VideoProcessingCancelledError,
    VideoProcessingTimeoutError,
)


@dataclass(frozen=True)
class VideoProcessingLimits:
    """Positive resource caps, independent of server environment variables.

    Attributes:
        max_windows (int): Maximum number of retained window plans.
        max_sampled_frames (int): Maximum total planned sample indices.
        max_retained_frame_bytes (int): Maximum prepared decoder-buffer bytes.
            Decoder internals, model weights/activations and caller-held output
            are outside this bound.
        max_candidates (int): Maximum candidates in a collected result, or one
            incremental window. Also caps model spans before mapping.
        max_timeline_entries (int): Maximum entries in a collected timeline,
            or filtered predictions in one incremental window.
        max_result_bytes (int): Maximum encoded result data, or one window's data.
    """

    max_windows: int = 10_000
    max_sampled_frames: int = 1_000_000
    max_retained_frame_bytes: int = 1024 * 1024 * 1024
    max_candidates: int = 250_000
    max_timeline_entries: int = 100_000
    max_result_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{item.name} must be a positive integer")


class VideoProcessingControl:
    """Check cancellation/deadlines at safe boundaries, never inside GPU work.

    Args:
        timeout_seconds (Optional[float]): Positive processing budget, starting
            when this control is constructed; None disables the deadline.
        is_cancelled (Optional[Callable[[], bool]]): Caller cancellation signal,
            for example a threading.Event's is_set method.
    """

    def __init__(
        self,
        *,
        timeout_seconds: Optional[float] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("Video processing timeout must be positive and finite")
        self.deadline = (
            None if timeout_seconds is None else monotonic() + timeout_seconds
        )
        self.is_cancelled = is_cancelled

    def check(self) -> None:
        """Raise when the deadline expires or the caller requests cancellation.

        Raises:
            VideoProcessingTimeoutError: The processing deadline expired.
            VideoProcessingCancelledError: The caller requested cancellation.
        """
        if self.deadline is not None and monotonic() >= self.deadline:
            raise VideoProcessingTimeoutError(
                "Action-recognition processing exceeded its deadline."
            )
        if self.is_cancelled is not None and self.is_cancelled():
            raise VideoProcessingCancelledError(
                "Action-recognition processing was cancelled."
            )
