"""Cooperative control for a request that processes video windows."""

import math
from time import monotonic
from typing import Callable, Optional


class VideoProcessingCancelledError(Exception):
    """The server detected that the caller disconnected."""


class VideoProcessingTimeoutError(Exception):
    """The request exhausted its processing time budget."""


class VideoProcessingControl:
    """Stop new work at explicit boundaries while an active model call finishes."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        is_disconnected: Optional[Callable[[], bool]] = None
    ):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Video processing timeout must be positive and finite")
        self.deadline = monotonic() + timeout_seconds
        self.is_disconnected = is_disconnected

    def check(self) -> None:
        """Stop processing when its deadline expires or its caller disconnects.

        Raises:
            VideoProcessingTimeoutError: The processing deadline expired.
            VideoProcessingCancelledError: The caller disconnected.
        """
        if monotonic() >= self.deadline:
            raise VideoProcessingTimeoutError(
                "Action-recognition processing exceeded its deadline."
            )
        elif self.is_disconnected is not None and self.is_disconnected():
            raise VideoProcessingCancelledError(
                "Action-recognition caller disconnected."
            )
