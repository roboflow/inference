"""PyAV-based VideoFrameProducer for low-latency RTSP ingest.

Why this exists: ffmpeg's h264 decoder holds a frame-reorder buffer sized from
the stream's DPB (~16 frames = ~530ms at 30fps) unless AV_CODEC_FLAG_LOW_DELAY
is set on the *codec* context. OpenCV's OPENCV_FFMPEG_CAPTURE_OPTIONS only
reaches the *format* context, so no cv2 configuration can disable that buffer
(measured: cv2 = 586ms, ffmpeg CLI with -flags low_delay = ~90ms on the same
stream). PyAV exposes the codec context directly, so we can set low_delay and
single-threaded decode and read frames in-process with no subprocess pipe.

Used by the processor for mode=stream jobs via VideoSource's producer-factory
path (VideoSourceIdentifier accepts Callable[[], VideoFrameProducer]).
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np

from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    "rtsp_transport": "tcp",
    "fflags": "nobuffer",
}

DEFAULT_OPEN_ATTEMPTS = 10
DEFAULT_OPEN_RETRY_DELAY_SECONDS = 0.5


def _retryable_open_error(av_module, error: Exception) -> bool:
    """Return whether a fresh RTSP publisher can plausibly recover on retry."""
    retryable_names = (
        "ConnectionRefusedError",
        "EOFError",
        "InvalidDataError",
        "TimeoutError",
    )
    retryable_types = tuple(
        candidate
        for candidate in (
            getattr(getattr(av_module, "error", None), name, None)
            for name in retryable_names
        )
        if isinstance(candidate, type)
    )
    return bool(retryable_types) and isinstance(error, retryable_types)


def _open_rtsp_with_retry(
    av_module,
    url: str,
    options: Dict[str, str],
    attempts: int,
    retry_delay_seconds: float,
):
    """Open a just-published relay path without leaking its credentialed URL."""
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            return av_module.open(
                url,
                options=options,
                timeout=(10.0, 10.0),  # (open, read) seconds
            )
        except Exception as error:
            if attempt >= attempts or not _retryable_open_error(av_module, error):
                raise
            logger.info(
                "low-latency RTSP publisher not ready (%s, attempt %d/%d)",
                type(error).__name__,
                attempt,
                attempts,
            )
            time.sleep(max(0.0, retry_delay_seconds))


class LowLatencyRtspProducer(VideoFrameProducer):
    def __init__(
        self,
        url: str,
        options: Optional[Dict[str, str]] = None,
        open_attempts: int = DEFAULT_OPEN_ATTEMPTS,
        open_retry_delay_seconds: float = DEFAULT_OPEN_RETRY_DELAY_SECONDS,
    ):
        import av

        self._url = url
        self._container = _open_rtsp_with_retry(
            av,
            url,
            {**DEFAULT_OPTIONS, **(options or {})},
            open_attempts,
            open_retry_delay_seconds,
        )
        self._stream = self._container.streams.video[0]
        codec_ctx = self._stream.codec_context
        try:
            from av.codec.context import Flags

            codec_ctx.flags |= Flags.low_delay
        except Exception:  # older PyAV: raw AV_CODEC_FLAG_LOW_DELAY bit
            codec_ctx.flags |= 1 << 19
        # frame-threaded decode adds thread_count-1 frames of delay
        codec_ctx.thread_count = 1
        self._demuxer = self._container.demux(self._stream)
        # Keep the decoded AV frame until the consumer explicitly retrieves it.
        # VideoSource's source-side FPS limiter calls grab() for every encoded
        # frame but retrieve() only for the selected stride.  Converting here
        # would therefore materialise every 60 FPS source frame into a host BGR
        # array even when a maxFps=5 job keeps only one in twelve.
        self._pending = None
        self._open = True

    @property
    def source_stream_metadata(self) -> Dict[str, object]:
        """Return bounded, non-secret evidence for the opened encoded stream."""
        codec_ctx = self._stream.codec_context
        rate = self._stream.average_rate or self._stream.guessed_rate
        codec_name = getattr(codec_ctx, "name", None)
        if codec_name is None:
            codec_name = getattr(getattr(codec_ctx, "codec", None), "name", None)
        metadata = {
            "width": int(codec_ctx.width),
            "height": int(codec_ctx.height),
        }
        if codec_name:
            metadata["codec"] = str(codec_name)[:64]
        if rate:
            metadata["fps"] = float(rate)
            numerator = getattr(rate, "numerator", None)
            denominator = getattr(rate, "denominator", None)
            if numerator is not None and denominator:
                metadata["fpsNumerator"] = int(numerator)
                metadata["fpsDenominator"] = int(denominator)
        return metadata

    def isOpened(self) -> bool:
        return self._open

    def grab(self) -> bool:
        try:
            for packet in self._demuxer:
                for frame in packet.decode():
                    self._pending = frame
                    return True
            return False
        except Exception as error:
            logger.warning("low-latency producer read failed: %s", error)
            self._open = False
            return False

    def retrieve(self) -> Tuple[bool, np.ndarray]:
        if self._pending is None:
            if not self.grab():
                return False, None
        frame, self._pending = self._pending, None
        image = frame.to_ndarray(format="bgr24")
        return True, image

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        pass  # cv2 CAP_PROP_* knobs don't apply; latency knobs are in __init__

    def discover_source_properties(self) -> SourceProperties:
        codec_ctx = self._stream.codec_context
        fps = self._stream.average_rate or self._stream.guessed_rate
        return SourceProperties(
            width=codec_ctx.width,
            height=codec_ctx.height,
            total_frames=-1,
            is_file=False,
            fps=float(fps) if fps else 30.0,
            is_reconnectable=True,
        )

    def release(self):
        self._open = False
        try:
            self._container.close()
        except Exception:
            pass
