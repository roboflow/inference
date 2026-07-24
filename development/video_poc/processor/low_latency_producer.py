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


class LowLatencyRtspProducer(VideoFrameProducer):
    def __init__(self, url: str, options: Optional[Dict[str, str]] = None):
        import av

        self._url = url
        self._container = av.open(
            url,
            options={**DEFAULT_OPTIONS, **(options or {})},
            timeout=(10.0, 10.0),  # (open, read) seconds
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
        self._pending: Optional[np.ndarray] = None
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def grab(self) -> bool:
        try:
            for packet in self._demuxer:
                for frame in packet.decode():
                    self._pending = frame.to_ndarray(format="bgr24")
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
        image, self._pending = self._pending, None
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
