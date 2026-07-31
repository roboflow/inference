"""Jetson GStreamer RTSP(S) frame producer (ENT-1544 B2)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Union

import cv2
from numpy import ndarray

from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)
from inference.core.interfaces.camera.gstreamer_rtsp_pipeline import (
    build_gstreamer_rtsp_pipeline,
    gstreamer_rtsp_capture_available,
)
from inference.core.interfaces.camera.rtsp_tls import (
    GST_SSL_CA_CERTIFICATE_ENV_VAR,
    is_rtsps_url,
)

__all__ = [
    "GStreamerRtspVideoFrameProducer",
    "gstreamer_rtsp_capture_available",
    "should_use_gstreamer_rtsp_producer",
]


def should_use_gstreamer_rtsp_producer(video: Union[str, int]) -> bool:
    """Return True when the GStreamer RTSP(S) producer should be preferred."""
    return isinstance(video, str) and is_rtsps_url(video)


class GStreamerRtspVideoFrameProducer(VideoFrameProducer):
    """Decode RTSP(S) via a GStreamer pipeline (OpenCV CAP_GSTREAMER backend)."""

    def __init__(self, video: str):
        self._source_ref = video
        self._apply_ca_bundle_env()
        pipeline = build_gstreamer_rtsp_pipeline(video)
        self.stream = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    @staticmethod
    def _apply_ca_bundle_env() -> None:
        ca_bundle = os.getenv(GST_SSL_CA_CERTIFICATE_ENV_VAR)
        if ca_bundle:
            os.environ.setdefault("SSL_CERT_FILE", ca_bundle)

    def isOpened(self) -> bool:
        return self.stream.isOpened()

    def grab(self) -> bool:
        return self.stream.grab()

    def retrieve(self) -> Tuple[bool, ndarray]:
        return self.stream.retrieve()

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        for property_id, value in properties.items():
            cv2_id = getattr(cv2, "CAP_PROP_" + property_id.upper())
            self.stream.set(cv2_id, value)

    def discover_source_properties(self) -> SourceProperties:
        width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        is_file = total_frames > 0 and os.path.exists(self._source_ref)
        timestamp_created = None
        if is_file:
            file_length_seconds = total_frames / fps
            last_modified = datetime.fromtimestamp(os.path.getmtime(self._source_ref))
            timestamp_created = last_modified - timedelta(seconds=file_length_seconds)

        return SourceProperties(
            width=width,
            height=height,
            total_frames=total_frames,
            is_file=is_file,
            fps=fps,
            timestamp_created=timestamp_created,
        )

    def release(self) -> None:
        self.stream.release()
