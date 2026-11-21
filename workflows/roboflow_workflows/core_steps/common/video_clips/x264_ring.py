"""`X264Ring`: the default software encoder backend (PyAV libx264).

On the caller's thread `submit()` optionally downscales to `max_width`
(`cv2.resize`), crops to even dimensions and converts BGR to I420 with
`cv2.cvtColor` (releases the GIL; never PyAV `reformat`). The I420 array is a
new buffer, so the ring never keeps a reference to the pipeline's frame (in-place
visualisations cannot leak into the clip).

The encoder thread writes the planes into one reused `av.VideoFrame` and runs
libx264 with ``preset=ultrafast``, ``tune=zerolatency``, one thread, forced IDR
each output second (``forced-idr=1``, no scene-cut keyframes) and ``yuv420p``.
Packets are Annex B with in-band SPS/PPS, which the shared remux turns into an
MP4.
"""

from fractions import Fraction
from typing import Any, List, Optional

import cv2
import numpy as np
from roboflow_workflows.core_steps.common.video_clips.grid import PTS_CLOCK_RATE
from roboflow_workflows.core_steps.common.video_clips.ring import (
    BaseEncodedRing,
    EncodedOutput,
    PreparedFrame,
    import_av,
)

X264_ENCODER = "libx264"
X264_CRF = "23"
NOMINAL_ENCODER_FPS = 30
UNBOUNDED_GOP = 1_000_000


class X264Ring(BaseEncodedRing):
    """Rolling libx264 ring for one `video_identifier`.

    Constructor arguments are those of `BaseEncodedRing`.
    """

    backend = "x264"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._codec: Any = None
        self._video_frame: Any = None
        self._picture_type_i: Any = None
        self._picture_type_none: Any = None

    def _prepare_frame(self, image: Any) -> Optional[PreparedFrame]:
        pixels = _as_bgr(image.numpy_image)
        height, width = pixels.shape[:2]
        if self.max_width and width > self.max_width:
            target_height = max(2, round(height * self.max_width / width))
            pixels = cv2.resize(
                pixels,
                (self.max_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
            height, width = pixels.shape[:2]
        even_width = width - width % 2
        even_height = height - height % 2
        if even_width < 2 or even_height < 2:
            return None

        i420 = cv2.cvtColor(pixels[:even_height, :even_width], cv2.COLOR_BGR2YUV_I420)
        prepared = PreparedFrame(
            payload=i420, width=even_width, height=even_height, nbytes=i420.nbytes
        )

        return prepared

    def _encode_frame(
        self, frame: PreparedFrame, *, pts: int, force_keyframe: bool
    ) -> List[EncodedOutput]:
        if self._codec is None:
            self._open_encoder(width=frame.width, height=frame.height)

        _write_i420_planes(self._video_frame, frame.payload, frame.height)
        self._video_frame.pts = pts
        self._video_frame.pict_type = (
            self._picture_type_i if force_keyframe else self._picture_type_none
        )
        outputs = [
            EncodedOutput(payload=bytes(packet), is_keyframe=bool(packet.is_keyframe))
            for packet in self._codec.encode(self._video_frame)
        ]

        return outputs

    def _flush_encoder(self) -> List[EncodedOutput]:
        if self._codec is None:
            return []

        outputs = [
            EncodedOutput(payload=bytes(packet), is_keyframe=bool(packet.is_keyframe))
            for packet in self._codec.encode(None)
        ]

        return outputs

    def _close_encoder(self) -> None:
        self._codec = None
        self._video_frame = None

    def _open_encoder(self, *, width: int, height: int) -> None:
        av = import_av()
        codec = av.CodecContext.create(X264_ENCODER, "w")
        codec.width = width
        codec.height = height
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, PTS_CLOCK_RATE)
        codec.framerate = Fraction(NOMINAL_ENCODER_FPS, 1)
        codec.gop_size = UNBOUNDED_GOP
        codec.thread_count = 1
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": X264_CRF,
            "threads": "1",
            "forced-idr": "1",
            "x264-params": "scenecut=0",
        }
        codec.open()
        self._codec = codec
        self._video_frame = av.VideoFrame(width, height, "yuv420p")
        self._picture_type_i = av.video.frame.PictureType.I
        self._picture_type_none = av.video.frame.PictureType.NONE


def _as_bgr(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim == 2:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    if pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if pixels.shape[2] == 1:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)

    return pixels


def _write_i420_planes(video_frame: Any, i420: np.ndarray, height: int) -> None:
    # I420 layout from cv2: Y (h x w), then U and V (h/2 x w/2 each) packed
    # row-major into the remaining h/2 rows.
    width = i420.shape[1]
    chroma_size = (height // 2) * (width // 2)
    flat = i420.reshape(-1)
    y_plane = i420[:height]
    u_plane = flat[height * width : height * width + chroma_size].reshape(
        height // 2, width // 2
    )
    v_plane = flat[height * width + chroma_size :].reshape(height // 2, width // 2)
    for plane, data in zip(video_frame.planes, (y_plane, u_plane, v_plane)):
        target = np.frombuffer(plane, dtype=np.uint8).reshape(
            plane.height, plane.line_size
        )
        target[:, : data.shape[1]] = data
