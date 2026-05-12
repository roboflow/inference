import time
from typing import Any, Iterator, List, Optional

from inference.core import logger

H264_LIBX264_TIMING_SAMPLE_INTERVAL = 120


def add_h264_libx264_logging() -> None:
    """Log aiortc's default H.264 encode and codec recreation timings."""
    import aiortc.codecs.h264 as h264

    encoder_cls = h264.H264Encoder
    if getattr(encoder_cls, "_roboflow_h264_libx264_logging_patched", False):
        return

    original_encode_frame = encoder_cls._encode_frame

    def _encode_frame(self: Any, frame: Any, force_keyframe: bool) -> Iterator[bytes]:
        frame_count = getattr(self, "_roboflow_h264_libx264_frame_count", 0) + 1
        self._roboflow_h264_libx264_frame_count = frame_count

        recreate_reason = _recreate_reason(self, frame)
        started_at = time.perf_counter()
        packages = list(original_encode_frame(self, frame, force_keyframe))
        encode_ms = (time.perf_counter() - started_at) * 1000

        if recreate_reason is not None:
            logger.info(
                "[WEBRTC_LIBX264] recreated_encoder reason=%s frame=%s "
                "encode_ms=%.2f payload_bytes=%s resolution=%sx%s "
                "target_bps=%s codec_bps=%s",
                recreate_reason,
                frame_count,
                encode_ms,
                _payload_bytes(packages),
                frame.width,
                frame.height,
                getattr(self, "target_bitrate", None),
                _codec_bitrate(self),
            )

        if recreate_reason is not None or _should_log_sample(frame_count):
            logger.info(
                "[WEBRTC_LIBX264] encode_sample frame=%s encode_ms=%.2f "
                "payload_bytes=%s resolution=%sx%s target_bps=%s "
                "codec_bps=%s encoder=%s force_keyframe=%s",
                frame_count,
                encode_ms,
                _payload_bytes(packages),
                frame.width,
                frame.height,
                getattr(self, "target_bitrate", None),
                _codec_bitrate(self),
                getattr(getattr(self, "codec", None), "name", None),
                force_keyframe,
            )

        yield from packages

    encoder_cls._encode_frame = _encode_frame
    encoder_cls._roboflow_h264_libx264_logging_patched = True
    logger.info("[WEBRTC_LIBX264] patched aiortc H264Encoder timing logging")


def _recreate_reason(encoder: Any, frame: Any) -> Optional[str]:
    codec = getattr(encoder, "codec", None)
    if codec is None:
        return "initial"
    if frame.width != codec.width or frame.height != codec.height:
        return "resolution"

    codec_bitrate = getattr(codec, "bit_rate", None)
    target_bitrate = getattr(encoder, "target_bitrate", None)
    if codec_bitrate and target_bitrate:
        bitrate_change_ratio = abs(target_bitrate - codec_bitrate) / codec_bitrate
        if bitrate_change_ratio > 0.1:
            return "bitrate"
    return None


def _should_log_sample(frame_count: int) -> bool:
    return frame_count == 1 or frame_count % H264_LIBX264_TIMING_SAMPLE_INTERVAL == 0


def _payload_bytes(packages: List[bytes]) -> int:
    return sum(len(package) for package in packages)


def _codec_bitrate(encoder: Any) -> Optional[int]:
    codec = getattr(encoder, "codec", None)
    if codec is None:
        return None
    return getattr(codec, "bit_rate", None)
