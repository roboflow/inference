import time
from typing import Any, List, Tuple

from inference.core import logger

VP8_TIMING_SAMPLE_INTERVAL = 120


def patch_vp8_timing_logs() -> None:
    """Add sampled timing logs for aiortc VP8 encode/decode diagnostics."""
    import aiortc.codecs.vpx as vpx

    if getattr(vpx.Vp8Encoder, "_roboflow_vp8_timing_patched", False):
        return

    original_encode = vpx.Vp8Encoder.encode
    original_decode = vpx.Vp8Decoder.decode

    def encode(
        self: Any, frame: Any, force_keyframe: bool = False
    ) -> Tuple[List[bytes], int]:
        frame_count = _increment_counter(self, "_roboflow_vp8_encode_frame_count")
        should_log = _should_log_sample(frame_count)
        started_at = time.perf_counter() if should_log else None

        payloads, timestamp = original_encode(self, frame, force_keyframe)

        if frame_count == 1:
            logger.info(
                "[WEBRTC_VP8] encoder_used encoder=libvpx resolution=%sx%s "
                "target_bps=%s codec_bps=%s",
                getattr(frame, "width", None),
                getattr(frame, "height", None),
                getattr(self, "target_bitrate", None),
                getattr(getattr(self, "codec", None), "bit_rate", None),
            )

        if should_log and started_at is not None:
            logger.info(
                "[WEBRTC_VP8] encode_sample frame=%s encode_ms=%.2f "
                "payload_bytes=%s packets=%s resolution=%sx%s target_bps=%s "
                "codec_bps=%s timestamp=%s",
                frame_count,
                (time.perf_counter() - started_at) * 1000,
                sum(len(payload) for payload in payloads),
                len(payloads),
                getattr(frame, "width", None),
                getattr(frame, "height", None),
                getattr(self, "target_bitrate", None),
                getattr(getattr(self, "codec", None), "bit_rate", None),
                timestamp,
            )

        return payloads, timestamp

    def decode(self: Any, encoded_frame: Any) -> List[Any]:
        packet_count = _increment_counter(self, "_roboflow_vp8_decode_packet_count")
        should_log = _should_log_sample(packet_count)
        started_at = time.perf_counter() if should_log else None

        frames = original_decode(self, encoded_frame)

        if packet_count == 1:
            logger.info("[WEBRTC_VP8] decoder_used decoder=libvpx")

        if should_log and started_at is not None:
            width, height = _first_frame_resolution(frames)
            logger.info(
                "[WEBRTC_VP8] decode_sample packet=%s decode_ms=%.2f "
                "encoded_bytes=%s decoded_frames=%s resolution=%sx%s timestamp=%s",
                packet_count,
                (time.perf_counter() - started_at) * 1000,
                len(getattr(encoded_frame, "data", b"")),
                len(frames),
                width,
                height,
                getattr(encoded_frame, "timestamp", None),
            )

        return frames

    vpx.Vp8Encoder.encode = encode
    vpx.Vp8Decoder.decode = decode
    vpx.Vp8Encoder._roboflow_vp8_timing_patched = True
    vpx.Vp8Decoder._roboflow_vp8_timing_patched = True
    logger.info("[WEBRTC_VP8] patched aiortc VP8 encoder/decoder timing logs")


def _increment_counter(instance: Any, attribute: str) -> int:
    count = getattr(instance, attribute, 0) + 1
    setattr(instance, attribute, count)
    return count


def _should_log_sample(count: int) -> bool:
    return count == 1 or count % VP8_TIMING_SAMPLE_INTERVAL == 0


def _first_frame_resolution(frames: List[Any]) -> Tuple[Any, Any]:
    if not frames:
        return None, None
    return getattr(frames[0], "width", None), getattr(frames[0], "height", None)
