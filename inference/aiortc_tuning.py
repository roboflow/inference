import logging
import time
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)

H264_BITRATES = {
    "DEFAULT_BITRATE": 6_000_000,
    "MIN_BITRATE": 2_000_000,
    "MAX_BITRATE": 12_000_000,
}

VP8_BITRATES = {
    "DEFAULT_BITRATE": 4_000_000,
    "MIN_BITRATE": 1_500_000,
    "MAX_BITRATE": 8_000_000,
}


def _apply_codec_bitrates(
    module: Any,
    bitrates: Dict[str, int],
) -> Dict[str, int]:
    applied: Dict[str, int] = {}
    for constant_name, value in bitrates.items():
        setattr(module, constant_name, value)
        applied[constant_name] = value
    return applied


def apply_aiortc_bitrate_limits() -> None:
    """Patch aiortc codec bitrate constants for WebRTC streaming tests.

    aiortc does not expose a public sender bitrate API. These constants are read by
    the encoder when RTP senders start, so this must run before peer connections
    begin sending video.
    """
    import aiortc.codecs.h264 as h264
    import aiortc.codecs.vpx as vpx

    h264_applied = _apply_codec_bitrates(h264, H264_BITRATES)
    vp8_applied = _apply_codec_bitrates(vpx, VP8_BITRATES)

    LOGGER.warning(
        "Applied aiortc WebRTC bitrate tuning: h264=%s vp8=%s",
        h264_applied,
        vp8_applied,
    )
    _patch_encoder_bitrate_logging(h264.H264Encoder, "h264")
    _patch_encoder_bitrate_logging(vpx.Vp8Encoder, "vp8")


def _patch_encoder_bitrate_logging(encoder_cls: Any, codec_name: str) -> None:
    if getattr(encoder_cls, "_roboflow_bitrate_logging_patched", False):
        return

    target_bitrate_property = encoder_cls.target_bitrate
    getter = target_bitrate_property.fget
    setter = target_bitrate_property.fset
    if getter is None or setter is None:
        return

    def target_bitrate(self: Any, bitrate: int) -> None:
        previous_bitrate = getter(self)
        setter(self, bitrate)
        applied_bitrate = getter(self)
        now = time.monotonic()
        last_logged_at = getattr(self, "_roboflow_last_bitrate_log_at", 0.0)
        last_logged_bitrate = getattr(self, "_roboflow_last_logged_bitrate", None)
        should_log = (
            last_logged_bitrate != applied_bitrate
            or (now - last_logged_at) >= 5.0
        )
        if should_log:
            LOGGER.warning(
                "[WEBRTC_BITRATE] codec=%s requested_bps=%s applied_bps=%s previous_bps=%s",
                codec_name,
                bitrate,
                applied_bitrate,
                previous_bitrate,
            )
            self._roboflow_last_bitrate_log_at = now
            self._roboflow_last_logged_bitrate = applied_bitrate

    encoder_cls.target_bitrate = property(getter, target_bitrate)
    encoder_cls._roboflow_bitrate_logging_patched = True


def prefer_h264_for_peer_connection(peer_connection: Any) -> bool:
    from aiortc import RTCRtpSender

    codecs = RTCRtpSender.getCapabilities("video").codecs
    h264_codecs = [codec for codec in codecs if codec.mimeType.lower() == "video/h264"]
    if not h264_codecs:
        LOGGER.warning("Cannot prefer H.264 because aiortc has no H.264 capability")
        return False

    preferred_codecs = h264_codecs + [
        codec for codec in codecs if codec not in h264_codecs
    ]
    applied = False
    for transceiver in peer_connection.getTransceivers():
        if transceiver.kind != "video":
            continue
        transceiver.setCodecPreferences(preferred_codecs)
        applied = True

    if applied:
        LOGGER.warning("Preferred H.264 for WebRTC video transceivers")
    return applied
