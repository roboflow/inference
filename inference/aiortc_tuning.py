import logging
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
