"""Helpers for WebRTC codec negotiation."""

from typing import TYPE_CHECKING, List

from inference_sdk.utils.logging import get_logger
from inference_sdk.webrtc.config import VideoCodecPreference

if TYPE_CHECKING:
    from aiortc import RTCPeerConnection, RTCRtpCodecCapability


logger = get_logger("webrtc.codec_preferences")

_VIDEO_CODEC_MIME_TYPES = {
    "h264": "video/h264",
    "vp8": "video/vp8",
}


def apply_video_codec_preference(
    peer_connection: "RTCPeerConnection",
    video_codec: VideoCodecPreference,
) -> bool:
    """Prefer a video codec for video transceivers before offer creation."""
    if video_codec == "auto":
        return False

    preferred_mime_type = _VIDEO_CODEC_MIME_TYPES[video_codec]

    from aiortc import RTCRtpSender

    codecs = RTCRtpSender.getCapabilities("video").codecs
    if not _has_codec_with_mime_type(codecs, preferred_mime_type):
        logger.warning(
            "WebRTC video codec preference unavailable: %s",
            video_codec,
        )
        return False

    preferred_codecs = _order_codecs_by_mime_type(codecs, preferred_mime_type)
    applied = False
    for transceiver in peer_connection.getTransceivers():
        if transceiver.kind != "video":
            continue
        transceiver.setCodecPreferences(preferred_codecs)
        applied = True

    if applied:
        logger.debug("Preferred WebRTC video codec: %s", video_codec)
    return applied


def _order_codecs_by_mime_type(
    codecs: List["RTCRtpCodecCapability"],
    preferred_mime_type: str,
) -> List["RTCRtpCodecCapability"]:
    preferred_mime_type = preferred_mime_type.lower()
    preferred_codecs = [
        codec for codec in codecs if codec.mimeType.lower() == preferred_mime_type
    ]
    if not preferred_codecs:
        return list(codecs)

    return preferred_codecs + [
        codec for codec in codecs if codec.mimeType.lower() != preferred_mime_type
    ]


def _has_codec_with_mime_type(
    codecs: List["RTCRtpCodecCapability"],
    mime_type: str,
) -> bool:
    mime_type = mime_type.lower()
    return any(codec.mimeType.lower() == mime_type for codec in codecs)
