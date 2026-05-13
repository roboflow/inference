import time
from typing import Any, Dict, Mapping, Optional

from inference.core import logger

WEBRTC_START_BITRATE_BPS = 6_000_000
WEBRTC_MIN_BITRATE_BPS = 2_000_000
WEBRTC_MAX_BITRATE_BPS = 16_000_000

CODEC_BITRATE_LIMITS_BPS = {
    "DEFAULT_BITRATE": WEBRTC_START_BITRATE_BPS,
    "MIN_BITRATE": WEBRTC_MIN_BITRATE_BPS,
    "MAX_BITRATE": WEBRTC_MAX_BITRATE_BPS,
}

_BITRATE_TUNING_APPLIED = False


def apply_aiortc_bitrate_tuning() -> None:
    """Apply WebRTC bitrate defaults before aiortc peer connections are created."""
    global _BITRATE_TUNING_APPLIED

    if _BITRATE_TUNING_APPLIED:
        return

    import aiortc.codecs.h264 as h264
    import aiortc.codecs.vpx as vpx

    h264_limits = _apply_codec_bitrate_limits(h264, CODEC_BITRATE_LIMITS_BPS)
    vp8_limits = _apply_codec_bitrate_limits(vpx, CODEC_BITRATE_LIMITS_BPS)
    _patch_receiver_remb_estimator()
    _BITRATE_TUNING_APPLIED = True

    logger.info(
        "[WEBRTC_BITRATE] applied aiortc bitrate tuning h264=%s vp8=%s "
        "remb_start_bps=%s remb_min_bps=%s remb_max_bps=%s",
        h264_limits,
        vp8_limits,
        WEBRTC_START_BITRATE_BPS,
        WEBRTC_MIN_BITRATE_BPS,
        WEBRTC_MAX_BITRATE_BPS,
    )


def prefer_h264_for_peer_connection(peer_connection: Any) -> bool:
    """Prefer H.264 when the browser offers it, keeping other codecs as fallback."""
    from aiortc import RTCRtpSender

    codecs = RTCRtpSender.getCapabilities("video").codecs
    h264_codecs = [codec for codec in codecs if codec.mimeType.lower() == "video/h264"]
    if not h264_codecs:
        logger.info("[WEBRTC_BITRATE] H.264 codec preference unavailable")
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
        logger.info("[WEBRTC_BITRATE] preferred H.264 for video transceivers")
    return applied


def _apply_codec_bitrate_limits(
    module: Any,
    bitrate_limits: Mapping[str, int],
) -> Dict[str, int]:
    applied = {}
    for constant_name, value in bitrate_limits.items():
        setattr(module, constant_name, value)
        applied[constant_name] = value
    return applied


def _patch_receiver_remb_estimator(rate_module: Optional[Any] = None) -> None:
    if rate_module is None:
        import aiortc.rate as rate_module

    estimator_cls = rate_module.RemoteBitrateEstimator
    if getattr(estimator_cls, "_roboflow_remb_tuning_patched", False):
        return

    original_init = estimator_cls.__init__
    original_add = estimator_cls.add

    def __init__(self: Any) -> None:
        original_init(self)
        now_ms = int(time.time() * 1000)
        self.rate_control.set_estimate(WEBRTC_START_BITRATE_BPS, now_ms)
        self.rate_control.latest_estimated_throughput = WEBRTC_START_BITRATE_BPS

    def add(
        self: Any,
        arrival_time_ms: int,
        abs_send_time: int,
        payload_size: int,
        ssrc: int,
    ) -> Any:
        remb = original_add(
            self,
            arrival_time_ms,
            abs_send_time,
            payload_size,
            ssrc,
        )
        if remb is None:
            return None

        requested_bitrate, ssrcs = remb
        applied_bitrate = _clamp_bitrate(
            requested_bitrate,
            min_bitrate=WEBRTC_MIN_BITRATE_BPS,
            max_bitrate=WEBRTC_MAX_BITRATE_BPS,
        )
        if applied_bitrate != requested_bitrate:
            self.rate_control.current_bitrate = applied_bitrate
        return applied_bitrate, ssrcs

    estimator_cls.__init__ = __init__
    estimator_cls.add = add
    estimator_cls._roboflow_remb_tuning_patched = True


def _clamp_bitrate(
    bitrate: int,
    min_bitrate: int,
    max_bitrate: int,
) -> int:
    return min(max(bitrate, min_bitrate), max_bitrate)
