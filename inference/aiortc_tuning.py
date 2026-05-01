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

BROWSER_INPUT_REMB_START_BITRATE = 6_000_000
BROWSER_INPUT_REMB_MIN_BITRATE = 2_000_000
BROWSER_INPUT_REMB_MAX_BITRATE = 12_000_000

H264_NVENC_ENCODER = "h264_nvenc"
H264_NVENC_OPTION_SETS = (
    {
        "preset": "p1",
        "tune": "ull",
        "rc": "cbr",
        "zerolatency": "1",
        "delay": "0",
        "bf": "0",
    },
    {
        "preset": "llhp",
        "rc": "cbr",
        "zerolatency": "1",
        "delay": "0",
        "bf": "0",
    },
    {
        "preset": "fast",
        "rc": "cbr",
        "bf": "0",
    },
    {},
)


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
    _patch_h264_nvenc_encoder(h264)
    _patch_encoder_timing_logging(h264.H264Encoder, "h264")
    _patch_encoder_timing_logging(vpx.Vp8Encoder, "vp8")
    _patch_receiver_remb_estimator()


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


def _patch_h264_nvenc_encoder(h264_module: Any) -> None:
    encoder_cls = h264_module.H264Encoder
    if getattr(encoder_cls, "_roboflow_h264_nvenc_patched", False):
        return

    original_encode_frame = encoder_cls._encode_frame

    def _encode_frame(self: Any, frame: Any, force_keyframe: bool) -> Any:
        if self.codec and _h264_codec_needs_recreate(self, frame):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            frame.pict_type = h264_module.av.video.frame.PictureType.I
        else:
            frame.pict_type = h264_module.av.video.frame.PictureType.NONE

        if self.codec is None:
            try:
                self.codec, options = _create_h264_nvenc_context(
                    h264_module,
                    frame,
                    self.target_bitrate,
                )
                LOGGER.warning(
                    "[WEBRTC_NVENC] active encoder=%s input=%sx%s target_bps=%s options=%s",
                    H264_NVENC_ENCODER,
                    frame.width,
                    frame.height,
                    self.target_bitrate,
                    options,
                )
            except Exception as error:
                self.codec = None
                LOGGER.warning(
                    "[WEBRTC_NVENC] unavailable falling_back=libx264 input=%sx%s target_bps=%s error=%r",
                    frame.width,
                    frame.height,
                    self.target_bitrate,
                    error,
                )
                yield from original_encode_frame(self, frame, force_keyframe)
                return

        data_to_send = b""
        try:
            for package in self.codec.encode(frame):
                data_to_send += bytes(package)
        except Exception as error:
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None
            LOGGER.warning(
                "[WEBRTC_NVENC] encode_failed falling_back=libx264 input=%sx%s target_bps=%s error=%r",
                frame.width,
                frame.height,
                self.target_bitrate,
                error,
            )
            yield from original_encode_frame(self, frame, force_keyframe)
            return

        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    encoder_cls._encode_frame = _encode_frame
    encoder_cls._roboflow_h264_nvenc_patched = True
    LOGGER.warning("Applied aiortc WebRTC H.264 NVENC preference")


def _h264_codec_needs_recreate(encoder: Any, frame: Any) -> bool:
    codec_bitrate = getattr(encoder.codec, "bit_rate", 0) or 0
    target_bitrate = encoder.target_bitrate
    bitrate_changed = (
        codec_bitrate <= 0
        or abs(target_bitrate - codec_bitrate) / codec_bitrate > 0.1
    )
    return (
        frame.width != encoder.codec.width
        or frame.height != encoder.codec.height
        or bitrate_changed
    )


def _create_h264_nvenc_context(
    h264_module: Any,
    frame: Any,
    target_bitrate: int,
) -> Any:
    last_error = None
    for options in H264_NVENC_OPTION_SETS:
        try:
            codec = h264_module.av.CodecContext.create(H264_NVENC_ENCODER, "w")
            codec.width = frame.width
            codec.height = frame.height
            codec.bit_rate = target_bitrate
            codec.pix_fmt = "yuv420p"
            codec.framerate = h264_module.fractions.Fraction(
                h264_module.MAX_FRAME_RATE,
                1,
            )
            codec.time_base = h264_module.fractions.Fraction(
                1,
                h264_module.MAX_FRAME_RATE,
            )
            codec.options = dict(options)
            try:
                codec.profile = "Baseline"
            except Exception as profile_error:
                LOGGER.warning(
                    "[WEBRTC_NVENC] profile_unavailable encoder=%s error=%r",
                    H264_NVENC_ENCODER,
                    profile_error,
                )
            codec.open()
            return codec, options
        except Exception as error:
            last_error = error

    raise RuntimeError(f"Could not open {H264_NVENC_ENCODER}: {last_error!r}")


def _patch_encoder_timing_logging(encoder_cls: Any, codec_name: str) -> None:
    if getattr(encoder_cls, "_roboflow_encode_timing_patched", False):
        return

    original_encode = encoder_cls.encode

    def encode(self: Any, frame: Any, force_keyframe: bool = False) -> Any:
        started_at = time.perf_counter()
        payloads, timestamp = original_encode(self, frame, force_keyframe)
        encode_ms = (time.perf_counter() - started_at) * 1000

        frame_count = getattr(self, "_roboflow_encode_frame_count", 0) + 1
        self._roboflow_encode_frame_count = frame_count
        payload_bytes = sum(len(payload) for payload in payloads)
        av_codec = getattr(self, "codec", None)
        codec_encoder = getattr(av_codec, "name", None) or "uninitialized"
        codec_bitrate = getattr(av_codec, "bit_rate", None)
        target_bitrate = getattr(self, "target_bitrate", None)
        width = getattr(frame, "width", None)
        height = getattr(frame, "height", None)

        if frame_count <= 3 or frame_count % 30 == 0 or encode_ms >= 50.0:
            LOGGER.warning(
                "[WEBRTC_ENCODE_TIMING] codec=%s encoder=%s frame=%d input=%sx%s "
                "force_keyframe=%s encode_ms=%.1f payloads=%d payload_bytes=%d "
                "target_bps=%s codec_bps=%s",
                codec_name,
                codec_encoder,
                frame_count,
                width,
                height,
                force_keyframe,
                encode_ms,
                len(payloads),
                payload_bytes,
                target_bitrate,
                codec_bitrate,
            )

        return payloads, timestamp

    encoder_cls.encode = encode
    encoder_cls._roboflow_encode_timing_patched = True
    LOGGER.warning("Applied aiortc WebRTC encode timing for codec=%s", codec_name)


def _patch_receiver_remb_estimator() -> None:
    import aiortc.rate as rate

    estimator_cls = rate.RemoteBitrateEstimator
    if getattr(estimator_cls, "_roboflow_remb_tuning_patched", False):
        return

    original_init = estimator_cls.__init__
    original_add = estimator_cls.add

    def __init__(self: Any) -> None:
        original_init(self)
        now_ms = int(time.time() * 1000)
        self.rate_control.set_estimate(BROWSER_INPUT_REMB_START_BITRATE, now_ms)
        self.rate_control.latest_estimated_throughput = (
            BROWSER_INPUT_REMB_START_BITRATE
        )
        self._roboflow_last_remb_log_at = 0.0
        self._roboflow_last_logged_remb_bitrate = None
        LOGGER.warning(
            "[WEBRTC_REMB] seeded receiver estimate start_bps=%s min_bps=%s max_bps=%s",
            BROWSER_INPUT_REMB_START_BITRATE,
            BROWSER_INPUT_REMB_MIN_BITRATE,
            BROWSER_INPUT_REMB_MAX_BITRATE,
        )

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
        applied_bitrate = min(
            max(requested_bitrate, BROWSER_INPUT_REMB_MIN_BITRATE),
            BROWSER_INPUT_REMB_MAX_BITRATE,
        )
        if applied_bitrate != requested_bitrate:
            self.rate_control.current_bitrate = applied_bitrate

        now = time.monotonic()
        last_logged_at = getattr(self, "_roboflow_last_remb_log_at", 0.0)
        last_logged_bitrate = getattr(self, "_roboflow_last_logged_remb_bitrate", None)
        should_log = (
            last_logged_bitrate != applied_bitrate
            or (now - last_logged_at) >= 5.0
        )
        if should_log:
            LOGGER.warning(
                "[WEBRTC_REMB] requested_bps=%s applied_bps=%s ssrcs=%s",
                requested_bitrate,
                applied_bitrate,
                ssrcs,
            )
            self._roboflow_last_remb_log_at = now
            self._roboflow_last_logged_remb_bitrate = applied_bitrate

        return applied_bitrate, ssrcs

    estimator_cls.__init__ = __init__
    estimator_cls.add = add
    estimator_cls._roboflow_remb_tuning_patched = True
    LOGGER.warning(
        "Applied aiortc WebRTC receiver REMB tuning: start_bps=%s min_bps=%s max_bps=%s",
        BROWSER_INPUT_REMB_START_BITRATE,
        BROWSER_INPUT_REMB_MIN_BITRATE,
        BROWSER_INPUT_REMB_MAX_BITRATE,
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
