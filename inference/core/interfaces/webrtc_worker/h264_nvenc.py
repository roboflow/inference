import fractions
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from inference.core import logger

H264_NVENC_ENCODER = "h264_nvenc"
H264_NVENC_TIMING_SAMPLE_INTERVAL = 120
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
)


def prefer_h264_nvenc_encoder() -> None:
    """Prefer NVIDIA H.264 hardware encoding when aiortc sends H.264 video."""
    import aiortc.codecs.h264 as h264

    encoder_cls = h264.H264Encoder
    if getattr(encoder_cls, "_roboflow_h264_nvenc_patched", False):
        return

    original_encode_frame = encoder_cls._encode_frame

    def _encode_frame(self: Any, frame: Any, force_keyframe: bool) -> Iterator[bytes]:
        if self.codec is not None and not getattr(
            self, "_roboflow_h264_nvenc_active", False
        ):
            yield from original_encode_frame(self, frame, force_keyframe)
            return

        if self.codec is not None:
            if _frame_size_changed(self, frame):
                logger.info(
                    "[WEBRTC_NVENC] resolution_changed recreating_encoder "
                    "previous=%sx%s next=%sx%s target_bps=%s",
                    self.codec.width,
                    self.codec.height,
                    frame.width,
                    frame.height,
                    self.target_bitrate,
                )
                _reset_encoder(self)
            elif _bitrate_change_ratio(self) > 0.1:
                _update_nvenc_bitrate(self)

        _set_picture_type(h264, frame, force_keyframe)

        if self.codec is None:
            codec, options, open_ms = _create_nvenc_context(
                h264_module=h264,
                frame=frame,
                target_bitrate=self.target_bitrate,
            )
            if codec is None:
                yield from original_encode_frame(self, frame, force_keyframe)
                return

            self.codec = codec
            self._roboflow_h264_nvenc_active = True
            logger.info(
                "[WEBRTC_NVENC] open_context encoder=%s open_ms=%.2f "
                "resolution=%sx%s target_bps=%s options=%s",
                H264_NVENC_ENCODER,
                open_ms,
                frame.width,
                frame.height,
                self.target_bitrate,
                options,
            )

        should_log_timing = _should_log_timing(self)
        encode_started = time.perf_counter() if should_log_timing else None
        try:
            data_to_send = _encode_to_bytes(self.codec, frame)
        except Exception as error:
            logger.info(
                "[WEBRTC_NVENC] encoder=%s failed; falling_back=libx264 "
                "resolution=%sx%s target_bps=%s error=%r",
                H264_NVENC_ENCODER,
                frame.width,
                frame.height,
                self.target_bitrate,
                error,
            )
            _reset_encoder(self)
            yield from original_encode_frame(self, frame, force_keyframe)
            return

        if should_log_timing and encode_started is not None:
            logger.info(
                "[WEBRTC_NVENC] encode_sample frame=%s encode_ms=%.2f "
                "payload_bytes=%s resolution=%sx%s target_bps=%s codec_bps=%s",
                self._roboflow_h264_nvenc_frame_count,
                (time.perf_counter() - encode_started) * 1000,
                len(data_to_send),
                frame.width,
                frame.height,
                self.target_bitrate,
                getattr(self.codec, "bit_rate", None),
            )

        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    encoder_cls._encode_frame = _encode_frame
    encoder_cls._roboflow_h264_nvenc_patched = True
    logger.info(
        "[WEBRTC_NVENC] patched aiortc H264Encoder to prefer encoder=%s",
        H264_NVENC_ENCODER,
    )


def _create_nvenc_context(
    h264_module: Any,
    frame: Any,
    target_bitrate: int,
) -> Tuple[Optional[Any], Dict[str, str], float]:
    last_error: Optional[Exception] = None

    for options in H264_NVENC_OPTION_SETS:
        try:
            started_at = time.perf_counter()
            codec = h264_module.av.CodecContext.create(H264_NVENC_ENCODER, "w")
            codec.width = frame.width
            codec.height = frame.height
            codec.bit_rate = target_bitrate
            codec.pix_fmt = "yuv420p"
            codec.framerate = fractions.Fraction(h264_module.MAX_FRAME_RATE, 1)
            codec.time_base = fractions.Fraction(1, h264_module.MAX_FRAME_RATE)
            codec.options = options
            _try_set_baseline_profile(codec)
            codec.open()
            return codec, options, (time.perf_counter() - started_at) * 1000
        except Exception as error:
            last_error = error

    logger.info(
        "[WEBRTC_NVENC] encoder=%s unavailable; falling_back=libx264 "
        "resolution=%sx%s target_bps=%s error=%r",
        H264_NVENC_ENCODER,
        frame.width,
        frame.height,
        target_bitrate,
        last_error,
    )
    return None, {}, 0.0


def log_answer_video_codecs(sdp: str) -> None:
    video_payloads, rtpmap = _parse_video_codecs(sdp)
    ordered_codecs = [
        f"{payload}:{rtpmap.get(payload, 'unknown')}" for payload in video_payloads
    ]
    logger.info(
        "[WEBRTC_NVENC] answer_video_codecs preferred=%s codecs=%s",
        ordered_codecs[0] if ordered_codecs else None,
        ",".join(ordered_codecs[:16]),
    )


def _try_set_baseline_profile(codec: Any) -> None:
    try:
        codec.profile = "Baseline"
    except Exception:
        pass


def _set_picture_type(h264_module: Any, frame: Any, force_keyframe: bool) -> None:
    if force_keyframe:
        frame.pict_type = h264_module.av.video.frame.PictureType.I
    else:
        frame.pict_type = h264_module.av.video.frame.PictureType.NONE


def _frame_size_changed(encoder: Any, frame: Any) -> bool:
    return frame.width != encoder.codec.width or frame.height != encoder.codec.height


def _bitrate_change_ratio(encoder: Any) -> float:
    codec_bitrate = getattr(encoder.codec, "bit_rate", None)
    if not codec_bitrate:
        return 0.0
    return abs(encoder.target_bitrate - codec_bitrate) / codec_bitrate


def _update_nvenc_bitrate(encoder: Any) -> None:
    previous_bitrate = getattr(encoder.codec, "bit_rate", None)
    started_at = time.perf_counter()
    try:
        encoder.codec.bit_rate = encoder.target_bitrate
    except Exception as error:
        logger.info(
            "[WEBRTC_NVENC] bitrate_update_failed previous_bps=%s "
            "target_bps=%s error=%r",
            previous_bitrate,
            encoder.target_bitrate,
            error,
        )
        _reset_encoder(encoder)
        return

    logger.info(
        "[WEBRTC_NVENC] bitrate_updated update_ms=%.2f previous_bps=%s "
        "target_bps=%s codec_bps=%s",
        (time.perf_counter() - started_at) * 1000,
        previous_bitrate,
        encoder.target_bitrate,
        getattr(encoder.codec, "bit_rate", None),
    )


def _should_log_timing(encoder: Any) -> bool:
    frame_count = getattr(encoder, "_roboflow_h264_nvenc_frame_count", 0) + 1
    encoder._roboflow_h264_nvenc_frame_count = frame_count
    return frame_count == 1 or frame_count % H264_NVENC_TIMING_SAMPLE_INTERVAL == 0


def _encode_to_bytes(codec: Any, frame: Any) -> bytes:
    data_to_send = b""
    for package in codec.encode(frame):
        data_to_send += bytes(package)
    return data_to_send


def _reset_encoder(encoder: Any) -> None:
    encoder.buffer_data = b""
    encoder.buffer_pts = None
    encoder.codec = None
    encoder._roboflow_h264_nvenc_active = False


def _parse_video_codecs(sdp: str) -> Tuple[List[str], Dict[str, str]]:
    video_payloads: List[str] = []
    rtpmap: Dict[str, str] = {}
    in_video_section = False

    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if line.startswith("m="):
            if in_video_section:
                break
            if line.startswith("m=video "):
                parts = line.split()
                video_payloads = parts[3:]
                in_video_section = True
            continue

        if not in_video_section or not line.startswith("a=rtpmap:"):
            continue

        payload_and_codec = line[len("a=rtpmap:") :]
        payload, separator, codec = payload_and_codec.partition(" ")
        if separator:
            rtpmap[payload] = codec.split("/", 1)[0]

    return video_payloads, rtpmap
