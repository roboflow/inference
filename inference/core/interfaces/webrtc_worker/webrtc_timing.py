import math
import time
from typing import Any, Dict, Optional, Tuple

from inference.core import logger

WEBRTC_TIMING_LOG_EVERY_FRAMES = 30


def apply_aiortc_timing_logs() -> None:
    """Patch aiortc codec classes with sampled encode/decode timing logs."""
    import aiortc.codecs.h264 as h264
    import aiortc.codecs.vpx as vpx

    _patch_decoder(h264.H264Decoder, "h264")
    _patch_decoder(vpx.Vp8Decoder, "vp8")
    _patch_encoder(h264.H264Encoder, "h264", default_encoder_name="libx264")
    _patch_encoder(vpx.Vp8Encoder, "vp8", default_encoder_name="libvpx")


def record_frame_timing(
    owner: Any,
    frame_id: int,
    input_recv_ms: float,
    auto_detect_ms: float,
    process_ms: float,
    data_output_ms: float,
    total_pre_encode_ms: float,
    input_resolution: Tuple[Optional[int], Optional[int]],
    output_resolution: Tuple[Optional[int], Optional[int]],
    fps: Optional[float],
    errors_count: int,
) -> None:
    stats = _stats_container(owner, "_roboflow_frame_timing_stats")
    sample_count, should_log = _record_metrics(
        stats,
        {
            "input_recv_ms": input_recv_ms,
            "auto_detect_ms": auto_detect_ms,
            "process_ms": process_ms,
            "data_output_ms": data_output_ms,
            "total_pre_encode_ms": total_pre_encode_ms,
        },
    )
    if not should_log:
        return

    logger.info(
        "[WEBRTC_E2E_TIMING] stage=frame frame=%d samples=%d "
        "input_recv_wait_decode_ms=%.2f avg_input_recv_wait_decode_ms=%.2f "
        "auto_detect_ms=%.2f avg_auto_detect_ms=%.2f "
        "process_inference_render_ms=%.2f avg_process_inference_render_ms=%.2f "
        "data_output_ms=%.2f avg_data_output_ms=%.2f "
        "total_pre_encode_ms=%.2f avg_total_pre_encode_ms=%.2f "
        "input_resolution=%sx%s output_resolution=%sx%s fps=%.2f errors=%d",
        frame_id,
        sample_count,
        input_recv_ms,
        stats["input_recv_ms"]["avg"],
        auto_detect_ms,
        stats["auto_detect_ms"]["avg"],
        process_ms,
        stats["process_ms"]["avg"],
        data_output_ms,
        stats["data_output_ms"]["avg"],
        total_pre_encode_ms,
        stats["total_pre_encode_ms"]["avg"],
        input_resolution[0],
        input_resolution[1],
        output_resolution[0],
        output_resolution[1],
        fps if fps is not None else 0.0,
        errors_count,
    )


def _patch_encoder(
    encoder_cls: Any,
    codec_name: str,
    default_encoder_name: str,
) -> None:
    patch_flag = f"_roboflow_{codec_name}_encode_timing_patched"
    if getattr(encoder_cls, patch_flag, False):
        return

    original_encode = encoder_cls.encode

    def encode(self: Any, frame: Any, force_keyframe: bool = False) -> Any:
        codec_before = getattr(self, "codec", None)
        codec_before_id = id(codec_before) if codec_before is not None else None
        started_at = time.perf_counter()
        payloads, timestamp = original_encode(self, frame, force_keyframe)
        encode_ms = _elapsed_ms(started_at)

        codec_after = getattr(self, "codec", None)
        codec_after_id = id(codec_after) if codec_after is not None else None
        codec_recreated = codec_before_id != codec_after_id
        payload_bytes = sum(len(payload) for payload in payloads)

        _record_encode_timing(
            encoder=self,
            codec_name=codec_name,
            encoder_name=_encoder_name(self, default_encoder_name),
            encode_ms=encode_ms,
            frame=frame,
            target_bitrate=getattr(self, "target_bitrate", None),
            payload_count=len(payloads),
            payload_bytes=payload_bytes,
            force_keyframe=force_keyframe,
            codec_recreated=codec_recreated,
        )
        return payloads, timestamp

    encoder_cls.encode = encode
    setattr(encoder_cls, patch_flag, True)


def _patch_decoder(decoder_cls: Any, codec_name: str) -> None:
    patch_flag = f"_roboflow_{codec_name}_decode_timing_patched"
    if getattr(decoder_cls, patch_flag, False):
        return

    original_decode = decoder_cls.decode

    def decode(self: Any, encoded_frame: Any) -> Any:
        started_at = time.perf_counter()
        frames = original_decode(self, encoded_frame)
        decode_ms = _elapsed_ms(started_at)
        encoded_bytes = len(getattr(encoded_frame, "data", b""))
        _record_decode_timing(
            decoder=self,
            codec_name=codec_name,
            decode_ms=decode_ms,
            frames_out=len(frames),
            encoded_bytes=encoded_bytes,
        )
        return frames

    decoder_cls.decode = decode
    setattr(decoder_cls, patch_flag, True)


def _record_encode_timing(
    encoder: Any,
    codec_name: str,
    encoder_name: str,
    encode_ms: float,
    frame: Any,
    target_bitrate: Optional[int],
    payload_count: int,
    payload_bytes: int,
    force_keyframe: bool,
    codec_recreated: bool,
) -> None:
    stats = _stats_container(encoder, "_roboflow_encode_timing_stats")
    sample_count, should_log = _record_metrics(stats, {"encode_ms": encode_ms})
    if not should_log:
        return

    logger.info(
        "[WEBRTC_E2E_TIMING] stage=encode codec=%s encoder=%s samples=%d "
        "encode_ms=%.2f avg_encode_ms=%.2f min_encode_ms=%.2f max_encode_ms=%.2f "
        "resolution=%sx%s target_bps=%s payloads=%d payload_bytes=%d "
        "force_keyframe=%s codec_recreated=%s",
        codec_name,
        encoder_name,
        sample_count,
        encode_ms,
        stats["encode_ms"]["avg"],
        stats["encode_ms"]["min"],
        stats["encode_ms"]["max"],
        getattr(frame, "width", None),
        getattr(frame, "height", None),
        target_bitrate,
        payload_count,
        payload_bytes,
        force_keyframe,
        codec_recreated,
    )


def _record_decode_timing(
    decoder: Any,
    codec_name: str,
    decode_ms: float,
    frames_out: int,
    encoded_bytes: int,
) -> None:
    stats = _stats_container(decoder, "_roboflow_decode_timing_stats")
    sample_count, should_log = _record_metrics(stats, {"decode_ms": decode_ms})
    if not should_log:
        return

    logger.info(
        "[WEBRTC_E2E_TIMING] stage=decode codec=%s samples=%d "
        "decode_ms=%.2f avg_decode_ms=%.2f min_decode_ms=%.2f max_decode_ms=%.2f "
        "frames_out=%d encoded_bytes=%d",
        codec_name,
        sample_count,
        decode_ms,
        stats["decode_ms"]["avg"],
        stats["decode_ms"]["min"],
        stats["decode_ms"]["max"],
        frames_out,
        encoded_bytes,
    )


def _record_metrics(
    stats: Dict[str, Dict[str, float]],
    metrics: Dict[str, float],
) -> Tuple[int, bool]:
    sample_count = 0
    for metric_name, value in metrics.items():
        metric_stats = stats.setdefault(
            metric_name,
            {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": math.inf,
                "max": 0.0,
            },
        )
        metric_stats["count"] += 1
        metric_stats["sum"] += value
        metric_stats["avg"] = metric_stats["sum"] / metric_stats["count"]
        metric_stats["min"] = min(metric_stats["min"], value)
        metric_stats["max"] = max(metric_stats["max"], value)
        sample_count = int(metric_stats["count"])

    return sample_count, _should_log_sample(sample_count)


def _stats_container(owner: Any, attr_name: str) -> Dict[str, Dict[str, float]]:
    stats = getattr(owner, attr_name, None)
    if stats is None:
        stats = {}
        setattr(owner, attr_name, stats)
    return stats


def _should_log_sample(sample_count: int) -> bool:
    return sample_count <= 5 or sample_count % WEBRTC_TIMING_LOG_EVERY_FRAMES == 0


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _encoder_name(encoder: Any, default_encoder_name: str) -> str:
    if getattr(encoder, "_roboflow_h264_nvenc_active", False):
        return "h264_nvenc"

    codec = getattr(encoder, "codec", None)
    codec_name = getattr(codec, "name", None)
    if codec_name:
        return codec_name
    return default_encoder_name
