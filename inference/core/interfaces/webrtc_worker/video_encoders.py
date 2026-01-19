from __future__ import annotations

import logging
import multiprocessing
from typing import Dict, Optional

import av

logger = logging.getLogger(__name__)


def _fast_rampup_options(target_bitrate_bps: int) -> Dict[str, str]:

    kbps = max(1, target_bitrate_bps // 1000)

    return {
        "deadline": "realtime",  # Realtime/latency
        "cpu-used": "8",  # IMPORTANT: positive = faster encode (lower quality), negative = slower/higher quality
        "lag-in-frames": "0",
        "error-resilient": "1",
        "frame-parallel": "1",

        # Rate control: stable CBR-ish behavior
        "rc_end_usage": "cbr",
        "rc_target_bitrate": str(kbps),

        # Buffers (these are in milliseconds in libvpx)
        # Keep them fairly small for quick adaptation and less startup mush.
        "rc_buf_initial_sz": "200",
        "rc_buf_optimal_sz": "300",
        "rc_buf_sz": "400",

        # Don’t be too stingy at startup
        "undershoot-pct": "100",
        "overshoot-pct": "15",

        # lower = better quality, higher = worse quality
        "rc_min_quantizer": "10",
        "rc_max_quantizer": "56",

        "noise-sensitivity": "0",
        "static-thresh": "0",
        "token-partitions": "1",

        # Optional: allow frame dropping under CPU pressure to keep latency under control
        # This option name may vary by build. If it errors, delete it.
        "drop-frame": "1",
    }


def register_custom_vp8_encoder(
    *,
    initial_gop_size: int = 30,
    steady_gop_size: int = 120,
    keyframe_first_n_frames: int = 2,
    cpu_used: int = 8,
) -> bool:
    """
    Returns true if all ok, false if it failed to patch
    """


    try:
        from aiortc.codecs import vpx as vpx_module
    except ImportError:
        logger.debug("VPX module not available")
        return False

    try:
        OriginalVp8Encoder = vpx_module.Vp8Encoder
        original_encode = OriginalVp8Encoder.encode

        def _create_fast_codec(encoder, frame: av.VideoFrame) -> None:
            codec = av.CodecContext.create("libvpx", "w")
            codec.width = frame.width
            codec.height = frame.height
            codec.pix_fmt = "yuv420p"
            codec.bit_rate = int(encoder.target_bitrate)

            # Faster ramp: smaller GOP initially.
            codec.gop_size = int(initial_gop_size)

            # Keep reasonable quantizer bounds at the codec level as a backstop.
            # (libvpx rc_min/max_quantizer still matter more)
            codec.qmin = 10
            codec.qmax = 56

            opts = _fast_rampup_options(int(encoder.target_bitrate))
            opts["cpu-used"] = str(int(cpu_used))
            codec.options = opts

            codec.thread_count = vpx_module.number_of_threads(
                frame.width * frame.height, multiprocessing.cpu_count()
            )

            encoder.codec = codec
            # Track frames to force early keyframes / transition to steady GOP
            encoder._fast_rampup_frames_encoded = 0  # type: ignore[attr-defined]

        def _should_recreate_codec(encoder, frame: av.VideoFrame) -> bool:
            if encoder.codec is None:
                return True

            # Resolution mismatch
            if frame.width != encoder.codec.width or frame.height != encoder.codec.height:
                return True

            # Pixel format mismatch (we force yuv420p)
            if encoder.codec.pix_fmt != "yuv420p":
                return True

            # Bitrate mismatch beyond ~10%
            current = float(encoder.codec.bit_rate or 0)
            target = float(getattr(encoder, "target_bitrate", 0) or 0)
            if current > 0 and target > 0:
                if abs(target - current) / current > 0.10:
                    return True

            return False

        def fast_encode(self, frame: av.VideoFrame, force_keyframe: bool = False):
            # Ensure yuv420p only when needed
            if frame.format is None or frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")

            if _should_recreate_codec(self, frame):
                self.codec = None

            if self.codec is None:
                _create_fast_codec(self, frame)

            # Force keyframes on the first N frames to stabilize quickly
            frames_encoded = getattr(self, "_fast_rampup_frames_encoded", 0)
            if frames_encoded < int(keyframe_first_n_frames):
                force_keyframe = True

            # After a short warmup, switch GOP to steadier value
            # (Do it once, and only if the codec exists.)
            if frames_encoded == int(keyframe_first_n_frames) and self.codec is not None:
                self.codec.gop_size = int(steady_gop_size)

            setattr(self, "_fast_rampup_frames_encoded", frames_encoded + 1)
            return original_encode(self, frame, force_keyframe)

        OriginalVp8Encoder.encode = fast_encode
        logger.info(
            "Registered custom VP8 encoder fast-ramp patch "
            "(initial_gop=%s steady_gop=%s keyframes_first=%s cpu-used=%s)",
            initial_gop_size,
            steady_gop_size,
            keyframe_first_n_frames,
            cpu_used,
        )
        return True

    except Exception as e:
        logger.warning("Failed to register custom VP8 encoder: %s", e, exc_info=True)
        return False



async def set_sender_bitrates(sender, *, min_bps: int, max_bps: int) -> None:
    """
    Set RTP sender encoding bitrate bounds (GoogCC floor/ceiling).
    Call this after you create the sender (addTrack/createTransceiver) and before/after negotiation.

    Example:
        video_sender = pc.getSenders()[...]
        await set_sender_bitrates(video_sender, min_bps=1_500_000, max_bps=6_000_000)
    """
    params = sender.getParameters()
    if not params.encodings:
        params.encodings = [{}]
    params.encodings[0]["minBitrate"] = int(min_bps)
    params.encodings[0]["maxBitrate"] = int(max_bps)
    await sender.setParameters(params)
