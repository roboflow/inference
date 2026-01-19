import logging
import multiprocessing

import av

logger = logging.getLogger(__name__)

def _fast_rampup_options(bitrate: int) -> dict:
    """VP8 options for fast quality ramp-up."""
    buf = str(bitrate // 2)  # 500ms buffer
    rate = str(bitrate)
    return {
        "bufsize": buf,
        "cpu-used": "-6",
        "deadline": "realtime",
        "lag-in-frames": "0",
        "minrate": rate,
        "maxrate": rate,
        "undershoot-pct": "100",
        "overshoot-pct": "15",
        "noise-sensitivity": "4",
        "static-thresh": "0",
        "partitions": "0",
        "error-resilient": "1",
    }


def register_custom_vp8_encoder() -> bool:
    """Register VP8 encoder with faster quality ramp-up settings."""
    try:
        from aiortc.codecs import vpx as vpx_module
    except ImportError:
        logger.debug("VPX module not available")
        return False

    try:
        OriginalVp8Encoder = vpx_module.Vp8Encoder
        original_encode = OriginalVp8Encoder.encode

        def _create_fast_codec(encoder, frame):
            """Create codec with fast ramp-up settings."""
            encoder.codec = av.CodecContext.create("libvpx", "w")
            encoder.codec.width = frame.width
            encoder.codec.height = frame.height
            encoder.codec.bit_rate = encoder.target_bitrate
            encoder.codec.pix_fmt = "yuv420p"
            encoder.codec.gop_size = 120
            encoder.codec.qmin = 4
            encoder.codec.qmax = 48
            encoder.codec.options = _fast_rampup_options(encoder.target_bitrate)
            encoder.codec.thread_count = vpx_module.number_of_threads(
                frame.width * frame.height, multiprocessing.cpu_count()
            )

        def fast_encode(self, frame, force_keyframe=False):
            # Ensure yuv420p format
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")

            # Invalidate codec on size/bitrate change
            if self.codec and (
                frame.width != self.codec.width
                or frame.height != self.codec.height
                or abs(self.target_bitrate - self.codec.bit_rate)
                / self.codec.bit_rate
                > 0.1
            ):
                self.codec = None

            # Create our custom codec before parent tries to
            if self.codec is None:
                _create_fast_codec(self, frame)

            return original_encode(self, frame, force_keyframe)

        OriginalVp8Encoder.encode = fast_encode
        logger.info("Registered custom VP8 encoder with fast ramp-up settings")
        return True
    except Exception as e:
        logger.warning("Failed to register custom VP8 encoder: %s", e)
        return False
