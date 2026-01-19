import logging

logger = logging.getLogger(__name__)


def register_custom_vp8_encoder() -> bool:
    """Register optimized VP8 encoder subclass in aiortc's codec module.

    Creates a subclass of Vp8Encoder with faster quality ramp-up settings
    (smaller buffers, higher undershoot) and replaces the original.

    Returns:
        True if encoder was registered, False otherwise
    """
    try:
        from aiortc.codecs import vpx as vpx_module
    except ImportError:
        logger.debug("VPX module not available")
        return False

    try:
        OriginalVp8Encoder = vpx_module.Vp8Encoder

        class CustomVp8Encoder(OriginalVp8Encoder):
            """VP8 encoder with faster quality ramp-up settings."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if hasattr(self, "_encoder") and self._encoder:
                    cfg = self._encoder.config
                    cfg.rc_target_bitrate = 2000  # kbps
                    cfg.rc_buf_initial_sz = 500  # ms (default 4000)
                    cfg.rc_buf_optimal_sz = 600  # ms
                    cfg.rc_buf_sz = 1000  # ms
                    cfg.rc_undershoot_pct = 95  # default 25
                    cfg.rc_min_quantizer = 2  # 0-63, lower = better quality
                    self._encoder.configure(cfg)

        vpx_module.Vp8Encoder = CustomVp8Encoder
        logger.info("Registered custom VP8 encoder")
        return True
    except AttributeError:
        logger.debug("VP8 encoder not available in aiortc")
        return False
    except Exception as e:
        logger.warning("Failed to register custom VP8 encoder: %s", e)
        return False
