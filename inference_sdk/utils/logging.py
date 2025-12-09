"""Centralized logging configuration for the Inference SDK."""

import logging
import sys

# SDK-wide logger name
SDK_LOGGER_NAME = "inference_sdk"

# Track if we've already configured
_configured = False


def get_logger(module_name: str) -> logging.Logger:
    """Get a logger for the specified module.

    Automatically configures basic logging on first use if no handlers exist.

    Args:
        module_name: Name of the module requesting the logger.

    Returns:
        logging.Logger: Configured logger for the module.
    """
    global _configured

    sdk_logger = logging.getLogger(SDK_LOGGER_NAME)

    # Configure basic logging on first use if needed
    if not _configured and not sdk_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
        sdk_logger.addHandler(handler)
        sdk_logger.setLevel(logging.INFO)
        sdk_logger.propagate = False
        _configured = True

    return logging.getLogger(f"{SDK_LOGGER_NAME}.{module_name}")
