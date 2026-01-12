import logging

from inference_models.configuration import (
    DISABLE_VERBOSE_LOGGER,
    LOG_LEVEL,
    VERBOSE_LOG_LEVEL,
)


def configure_log_level(
    logger: logging.Logger, log_level: str, fallback_level: int
) -> None:
    log_level = getattr(logging, log_level, fallback_level)
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    logger.propagate = False


LOGGER = logging.getLogger("inference-models")
configure_log_level(logger=LOGGER, log_level=LOG_LEVEL, fallback_level=logging.WARNING)
VERBOSE_LOGGER = logging.getLogger("inference-models-verbose")
configure_log_level(
    logger=VERBOSE_LOGGER, log_level=VERBOSE_LOG_LEVEL, fallback_level=logging.INFO
)


def verbose_info(
    message: str,
    verbose_requested: bool = True,
) -> None:
    if DISABLE_VERBOSE_LOGGER:
        return None
    if not verbose_requested:
        return None
    VERBOSE_LOGGER.info(message)


def verbose_debug(
    message: str,
    verbose_requested: bool = True,
) -> None:
    if DISABLE_VERBOSE_LOGGER:
        return None
    if not verbose_requested:
        return None
    VERBOSE_LOGGER.debug(message)
