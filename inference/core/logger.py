import logging
import os
import warnings
from typing import Any, Dict

from rich.logging import RichHandler
from structlog.processors import CallsiteParameter

from inference.core.env import (
    API_LOGGING_ENABLED,
    CORRELATION_ID_LOG_KEY,
    DEDICATED_DEPLOYMENT_ID,
    GCP_SERVERLESS,
    LOG_LEVEL,
)
from inference.core.utils.environment import str2bool

if LOG_LEVEL == "ERROR" or LOG_LEVEL == "FATAL":
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime.*")


def add_correlation(
    logger: logging.Logger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    from asgi_correlation_id import correlation_id

    request_id = correlation_id.get()
    if request_id:
        event_dict[CORRELATION_ID_LOG_KEY] = request_id
    return event_dict


def add_gcp_severity(
    logger: logging.Logger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    level_to_severity = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
    }
    event_dict["severity"] = level_to_severity.get(method_name, "DEFAULT")
    return event_dict


class GCPCloudLoggingProcessor:
    def __call__(self, logger, name, event_dict):
        if "event" in event_dict:
            event_dict["message"] = event_dict.pop("event")
        return event_dict


if API_LOGGING_ENABLED:
    import structlog

    is_gcp_environment = GCP_SERVERLESS or DEDICATED_DEPLOYMENT_ID is not None

    if is_gcp_environment:
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, LOG_LEVEL),
            force=True,
        )

    processors = [
        add_correlation,
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.CallsiteParameterAdder(
            [
                CallsiteParameter.FILENAME,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ],
        ),
    ]

    if is_gcp_environment:
        processors.insert(1, add_gcp_severity)
        processors.append(GCPCloudLoggingProcessor())

    processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger("inference")
    logger.setLevel(LOG_LEVEL)
else:
    logger = logging.getLogger("inference")
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(RichHandler())
    logger.propagate = False
