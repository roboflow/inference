import logging
import os
import sys
import traceback
import warnings
from typing import Any, Dict

import structlog
from rich.logging import RichHandler
from structlog import BoundLogger
from structlog.processors import CallsiteParameter
from structlog.stdlib import _FixedFindCallerLogger
from structlog.typing import EventDict, WrappedLogger

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
        "exception": "ERROR",  # exception logs at ERROR level
        "critical": "CRITICAL",
    }
    event_dict["severity"] = level_to_severity.get(method_name, "DEFAULT")
    return event_dict


class GCPCloudLoggingProcessor:
    def __call__(self, logger, name, event_dict):
        if "event" in event_dict:
            event_dict["message"] = event_dict.pop("event")
        return event_dict


class NoTracebackFormatter(logging.Formatter):
    def format(self, record):
        # Remove exc_info before formatting
        record.exc_info = None
        return super().format(record)


def structlog_exception_formatter(
    logger_instance: WrappedLogger, name: str, event_dict: EventDict
) -> EventDict:
    exc_info = event_dict.pop("exc_info", None)
    if not exc_info:
        return event_dict
    if isinstance(exc_info, BaseException):
        exc_type, exc_value, exc_tb = (
            exc_info.__class__,
            exc_info,
            exc_info.__traceback__,
        )
    elif isinstance(exc_info, tuple):
        exc_type, exc_value, exc_tb = exc_info
    else:
        exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb:
        tb_list = traceback.extract_tb(exc_tb)
    else:
        tb_list = []
    event_dict["exception"] = {
        "type": exc_type.__name__ if exc_type else "N/A",
        "message": str(exc_value) if exc_value else "N/A",
        "stacktrace": [
            {
                "filename": tb.filename,
                "lineno": tb.lineno,
                "function": tb.name,
                "code": tb.line,
            }
            for tb in tb_list
        ],
    }
    return event_dict


if API_LOGGING_ENABLED:

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
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
        structlog.processors.StackInfoRenderer(),
        structlog_exception_formatter,
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
    bounded_logger = logger.bind()
    handler = logging.StreamHandler()
    handler.setFormatter(NoTracebackFormatter("%(message)s"))
    bounded_logger._logger.addHandler(handler)
    bounded_logger._logger.propagate = False
else:
    logger = logging.getLogger("inference")
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(RichHandler())
    logger.propagate = False
