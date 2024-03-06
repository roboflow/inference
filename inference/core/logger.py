import logging
import os
import warnings
from typing import Any, Dict

from rich.logging import RichHandler
from structlog.processors import CallsiteParameter

from inference.core.env import LOG_LEVEL
from inference.core.utils.environment import str2bool

if LOG_LEVEL == "ERROR" or LOG_LEVEL == "FATAL":
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime.*")


def add_correlation(
    logger: logging.Logger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    from asgi_correlation_id import correlation_id

    request_id = correlation_id.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


if str2bool(os.getenv("API_LOGGING_ENABLED", "False")):
    import structlog

    structlog.configure(
        processors=[
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
            structlog.processors.JSONRenderer(),
        ],
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
