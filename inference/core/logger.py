import logging
import warnings

from rich.logging import RichHandler

from inference.core.env import LOG_LEVEL

logger = logging.getLogger("inference")
logger.setLevel(LOG_LEVEL)
logger.addHandler(RichHandler())
logger.propagate = False

if LOG_LEVEL == "ERROR" or LOG_LEVEL == "FATAL":
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime.*")
