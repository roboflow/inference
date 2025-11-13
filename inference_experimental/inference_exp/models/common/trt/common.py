from collections import deque
from typing import Deque, List, Tuple

from inference_exp.errors import MissingDependencyError
from inference_exp.logger import LOGGER

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not TRT tools required to run models with TRT backend - this error means that some additional "
        f"dependencies are not installed in the environment. If you run the `inference-exp` library directly in your "
        f"Python program, make sure the following extras of the package are installed: `trt10` - installation can only "
        f"succeed for Linux and Windows machines with Cuda 12 installed. Jetson devices, should have TRT 10.x "
        f"installed for all builds with Jetpack 6. "
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


class InferenceTRTLogger(trt.ILogger):

    def __init__(self, with_memory: bool = False, memory_size: int = 200):
        super().__init__()
        self._memory: Deque[Tuple[trt.ILogger.Severity, str]] = deque(
            maxlen=memory_size
        )
        self._with_memory = with_memory

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        if self._with_memory:
            self._memory.append((severity, msg))
        severity_str = str(severity)
        if severity_str == str(trt.Logger.VERBOSE):
            log_function = LOGGER.debug
        elif severity_str is str(trt.Logger.INFO):
            log_function = LOGGER.info
        elif severity_str is str(trt.Logger.WARNING):
            log_function = LOGGER.warning
        else:
            log_function = LOGGER.error
        log_function(msg)

    def get_memory(self) -> List[Tuple[trt.ILogger.Severity, str]]:
        return list(self._memory)
