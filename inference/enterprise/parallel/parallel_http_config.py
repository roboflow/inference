from inference.enterprise.parallel.parallel_http_api import ParallelHttpInterface
from inference.enterprise.parallel.dispatch_manager import DispatchModelManager
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES


model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = DispatchModelManager(model_registry)
model_manager.init_pingback()
interface = ParallelHttpInterface(model_manager)

app = interface.app