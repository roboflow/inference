import json

from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.interfaces.http.parallel_http_api import ParallelHttpInterface
from inference.core.managers.parallel import DispatchModelManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES


model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = DispatchModelManager(model_registry)
model_manager.init_pingback()
interface = ParallelHttpInterface(model_manager)

app = interface.app