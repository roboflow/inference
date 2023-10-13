import json
from mangum import Mangum

from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES


model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = WithFixedSizeCache(ModelManager(model_registry), max_size=MAX_ACTIVE_MODELS)
interface = HttpInterface(model_manager)
handler = Mangum(interface.app, lifespan="off")
