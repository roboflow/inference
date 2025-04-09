import json

from mangum import Mangum

from inference.core.cache import cache
from inference.core.env import ACTIVE_LEARNING_ENABLED, MAX_ACTIVE_MODELS
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.active_learning import ActiveLearningManager
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)

if ACTIVE_LEARNING_ENABLED:
    model_manager = ActiveLearningManager(model_registry=model_registry, cache=cache)
else:
    model_manager = ModelManager(model_registry)

model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
interface = HttpInterface(model_manager)
handler = Mangum(interface.app, lifespan="off")
