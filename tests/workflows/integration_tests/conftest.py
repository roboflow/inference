import os

import pytest

from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES


@pytest.fixture
def roboflow_api_key() -> str:
    return os.environ["ROBOFLOW_API_KEY"]


@pytest.fixture(scope="function")
def model_manager() -> ModelManager:
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    return WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
