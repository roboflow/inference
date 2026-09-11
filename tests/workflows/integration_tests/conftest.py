import os

import pytest

from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES


@pytest.fixture
def roboflow_api_key() -> str:
    return os.environ["ROBOFLOW_API_KEY"]


@pytest.fixture
def deep_lab_v3_api_key() -> str:
    return os.environ["DEEP_LAB_V3_API_KEY"]


@pytest.fixture(scope="function")
def raw_model_manager() -> ModelManager:
    # The decorated server manager - what this fixture returned before Phase 11.
    # For INSPECTION only (`raw_model_manager.models()`); never bound to a workflow.
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    return WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)


@pytest.fixture(scope="function")
def model_manager(raw_model_manager) -> ModelManagerModelsProvider:
    # What the composition roots inject: the adapter over that SAME manager, so a
    # test can run a workflow with `model_manager` and inspect
    # `raw_model_manager.models()` afterwards. Class-level patches on
    # `ModelManager.infer_from_request_sync` / `add_model` still intercept -
    # the adapter forwards to the wrapped instance.
    return ModelManagerModelsProvider(raw_model_manager)
