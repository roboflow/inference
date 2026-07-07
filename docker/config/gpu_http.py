import multiprocessing
from functools import partial

from inference.core.cache import cache
from inference.core.env import (
    ACTIVE_LEARNING_ENABLED,
    ENABLE_STREAM_API,
    LAMBDA,
    LEGACY_MMP_ADAPTER_ENABLED,
    MAX_ACTIVE_MODELS,
    STREAM_API_PRELOADED_PROCESSES,
)
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.interfaces.stream_manager.manager_app.app import start
from inference.core.managers.active_learning import (
    ActiveLearningManager,
    BackgroundTaskActiveLearningManager,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES

if ENABLE_STREAM_API:
    multiprocessing_context = multiprocessing.get_context(method="spawn")
    stream_manager_process = multiprocessing_context.Process(
        target=partial(start, expected_warmed_up_pipelines=STREAM_API_PRELOADED_PROCESSES),
    )
    stream_manager_process.start()

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)

if ACTIVE_LEARNING_ENABLED:
    if LAMBDA:
        model_manager = ActiveLearningManager(
            model_registry=model_registry, cache=cache
        )
    else:
        model_manager = BackgroundTaskActiveLearningManager(
            model_registry=model_registry, cache=cache
        )
else:
    model_manager = ModelManager(model_registry=model_registry)

model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
if LEGACY_MMP_ADAPTER_ENABLED:
    from inference.core.managers.mmp_adapter import ModelManagerAdapter

    model_manager = ModelManagerAdapter(legacy_stack=model_manager)
model_manager.init_pingback()
interface = HttpInterface(
    model_manager,
)
app = interface.app

if LEGACY_MMP_ADAPTER_ENABLED:

    @app.on_event("startup")
    async def start_mmp_adapter():
        await model_manager.start()

    @app.on_event("shutdown")
    async def stop_mmp_adapter():
        await model_manager.shutdown()
