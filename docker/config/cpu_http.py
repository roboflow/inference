from functools import partial
from multiprocessing import Process

from inference.core.cache import cache
from inference.core.env import (
    ACTIVE_LEARNING_ENABLED,
    ENABLE_STREAM_API,
    GCP_SERVERLESS,
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
    stream_manager_process = Process(
        target=partial(start, expected_warmed_up_pipelines=STREAM_API_PRELOADED_PROCESSES),
    )
    stream_manager_process.start()

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)

if ACTIVE_LEARNING_ENABLED:
    if LAMBDA or GCP_SERVERLESS:
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
    import importlib.metadata as _md

    _adapter_factory = None
    for _ep in _md.entry_points(group="inference.legacy_adapter"):
        if _ep.name == "mmp":
            _adapter_factory = _ep.load()
            break
    if _adapter_factory is None:
        raise ImportError(
            "LEGACY_MMP_ADAPTER_ENABLED requires the Roboflow enterprise runtime (rf-legacy-bridge)."
        )
    model_manager = _adapter_factory(legacy_stack=model_manager)
model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app

if LEGACY_MMP_ADAPTER_ENABLED:

    @app.on_event("startup")
    async def start_mmp_adapter():
        await model_manager.start()

    @app.on_event("shutdown")
    async def stop_mmp_adapter():
        await model_manager.shutdown()
