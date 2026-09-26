from functools import partial
from multiprocessing import Process

from inference.core.cache import cache
from inference.core.env import (
    ACTIVE_LEARNING_ENABLED,
    ENABLE_STREAM_API,
    GCP_SERVERLESS,
    LAMBDA,
    MAX_ACTIVE_MODELS,
    STREAM_API_PRELOADED_PROCESSES,
    API_KEY as ROBOFLOW_API_KEY,
    MODEL_PREWARM_LIST,
    MODEL_PREWARM_MAX_PARALLEL,
    MODEL_PREWARM_RETRY_COUNT,
    MODEL_PREWARM_RETRY_DELAY,
    MODEL_PREWARM_GATE_READINESS,
    EVICTION_PROTECTION_WINDOW_SECONDS,
    EVICTION_PROTECTION_HIGH_FREQUENCY_THRESHOLD,
)
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.interfaces.stream_manager.manager_app.app import start
from inference.core.managers.active_learning import (
    ActiveLearningManager,
    BackgroundTaskActiveLearningManager,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.eviction_protected_cache import WithEvictionProtectedCache
from inference.core.managers.prewarming import ModelPrewarmConfig, ModelPrewarmingManager
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

model_manager = WithEvictionProtectedCache(
    model_manager,
    max_size=MAX_ACTIVE_MODELS,
    protection_window_seconds=EVICTION_PROTECTION_WINDOW_SECONDS,
    high_frequency_threshold=EVICTION_PROTECTION_HIGH_FREQUENCY_THRESHOLD,
)

# Pre-warm models if configured
prewarm_manager = None
if MODEL_PREWARM_LIST and ROBOFLOW_API_KEY:
    prewarm_configs = [
        ModelPrewarmConfig(
            model_id=model_id,
            api_key=ROBOFLOW_API_KEY,
            pin=True,
            required_for_readiness=MODEL_PREWARM_GATE_READINESS,
        )
        for model_id in MODEL_PREWARM_LIST
    ]
    prewarm_manager = ModelPrewarmingManager(
        model_manager=model_manager,
        models_to_prewarm=prewarm_configs,
        max_parallel_loads=MODEL_PREWARM_MAX_PARALLEL,
        retry_count=MODEL_PREWARM_RETRY_COUNT,
        retry_delay_seconds=MODEL_PREWARM_RETRY_DELAY,
    )
    # Execute pre-warming at startup
    prewarm_success = prewarm_manager.warmup(timeout=300.0)
    if not prewarm_success and MODEL_PREWARM_GATE_READINESS:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Pre-warming failed! Server readiness will be blocked.")

model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app

# Expose prewarm_manager for readiness checks
app.state.prewarm_manager = prewarm_manager
