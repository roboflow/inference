print("importing inference")

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
)
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.interfaces.stream_manager.manager_app.bootstrap import (
    run_stream_manager,
)
from inference.core.interfaces.streams_configuration import (
    LEGACY_PIPELINE_HOST_DESCRIPTOR,
    server_streams_configuration,
)
from inference.core.managers.active_learning import (
    ActiveLearningManager,
    BackgroundTaskActiveLearningManager,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES

print("infernece imports done")


if ENABLE_STREAM_API:
    stream_manager_process = Process(
        # The target's module is import-light: a spawned manager installs this
        # configuration and host before importing the manager runtime.
        target=partial(
            run_stream_manager,
            configuration=server_streams_configuration(),
            host_descriptor=LEGACY_PIPELINE_HOST_DESCRIPTOR,
            expected_warmed_up_pipelines=STREAM_API_PRELOADED_PROCESSES,
        ),
    )
    stream_manager_process.start()
    print("Stream Manager started")


print("initializing model manager")
model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)


if ACTIVE_LEARNING_ENABLED:
    print("Initializing Active Learning")
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

print("initializing model manager")
model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
model_manager.init_pingback()

print("initializing http interface")
interface = HttpInterface(model_manager)
app = interface.app
