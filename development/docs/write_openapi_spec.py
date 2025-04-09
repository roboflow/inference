


from functools import partial
from multiprocessing import Process

from inference.core.cache import cache
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

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)


model_manager = ModelManager(model_registry=model_registry)

model_manager = WithFixedSizeCache(model_manager, max_size=1)
model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app




import json
import os

from fastapi.openapi.utils import get_openapi

DOCS_ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "docs",
    )
)

filename = os.path.join(DOCS_ROOT_DIR, "openapi.json")

with open(filename, 'w') as f:
    json.dump(get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        
    ), f)