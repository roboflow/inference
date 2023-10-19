from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
import os
from prometheus_fastapi_instrumentator import Instrumentator

from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = WithFixedSizeCache(ModelManager(model_registry))
model_manager.model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app

# Setup Prometheus scraping endpoint at /metrics
# More info: https://github.com/trallnag/prometheus-fastapi-instrumentator
if os.environ.get("ENABLE_PROMETHEUS", False):
    instrumentor = Instrumentator()
    instrumentor.instrument(app).expose(app)

    @app.on_event("startup")
    async def _startup():
        instrumentor.expose(app)
