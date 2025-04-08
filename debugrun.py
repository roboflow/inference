import sys
import uvicorn
from inference.core.cache import cache
from inference.core.env import MAX_ACTIVE_MODELS
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry


"""
convenient script to run server in debug
(i.e. runs uvicorn directly)

It's a simplified version of docker/config/cpu_http.py

see https://www.loom.com/share/48f71894427a473cac39eca25f6ac759

- uv venv
- source .venv/bin/activate
- uv pip install -e .
- # start debugrun.py in debug mode
"""

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = BackgroundTaskActiveLearningManager(
    model_registry=model_registry, cache=cache
)

model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=9001)
    except Exception as e:
        print("Error starting server:", e)
        sys.exit(1)