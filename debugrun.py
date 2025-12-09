import sys
import uvicorn
from inference.core.cache import cache
from inference.core.env import MAX_ACTIVE_MODELS, ENABLE_STREAM_API
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.interfaces.stream_manager.manager_app.app import start

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

from functools import partial
from multiprocessing import Process


if __name__ == "__main__":
    print("Starting Stream Manager...")
    stream_manager_process = Process(
        target=partial(start, expected_warmed_up_pipelines=0),
    )
    stream_manager_process.start()

    print("Starting server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=9001)
    except Exception as e:
        print("Error starting server:", e)
        sys.exit(1)