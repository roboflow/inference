from threading import Thread

from redis import Redis

from inference.core.env import REDIS_HOST, REDIS_PORT
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.parallel.dispatch_manager import (
    DispatchModelManager,
    ResultsChecker,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES


class ParallelHttpInterface(HttpInterface):
    def __init__(self, model_manager: DispatchModelManager, root_path: str = None):
        super().__init__(model_manager, root_path)

        @self.app.on_event("startup")
        async def app_startup():
            model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
            checker = ResultsChecker(Redis(host=REDIS_HOST, port=REDIS_PORT))
            self.model_manager = DispatchModelManager(model_registry, checker)
            self.model_manager.init_pingback()
            checker_loop_thread = Thread(
                target=self.model_manager.checker.loop, args=(), daemon=True
            )
            checker_loop_thread.start()
            # keep checker loop reference so it doesn't get gc'd
            self.checker_loop = checker_loop_thread
