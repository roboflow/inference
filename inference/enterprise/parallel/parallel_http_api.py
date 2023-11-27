import asyncio

from inference.core.interfaces.http.http_api import HttpInterface
from inference.enterprise.parallel.dispatch_manager import DispatchModelManager


class ParallelHttpInterface(HttpInterface):
    def __init__(self, model_manager: DispatchModelManager, root_path: str = None):
        super().__init__(model_manager, root_path)

        @self.app.on_event("startup")
        async def app_startup():
            task = asyncio.create_task(self.model_manager.checker.loop())
            # keep checker loop reference so it doesn't get gc'd
            self.checker_loop = task
