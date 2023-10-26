import asyncio

from redis import ConnectionPool, Redis

from inference.core.env import REDIS_HOST, REDIS_PORT
from inference.core.interfaces.http.http_api import HttpInterface
from inference.enterprise.parallel.dispatch_manager import DispatchModelManager, ResultsChecker


class ParallelHttpInterface(HttpInterface):
    def __init__(self, model_manager: DispatchModelManager, root_path: str = None):
        super().__init__(model_manager, root_path)

        @self.app.on_event("startup")
        async def app_startup():
            assert REDIS_HOST is not None
            pool = ConnectionPool(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
            )
            self.checker = ResultsChecker()
            r = Redis(connection_pool=pool, decode_responses=True)
            self.checker.add_redis(r)
            self.model_manager.add_checker(self.checker)
            asyncio.create_task(self.checker.loop())
