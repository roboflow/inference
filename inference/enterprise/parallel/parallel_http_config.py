from redis import ConnectionPool, Redis
from redis.asyncio import Redis as AsyncRedis

from inference.core.env import REDIS_HOST, REDIS_PORT
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.parallel.dispatch_manager import (
    DispatchModelManager,
    ResultsChecker,
)
from inference.enterprise.parallel.parallel_http_api import ParallelHttpInterface
from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
if REDIS_HOST is None:
    raise RuntimeError("Redis must be configured to use async inference")
pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
model_manager = None
interface = ParallelHttpInterface(model_manager)

app = interface.app
