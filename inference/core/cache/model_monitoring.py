from inference.core.cache import cache
from inference.core.cache.base import BaseCache
from inference.core.cache.memory import MemoryCache
from inference.core.env import MODEL_MONITORING_CACHE_BACKEND
from inference.core.logger import logger

if MODEL_MONITORING_CACHE_BACKEND == "memory":
    model_monitoring_cache: BaseCache = MemoryCache()
    logger.info("Model monitoring cache initialised with MemoryCache")
else:
    model_monitoring_cache = cache
    logger.info(
        "Model monitoring cache initialised with default cache backend (%s)",
        type(cache).__name__,
    )
