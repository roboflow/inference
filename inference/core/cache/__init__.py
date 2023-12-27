from inference.core.cache.memory import MemoryCache
from inference.core.cache.redis import RedisCache
from inference.core.env import REDIS_HOST, REDIS_PORT, REDIS_SSL

if REDIS_HOST is not None:
    cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT, ssl=REDIS_SSL)
else:
    cache = MemoryCache()
