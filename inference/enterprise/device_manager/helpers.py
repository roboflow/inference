import time

from inference.core.cache import cache
from inference.core.env import METRICS_INTERVAL


def get_model_ids_by_server_id():
    now = time.time()
    start = now - METRICS_INTERVAL
    models = cache.zrangebyscore("models", min=start, max=now)
    model_ids_by_server_id = dict()
    for model in models:
        server_id, model_id = model.split(":")
        if server_id not in model_ids_by_server_id:
            model_ids_by_server_id[server_id] = []
        model_ids_by_server_id[server_id].append(model_id)
    return model_ids_by_server_id
