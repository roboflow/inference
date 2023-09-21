import time

from inference.core.cache import cache
from inference.core.env import METRICS_INTERVAL


def get_cache_model_items():
    """
    Retrieve and organize cached model items within a specified time interval.

    This method queries a cache for model items and retrieves those that fall
    within the time interval defined by the global constant METRICS_INTERVAL.
    It organizes the retrieved items into a hierarchical dictionary structure
    for efficient access.

    Returns:
        dict: A dictionary containing model items organized by server ID, API key,
        and model ID. The structure is as follows:
        - Keys: Server IDs associated with models.
          - Sub-keys: API keys associated with models on the server.
            - Values: Lists of model IDs associated with each API key on the server.

    Notes:
        - This method relies on a cache system for storing and retrieving model items.
        - It uses the global constant METRICS_INTERVAL to specify the time interval.
    """
    now = time.time()
    start = now - METRICS_INTERVAL
    models = cache.zrangebyscore("models", min=start, max=now)
    model_items = dict()
    for model in models:
        server_id, api_key, model_id = model.split(":")
        if server_id not in model_items:
            model_items[server_id] = dict()
        if api_key not in model_items[server_id]:
            model_items[server_id][api_key] = []
        model_items[server_id][api_key].append(model_id)
    return model_items
