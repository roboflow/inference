import time
import requests

from inference.core.logger import logger
from inference.core.cache import cache
from inference.enterprise.device_manager.version import __version__
from inference.core.managers.metrics import get_model_metrics, get_system_info
from inference.enterprise.device_manager.helpers import get_model_ids_by_server_id
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.enterprise.device_manager.container_service import container_service
from inference.core.env import (
    API_KEY,
    METRICS_INTERVAL,
    METRICS_URL,
    TAGS,
)


def aggregate_model_stats(container_id):
    now = time.time()
    start = now - METRICS_INTERVAL
    model_ids = get_model_ids_by_server_id().get(container_id, [])
    models = []
    for model_id in model_ids:
        reqs = cache.zrangebyscore(
            f"inference:{container_id}:{model_id}", min=start, max=now
        )
        model = {
            "dataset_id": model_id.split("/")[0],
            "version": model_id.split("/")[1],
            "api_key": reqs[0]["request"]["api_key"],
            "metrics": get_model_metrics(container_id, model_id, min=start, max=now),
        }
        models.append(model)
    return models


def build_container_stats():
    containers = []
    for id in container_service.get_container_ids():
        container = container_service.get_container_by_id(id)
        if container:
            container_stats = {}
            models = aggregate_model_stats(id)
            container_stats["uuid"] = container.id
            container_stats["startup_time"] = container.startup_time
            container_stats["models"] = models
            containers.append(container_stats)
    return containers


def aggregate_device_stats():
    window_start_timestamp = str(int(time.time()))
    all_data = {
        "api_key": API_KEY,
        "timestamp": window_start_timestamp,
        "device": {
            "id": GLOBAL_DEVICE_ID,
            "name": GLOBAL_DEVICE_ID,
            "type": f"roboflow-device-manager=={__version__}",
            "tags": TAGS,
            "system_info": get_system_info(),
            "containers": build_container_stats(),
        },
    }
    return all_data


def report_metrics():
    all_data = aggregate_device_stats()
    logger.info(f"Sending metrics to Roboflow {str(all_data)}.")
    res = requests.post(METRICS_URL, json=all_data)
    res.raise_for_status()
    logger.debug("Sent metrics to Roboflow")
