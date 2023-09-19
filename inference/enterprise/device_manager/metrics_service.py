import time
import requests

from inference.core.logger import logger
from inference.core.cache import cache
from inference.core.version import __version__
from inference.core.managers.metrics import get_model_metrics, get_system_info
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.enterprise.device_manager.container_service import ContainerService
from inference.core.env import (
    API_KEY,
    METRICS_INTERVAL,
    METRICS_URL,
    TAGS,
)


class MetricsService:
    def __init__(self):
        self.container_service = ContainerService()

    def get_model_ids_by_server_id(self):
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

    def aggregate_model_stats(self, container_id):
        now = time.time()
        start = now - METRICS_INTERVAL
        model_ids = self.get_model_ids_by_server_id().get(container_id, [])
        models = []
        for model_id in model_ids:
            model_metrics = get_model_metrics(
                container_id, model_id, min=start, max=now
            )
            models.append(model_metrics)
        return models

    def build_container_stats(self):
        containers = []
        for id in self.container_service.get_container_ids():
            container = self.container_service.get_container_by_id(id)
            if container:
                container_stats = {}
                container_stats["uuid"] = container.id
                # TODO get real container startup_time
                container_stats["startup_time"] = str(int(time.time()))
                container_stats["models"] = self.aggregate_model_stats(id)
                containers.append(container_stats)
        return containers

    def aggregate_device_stats(self):
        window_start_timestamp = str(int(time.time()))
        all_data = {
            "api_key": API_KEY,
            "timestamp": window_start_timestamp,
            "device": {
                "id": GLOBAL_DEVICE_ID,
                "name": GLOBAL_DEVICE_ID,
                "type": f"roboflow-inference-server=={__version__}",
                "tags": TAGS,
                "system_info": get_system_info(),
                "containers": self.build_container_stats(),
            },
        }
        return all_data

    def report_metrics(self):
        all_data = self.aggregate_device_stats()
        logger.info(
            "Sending metrics to Roboflow {} {}.".format(METRICS_URL, str(all_data))
        )
        res = requests.post(METRICS_URL, json=all_data)
        res.raise_for_status()
        logger.debug(
            "Sent metrics to Roboflow {} at {}.".format(METRICS_URL, str(all_data))
        )
