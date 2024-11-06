import re
import time
from typing import Callable

from prometheus_client.core import REGISTRY, CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector
from prometheus_fastapi_instrumentator import Instrumentator

from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.logger import logger
from inference.core.managers.metrics import get_model_metrics


class InferenceInstrumentator:
    """
    Class responsible for managing the Prometheus metrics for the inference server.

    This class inititalizes the Prometheus Instrumentator and exposes the metrics endpoint.

    """

    def __init__(self, app, model_manager, endpoint: str = "/metrics"):
        self.instrumentator = Instrumentator()
        self.instrumentator.instrument(app).expose(app, endpoint)
        self.collector = CustomCollector(model_manager)
        REGISTRY.register(self.collector)


class CustomCollector(Collector):
    def __init__(self, model_manager, time_window: int = 10):
        super(CustomCollector, self).__init__()
        self.model_manager = model_manager
        self.time_window = time_window

    def get_metrics(self, maxModels: int = 25):
        now = time.time()
        start = now - self.time_window
        count = 0
        results = {}
        if self.model_manager is None:
            logger.warning(
                "This inference server type does not support custom Prometheus metrics, skipping."
            )
            return results
        for model_id in self.model_manager.models():
            if count >= maxModels:
                break
            try:
                results[model_id] = get_model_metrics(
                    GLOBAL_INFERENCE_SERVER_ID, model_id, min=start, max=now
                )
            except Exception as e:
                logger.debug(
                    "Error getting metrics for model " + model_id + ": " + str(e)
                )
            count += 1
        return results

    def sanitize_string(self, input_string):
        sanitized_string = re.sub(r"[^a-zA-Z0-9_]", "_", input_string)
        return sanitized_string

    def collect(self):
        results = self.get_metrics()
        num_inferences_total = 0
        num_errors_total = 0
        avg_inference_time_total = 0
        for model_id, metrics in results.items():
            sane_model_id = self.sanitize_string(model_id)
            yield GaugeMetricFamily(
                f"num_inferences_{sane_model_id}",
                f"Number of inferences made in {self.time_window}s",
                value=metrics["num_inferences"],
            )
            yield GaugeMetricFamily(
                f"avg_inference_time_{sane_model_id}",
                f"Average inference time (over inferences completed in {self.time_window}s) to infer this model",
                value=metrics["avg_inference_time"],
            )
            yield GaugeMetricFamily(
                f"num_errors_{sane_model_id}",
                f"Number of errors in {self.time_window}s",
                value=metrics["num_errors"],
            )
            num_inferences_total += metrics["num_inferences"]
            num_errors_total += metrics["num_errors"]
            avg_inference_time_total += metrics["avg_inference_time"]
        yield GaugeMetricFamily(
            "num_inferences_total",
            f"Total number of inferences made in {self.time_window}s",
            value=num_inferences_total,
        )
        yield GaugeMetricFamily(
            "avg_inference_time_total",
            f"Average inference time (over inferences completed in {self.time_window}s) to infer all models.",
            value=avg_inference_time_total,
        )
        yield GaugeMetricFamily(
            "num_errors_total",
            f"Total number of errors in {self.time_window}s",
            value=num_errors_total,
        )
