import asyncio
import concurrent.futures
import re
import time
from typing import Callable, Dict, List
from urllib.parse import urlparse, urlunparse

from prometheus_client.core import REGISTRY, CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector
from prometheus_fastapi_instrumentator import Instrumentator

from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.env import METRICS_INCLUDE_SOURCE_LABELS
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

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.collector.stream_manager_client = stream_manager_client


class CustomCollector(Collector):
    def __init__(self, model_manager, time_window: int = 10):
        super(CustomCollector, self).__init__()
        self.model_manager = model_manager
        self.time_window = time_window
        self.stream_manager_client = None

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

    async def _fetch_stream_metrics(self) -> Dict[str, dict]:
        # Pipeline status is fetched via TCP IPC to the stream manager process.
        # Pipelines run in separate subprocesses, so socket-based IPC is required.
        pipelines_response = await self.stream_manager_client.list_pipelines()
        pipeline_ids = pipelines_response.pipelines
        metrics = {}
        for pipeline_id in pipeline_ids:
            status_response = await self.stream_manager_client.get_status(pipeline_id)
            report = status_response.report
            latency_reports = report.get("latency_reports", [])
            sources_metadata = report.get("sources_metadata", [])
            camera_fps = self._average_source_fps(sources_metadata)
            source_label = self._extract_source_label(sources_metadata)
            metrics[pipeline_id] = {
                "inference_throughput": report.get("inference_throughput", 0.0),
                "camera_fps": camera_fps,
                "frame_decoding_latency": self._average_latency_field(
                    latency_reports, "frame_decoding_latency"
                ),
                "inference_latency": self._average_latency_field(
                    latency_reports, "inference_latency"
                ),
                "e2e_latency": self._average_latency_field(
                    latency_reports, "e2e_latency"
                ),
                "source": source_label,
            }
        return metrics

    def get_stream_metrics(self) -> Dict[str, dict]:
        if self.stream_manager_client is None:
            return {}
        try:
            try:
                return asyncio.run(self._fetch_stream_metrics())
            except RuntimeError:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run, self._fetch_stream_metrics()
                    ).result()
        except Exception:
            logger.debug("Failed to fetch stream metrics", exc_info=True)
            return {}

    @staticmethod
    def _average_latency_field(latency_reports: List[dict], field: str) -> float:
        values = [r[field] for r in latency_reports if r.get(field) is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _average_source_fps(sources_metadata: List[dict]) -> float:
        values = []
        for src in sources_metadata:
            props = src.get("source_properties") or {}
            fps = props.get("fps")
            if fps is not None and fps > 0:
                values.append(fps)
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _sanitize_source_reference(ref: str) -> str:
        """Strip credentials and query parameters from URLs to avoid leaking
        secrets in metrics."""
        parsed = urlparse(ref)
        if parsed.scheme and parsed.hostname:
            netloc = parsed.hostname + (f":{parsed.port}" if parsed.port else "")
            sanitized = parsed._replace(netloc=netloc, query="", fragment="")
            return urlunparse(sanitized)
        return ref

    @staticmethod
    def _extract_source_label(sources_metadata: List[dict]) -> str:
        if not METRICS_INCLUDE_SOURCE_LABELS:
            return ""
        refs = []
        for src in sources_metadata:
            ref = src.get("source_reference")
            if ref is not None:
                refs.append(CustomCollector._sanitize_source_reference(str(ref)))
        return ",".join(refs) if refs else ""

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

        stream_metrics = self.get_stream_metrics()
        pipeline_labels = ["pipeline_id", "source"]
        inference_fps = GaugeMetricFamily(
            "inference_pipeline_inference_fps",
            "Inference throughput FPS",
            labels=pipeline_labels,
        )
        camera_fps = GaugeMetricFamily(
            "inference_pipeline_camera_fps",
            "Camera source FPS",
            labels=pipeline_labels,
        )
        frame_decoding_latency = GaugeMetricFamily(
            "inference_pipeline_frame_decoding_latency",
            "Average frame decoding latency (seconds)",
            labels=pipeline_labels,
        )
        inference_latency = GaugeMetricFamily(
            "inference_pipeline_inference_latency",
            "Average inference latency (seconds)",
            labels=pipeline_labels,
        )
        e2e_latency = GaugeMetricFamily(
            "inference_pipeline_e2e_latency",
            "Average end-to-end latency (seconds)",
            labels=pipeline_labels,
        )
        for pipeline_id, pm in stream_metrics.items():
            label_values = [pipeline_id, pm["source"]]
            inference_fps.add_metric(label_values, pm["inference_throughput"])
            camera_fps.add_metric(label_values, pm["camera_fps"])
            frame_decoding_latency.add_metric(
                label_values, pm["frame_decoding_latency"]
            )
            inference_latency.add_metric(label_values, pm["inference_latency"])
            e2e_latency.add_metric(label_values, pm["e2e_latency"])
        yield inference_fps
        yield camera_fps
        yield frame_decoding_latency
        yield inference_latency
        yield e2e_latency
        yield GaugeMetricFamily(
            "inference_pipeline_active_streams",
            "Number of active inference pipelines",
            value=len(stream_metrics),
        )
