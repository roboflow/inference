"""Prometheus text metrics at ``GET /metrics``.

Mirrors what the legacy ``inference`` server exposes on the same path, so
existing scrape configs keep working:

- HTTP request metrics from ``prometheus-fastapi-instrumentator``
  (``http_requests_total``, ``http_request_duration_seconds``, ...).
- Process, platform and GC collectors from ``prometheus_client``
  (``process_*``, ``python_info``, ``python_gc_*``).
- Per-model gauges with the legacy names: ``num_inferences_<model>``,
  ``avg_inference_time_<model>``, ``num_errors_<model>`` and the
  ``*_total`` roll-ups, computed over a short time window.
- Stream pipeline gauges with the legacy names (``inference_pipeline_*``),
  read from ``app.state.stream_manager_client`` when the app sets one.

Each app gets its own ``CollectorRegistry`` — nothing is registered in the
process-global ``prometheus_client.REGISTRY``, so building the app twice
(tests reload ``inference_server.app``) never hits duplicated-timeseries
errors.

``/v2/server/metrics`` (JSON snapshot, control-plane gated) is unrelated and
left as is.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import SplitResult, urlsplit

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from inference_server import configuration

logger = logging.getLogger(__name__)

METRICS_ENDPOINT = "/metrics"
# Same window and per-scrape model cap as the legacy server.
MODEL_METRICS_TIME_WINDOW_S = 10
MODEL_METRICS_MAX_MODELS = 25

_NON_METRIC_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_]")

# (monotonic finish time, per-response times in seconds, error)
_Sample = Tuple[float, Tuple[float, ...], bool]

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_URL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_EMBEDDED_CREDENTIALS_RE = re.compile(r"(://)[^/@:\s]+(?::[^/\s]*)?@")
_PATH_HOSTPORT_AFTER_AT_RE = re.compile(r"@[^/@]*:\d+(?:/|$)")
UNPARSEABLE_SOURCE = "<unparseable source>"


def sanitize_metric_name_part(value: str) -> str:
    return _NON_METRIC_NAME_CHARS.sub("_", value)


def _has_parseable_port(parts: SplitResult) -> bool:
    try:
        parts.port
    except ValueError:
        return False
    return True


def _sanitize_schemed_url(ref: str) -> str:
    if ref[:7].lower() == "file://":
        return ref
    for candidate in (ref, _EMBEDDED_CREDENTIALS_RE.sub(r"\1", ref)):
        try:
            parts = urlsplit(candidate)
        except ValueError:
            continue
        if "@" not in parts.netloc and ":" in parts.netloc:
            if not _has_parseable_port(parts):
                if "@" in candidate:
                    continue
            elif _PATH_HOSTPORT_AFTER_AT_RE.search(parts.path):
                continue
        credential_free_netloc = parts.netloc.rsplit("@", 1)[-1].lower()
        return f"{parts.scheme}://{credential_free_netloc}{parts.path}"
    return UNPARSEABLE_SOURCE


def _looks_like_host_with_port(host_port: str) -> bool:
    _, sep, port = host_port.rpartition(":")
    return bool(sep) and port.isdigit()


def _strip_schemeless_userinfo(ref: str) -> Optional[str]:
    path_start = ref.find("/")
    if path_start == -1:
        authority, path = ref, ""
    else:
        authority, path = ref[:path_start], ref[path_start:]
    if "@" not in authority:
        return None
    userinfo, _, host_port = authority.rpartition("@")
    has_password = ":" in userinfo and "\\" not in userinfo
    if not has_password and not _looks_like_host_with_port(host_port):
        return None
    return (host_port + path).split("?", 1)[0].split("#", 1)[0]


def sanitize_source_reference(ref: str) -> str:
    """Strip credentials and query parameters from a video source reference.

    Same rules as the stream source sanitiser used for observability output.

    Args:
        ref: Source reference, such as an RTSP URL or a file path.

    Returns:
        The reference without credentials or query parameters, or
        ``UNPARSEABLE_SOURCE`` when credentials cannot be separated safely.
    """
    if _SCHEME_RE.match(ref):
        sanitized = _sanitize_schemed_url(ref)
        return sanitized

    stripped = _strip_schemeless_userinfo(ref)
    sanitized = _URL_TOKEN_RE.sub(
        lambda match: _sanitize_schemed_url(match.group(0)),
        ref if stripped is None else stripped,
    )

    return sanitized


def _identity(model_id: str) -> str:
    return model_id


class ModelMetricsCollector(Collector):
    """Legacy-named per-model gauges built from recorded inferences.

    Every completed inference request is recorded under the loaded model's
    registry id with its finish time, the time of each response it produced
    and its outcome. A scrape reports the first ``max_models`` loaded models,
    in load order, under their display ids: the successful requests and the
    errors finished in the last ``time_window`` seconds and the mean of those
    successful requests' response times. Idle loaded models read zeros;
    samples of models that are no longer loaded are kept until they expire
    but not reported.
    """

    def __init__(
        self,
        time_window: float = MODEL_METRICS_TIME_WINDOW_S,
        max_models: int = MODEL_METRICS_MAX_MODELS,
    ):
        super().__init__()
        self.time_window = time_window
        self.max_models = max_models
        self._lock = threading.Lock()
        self._samples: Dict[str, Deque[_Sample]] = {}
        self._loaded: List[Tuple[str, str]] = []

    def record(
        self,
        model_id: str,
        response_times: Sequence[float],
        *,
        error: bool,
        now: Optional[float] = None,
    ) -> None:
        """Record one completed inference request.

        Args:
            model_id: Registry id of the model that served the request.
            response_times: Time in seconds reported for each response.
            error: Whether the inference raised.
            now: Finish time on the monotonic clock; defaults to now.
        """
        now = time.monotonic() if now is None else now
        sample = (now, tuple(response_times), error)
        with self._lock:
            self._prune(now)
            self._samples.setdefault(model_id, deque()).append(sample)

    def observe_loaded(
        self,
        model_ids: Iterable[str],
        resolve_model_id: Callable[[str], str] = _identity,
    ) -> None:
        """Set the models currently loaded, from the gateway ``stats()``.

        Args:
            model_ids: Registry ids of the loaded models, in load order.
            resolve_model_id: Maps a registry id to the id its gauges are
                named after.
        """
        loaded = [
            (model_id, resolve_model_id(model_id))
            for model_id in dict.fromkeys(model_ids)
        ]
        with self._lock:
            self._loaded = loaded

    def clear(self) -> None:
        """Drop every recorded inference and the loaded models."""
        with self._lock:
            self._samples.clear()
            self._loaded = []

    def metrics(self, now: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Compute the legacy per-model values over the current window.

        Args:
            now: Time on the monotonic clock; defaults to now.

        Returns:
            ``num_inferences``, ``num_errors`` and ``avg_inference_time`` keyed
            by display id.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            snapshot = [
                (display_id, list(self._samples.get(model_id, ())))
                for model_id, display_id in self._loaded[: self.max_models]
            ]

        results = {}
        for display_id, samples in snapshot:
            successes = [times for _, times, error in samples if not error]
            response_times = [value for times in successes for value in times]
            results.setdefault(
                display_id,
                {
                    "num_inferences": len(successes),
                    "num_errors": len(samples) - len(successes),
                    "avg_inference_time": (
                        sum(response_times) / len(response_times)
                        if response_times
                        else 0.0
                    ),
                },
            )

        return results

    def _prune(self, now: float) -> None:
        cutoff = now - self.time_window
        for model_id in list(self._samples):
            samples = self._samples[model_id]
            while samples and samples[0][0] < cutoff:
                samples.popleft()
            if not samples:
                del self._samples[model_id]

    def collect(self) -> Iterable[GaugeMetricFamily]:
        current = self.metrics()
        num_inferences_total = 0.0
        num_errors_total = 0.0
        avg_inference_time_total = 0.0
        for model_id, metrics in current.items():
            sane_model_id = sanitize_metric_name_part(model_id)
            yield GaugeMetricFamily(
                f"num_inferences_{sane_model_id}",
                f"Number of inferences made in {self.time_window}s",
                value=metrics["num_inferences"],
            )
            yield GaugeMetricFamily(
                f"avg_inference_time_{sane_model_id}",
                "Average inference time (over inferences completed in "
                f"{self.time_window}s) to infer this model",
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
            "Average inference time (over inferences completed in "
            f"{self.time_window}s) to infer all models.",
            value=avg_inference_time_total,
        )
        yield GaugeMetricFamily(
            "num_errors_total",
            f"Total number of errors in {self.time_window}s",
            value=num_errors_total,
        )


MODEL_METRICS = ModelMetricsCollector()


def record_inference(
    model_id: str, response_times: Sequence[float], error: bool
) -> None:
    """Record one completed inference request for the per-model gauges.

    Args:
        model_id: Registry id of the model that served the request.
        response_times: Time in seconds reported for each response.
        error: Whether the inference raised.
    """
    MODEL_METRICS.record(model_id, response_times, error=error)


@contextmanager
def measure_inference(
    model_id: str, *, responses: int, monitoring: bool = True
) -> Iterator[None]:
    """Time the enclosed inference request and record it.

    Nothing is recorded when ``monitoring`` is False or under ``OFFLINE_MODE``
    or ``DISABLE_INFERENCE_CACHE``. An exception raised inside the block is
    recorded as an error and re-raised; on success every response is
    recorded with the request's wall time.

    Args:
        model_id: Registry id of the model that serves the request.
        responses: Number of responses the request produces.
        monitoring: False when the request opted out of model monitoring.

    Yields:
        None.
    """
    enabled = (
        monitoring
        and not configuration.OFFLINE_MODE
        and not configuration.DISABLE_INFERENCE_CACHE
    )
    started = time.monotonic()
    try:
        yield
    except Exception:
        if enabled:
            record_inference(model_id, (), True)
        raise
    if enabled:
        elapsed = time.monotonic() - started
        record_inference(model_id, [elapsed] * responses, False)


def _average_latency_field(latency_reports: List[dict], field: str) -> float:
    values = [r[field] for r in latency_reports if r.get(field) is not None]
    if not values:
        return 0.0

    average = sum(values) / len(values)

    return average


def _average_source_fps(sources_metadata: List[dict]) -> float:
    values = []
    for source in sources_metadata:
        properties = source.get("source_properties") or {}
        fps = properties.get("fps")
        if fps is not None and fps > 0:
            values.append(fps)
    if not values:
        return 0.0

    average = sum(values) / len(values)

    return average


def _extract_source_label(sources_metadata: List[dict]) -> str:
    if not configuration.METRICS_INCLUDE_SOURCE_LABELS:
        return ""

    references = [
        sanitize_source_reference(str(source["source_reference"]))
        for source in sources_metadata
        if source.get("source_reference") is not None
    ]
    label = ",".join(references)

    return label


async def fetch_stream_metrics(stream_manager_client: Any) -> Dict[str, dict]:
    """Read per-pipeline gauge values from the stream manager.

    Args:
        stream_manager_client: Client exposing async ``list_pipelines()`` and
            ``get_status(pipeline_id)``, like the stream manager TCP client.

    Returns:
        Gauge values keyed by pipeline id.
    """
    pipelines_response = await stream_manager_client.list_pipelines()
    metrics = {}
    for pipeline_id in pipelines_response.pipelines:
        status_response = await stream_manager_client.get_status(pipeline_id)
        report = status_response.report
        latency_reports = report.get("latency_reports", [])
        sources_metadata = report.get("sources_metadata", [])
        metrics[pipeline_id] = {
            "inference_throughput": report.get("inference_throughput", 0.0),
            "camera_fps": _average_source_fps(sources_metadata),
            "frame_decoding_latency": _average_latency_field(
                latency_reports, "frame_decoding_latency"
            ),
            "inference_latency": _average_latency_field(
                latency_reports, "inference_latency"
            ),
            "e2e_latency": _average_latency_field(latency_reports, "e2e_latency"),
            "source": _extract_source_label(sources_metadata),
        }

    return metrics


class StreamMetricsCollector(Collector):
    """Legacy-named stream pipeline gauges.

    Always yields every pipeline family and
    ``inference_pipeline_active_streams``; they are empty and 0 when no stream
    manager client is present or it could not be read.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._metrics: Dict[str, dict] = {}

    def observe(self, metrics: Dict[str, dict]) -> None:
        """Record the latest stream metrics.

        Args:
            metrics: Values from ``fetch_stream_metrics``; empty when no
                stream manager client is present or it could not be read.
        """
        with self._lock:
            self._metrics = metrics

    def collect(self) -> Iterable[GaugeMetricFamily]:
        with self._lock:
            stream_metrics = self._metrics

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


def build_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)
    return registry


def install_prometheus_metrics(
    app: FastAPI,
    endpoint: str = METRICS_ENDPOINT,
    resolve_model_id: Callable[[str], str] = _identity,
) -> ModelMetricsCollector:
    """Instrument ``app`` and serve the Prometheus text format at ``endpoint``.

    Must run before the app starts (it adds middleware) and before any
    catch-all route or root static mount, which would shadow ``endpoint``.

    Args:
        app: Application to instrument.
        endpoint: Path of the metrics route.
        resolve_model_id: Maps a loaded model's registry id from ``stats()``
            to the id its inferences are recorded under.

    Returns:
        The per-model metrics collector.
    """
    registry = build_registry()
    Instrumentator(registry=registry).instrument(app)
    registry.register(MODEL_METRICS)
    stream_collector = StreamMetricsCollector()
    registry.register(stream_collector)

    @app.get(endpoint, include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        model_manager = getattr(request.app.state, "model_manager", None)
        if model_manager is None:
            MODEL_METRICS.observe_loaded([])
        else:
            try:
                stats: Dict[str, Any] = await model_manager.stats()
            except Exception:
                logger.debug("Model stats unavailable for /metrics", exc_info=True)
            else:
                models = stats.get("models") or {}
                if isinstance(models, list):
                    models = [m.get("model_id") for m in models if m.get("model_id")]
                MODEL_METRICS.observe_loaded(models, resolve_model_id)
        stream_manager_client = getattr(
            request.app.state, "stream_manager_client", None
        )
        if stream_manager_client is None:
            stream_collector.observe({})
        else:
            try:
                stream_metrics = await fetch_stream_metrics(stream_manager_client)
            except Exception:
                logger.debug("Failed to fetch stream metrics", exc_info=True)
                stream_metrics = {}
            stream_collector.observe(stream_metrics)
        return Response(
            content=generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return MODEL_METRICS
