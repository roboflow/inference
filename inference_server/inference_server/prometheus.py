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
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

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

logger = logging.getLogger(__name__)

METRICS_ENDPOINT = "/metrics"
# Same window and per-scrape model cap as the legacy server.
MODEL_METRICS_TIME_WINDOW_S = 10
MODEL_METRICS_MAX_MODELS = 25

_NON_METRIC_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_]")

# (timestamp, cumulative inference count, cumulative error count)
_Sample = Tuple[float, int, int]


def sanitize_metric_name_part(value: str) -> str:
    return _NON_METRIC_NAME_CHARS.sub("_", value)


class ModelMetricsCollector(Collector):
    """Legacy-named per-model gauges built from model manager stats.

    The model manager reports cumulative ``inference_count`` / ``error_count``
    per model. The legacy gauges are "how many in the last N seconds", so
    every scrape stores a sample and the value is the delta against the
    newest sample at least ``time_window`` old (or the oldest one kept),
    rescaled to ``time_window`` seconds. The first scrape after a model
    appears has no baseline and reports 0.
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
        self._history: Dict[str, Deque[_Sample]] = {}
        self._current: Dict[str, Dict[str, float]] = {}

    def observe(self, stats: Optional[dict], now: Optional[float] = None) -> None:
        """Record a stats snapshot. ``stats`` is the gateway ``stats()`` dict."""
        now = time.monotonic() if now is None else now
        models = (stats or {}).get("models") or {}
        if isinstance(models, list):
            models = {m.get("model_id"): m for m in models if m.get("model_id")}
        current: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for model_id in list(models)[: self.max_models]:
                entry = models[model_id] or {}
                try:
                    count = int(entry.get("inference_count") or 0)
                    errors = int(entry.get("error_count") or 0)
                    p50_ms = float(entry.get("latency_p50_ms") or 0.0)
                except (TypeError, ValueError):
                    logger.debug("Malformed stats for model %s", model_id)
                    continue
                history = self._history.setdefault(model_id, deque())
                num_inferences, num_errors = self._windowed_deltas(
                    history, now, count, errors
                )
                history.append((now, count, errors))
                current[model_id] = {
                    "num_inferences": num_inferences,
                    "num_errors": num_errors,
                    "avg_inference_time": p50_ms / 1000.0 if num_inferences else 0.0,
                }
            for gone in set(self._history) - set(current):
                del self._history[gone]
            self._current = current

    def _windowed_deltas(
        self, history: Deque[_Sample], now: float, count: int, errors: int
    ) -> Tuple[float, float]:
        cutoff = now - self.time_window
        # Keep only the newest sample at/before the cutoff plus newer ones.
        while len(history) >= 2 and history[1][0] <= cutoff:
            history.popleft()
        if not history:
            return 0.0, 0.0
        base_ts, base_count, base_errors = history[0]
        elapsed = now - base_ts
        if elapsed <= 0:
            return 0.0, 0.0
        # A drop means the model was reloaded and its counters restarted.
        d_count = count - base_count if count >= base_count else count
        d_errors = errors - base_errors if errors >= base_errors else errors
        scale = self.time_window / elapsed
        return d_count * scale, d_errors * scale

    def collect(self) -> Iterable[GaugeMetricFamily]:
        with self._lock:
            current = dict(self._current)
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
                "Median inference time in seconds of the model's recent "
                f"inferences (0 when none in {self.time_window}s)",
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
            "Sum over models of avg_inference_time_<model>",
            value=avg_inference_time_total,
        )
        yield GaugeMetricFamily(
            "num_errors_total",
            f"Total number of errors in {self.time_window}s",
            value=num_errors_total,
        )


def build_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)
    return registry


def install_prometheus_metrics(
    app: FastAPI, endpoint: str = METRICS_ENDPOINT
) -> ModelMetricsCollector:
    """Instrument ``app`` and serve the Prometheus text format at ``endpoint``.

    Must run before the app starts (it adds middleware) and before any
    catch-all route or root static mount, which would shadow ``endpoint``.
    """
    registry = build_registry()
    Instrumentator(registry=registry).instrument(app)
    collector = ModelMetricsCollector()
    registry.register(collector)

    @app.get(endpoint, include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        model_manager = getattr(request.app.state, "model_manager", None)
        if model_manager is None:
            collector.observe(None)
        else:
            try:
                stats: Dict[str, Any] = await model_manager.stats()
            except Exception:
                # Keep the previous model gauges; HTTP/process metrics still
                # render.
                logger.debug("Model stats unavailable for /metrics", exc_info=True)
            else:
                collector.observe(stats)
        return Response(
            content=generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return collector
