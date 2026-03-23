"""OpenTelemetry tracing setup and helpers for the inference server.

All public helpers are safe to import and call even when opentelemetry is not
installed — they degrade to noops.  Business-logic code should never need to
check whether OTel is available.

Usage::

    from inference.core.telemetry import (
        start_span, record_error, set_span_attribute,
    )

    with start_span("model.infer", {"model.id": model_id}):
        result = do_inference(...)
        set_span_attribute("model.load_time_seconds", elapsed)

    record_error(error)
"""

import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Graceful opentelemetry imports — everything below is noop when not installed
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.propagate import inject as _otel_inject
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace import StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_tracer = None
_provider = None
_meter_provider = None
_metrics: Optional[Dict[str, Any]] = None

# Header name for forcing a trace on a specific request.
FORCE_TRACE_HEADER = b"x-force-trace"

# ContextVar set by _ForceTraceASGIMiddleware, read by _ForceTraceRootSampler.
# Per-request isolation is automatic in ASGI (each request runs in its own task).
_force_trace_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_force_trace_flag", default=False
)


# ---------------------------------------------------------------------------
# Public helpers — always safe to call
# ---------------------------------------------------------------------------


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a new span as a child of the current context.

    Yields the span (or None when OTel is not available).
    """
    if not _OTEL_AVAILABLE:
        yield None
        return
    tracer = _get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def record_error(error: Exception) -> None:
    """Record an exception on the current active span and set ERROR status.

    Safe to call when OTel is not installed or there is no active span.
    """
    if not _OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, str(error))


def get_trace_id() -> Optional[str]:
    """Return the current trace ID as a hex string, or None."""
    if not _OTEL_AVAILABLE:
        return None
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        return format(ctx.trace_id, "032x")
    return None


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current active span.

    Noop when OTel is unavailable or there is no recording span.
    Callers never need to check for None spans.
    """
    if not _OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def inject_trace_context(headers: dict) -> dict:
    """Inject W3C traceparent/tracestate into *headers* dict and return it.

    Safe to call when OTel is not installed (returns headers unchanged).
    """
    if not _OTEL_AVAILABLE:
        return headers
    if headers is None:
        headers = {}
    _otel_inject(headers)
    return headers


def trace_context_log_processor(
    logger_instance: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Structlog processor that injects trace_id and span_id into log entries."""
    if not _OTEL_AVAILABLE:
        return event_dict
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


# ---------------------------------------------------------------------------
# Metrics helpers — always safe to call
# ---------------------------------------------------------------------------


def record_model_loaded(model_id: str, load_time: float) -> None:
    """Record a model load event: increment counters and record load duration."""
    if _metrics is None:
        return
    _metrics["models_loaded"].add(1)
    _metrics["model_loads"].add(1, {"model.id": model_id})
    _metrics["model_load_duration"].record(load_time, {"model.id": model_id})


def record_model_unloaded(model_id: str) -> None:
    """Record a model unload event."""
    if _metrics is None:
        return
    _metrics["models_loaded"].add(-1)
    _metrics["model_unloads"].add(1, {"model.id": model_id})


def record_inference(model_id: str, duration: float) -> None:
    """Record an inference execution."""
    if _metrics is None:
        return
    _metrics["model_infer_count"].add(1, {"model.id": model_id})
    _metrics["model_infer_duration"].record(duration, {"model.id": model_id})


def record_api_call(function_name: str, duration: float) -> None:
    """Record a Roboflow API call duration."""
    if _metrics is None:
        return
    _metrics["api_call_duration"].record(
        duration, {"roboflow_api.function": function_name}
    )


def record_error_metric(error_type: str) -> None:
    """Increment the error counter by error type."""
    if _metrics is None:
        return
    _metrics["errors"].add(1, {"error.type": error_type})


# ---------------------------------------------------------------------------
# Tracing ModelAccessManager for inference-models package
# ---------------------------------------------------------------------------

try:
    from inference_models.models.auto_loaders.access_manager import (
        AccessIdentifiers,
        LiberalModelAccessManager,
    )
    from inference_models.models.auto_loaders.entities import AnyModel

    _INFERENCE_MODELS_AVAILABLE = True
except ImportError:
    _INFERENCE_MODELS_AVAILABLE = False


def create_tracing_model_access_manager():
    """Create a ModelAccessManager that adds OTel spans to model loading.

    Returns None when OTel or inference_models is unavailable — callers
    should pass the result directly to AutoModel.from_pretrained() which
    falls back to its default when it receives None.
    """
    if not _OTEL_AVAILABLE or not _INFERENCE_MODELS_AVAILABLE:
        return None
    return _TracingModelAccessManager()


if _INFERENCE_MODELS_AVAILABLE:

    class _TracingModelAccessManager(LiberalModelAccessManager):
        """ModelAccessManager that wraps model loading in an OTel span.

        Hooks called by AutoModel.from_pretrained():
          on_model_package_access_granted  -> start span
          on_file_created / on_file_renamed -> span events
          on_model_loaded                  -> end span
        """

        def __init__(self):
            super().__init__()
            self._span = None
            self._token = None

        def on_model_package_access_granted(
            self, access_identifiers: AccessIdentifiers
        ) -> None:
            super().on_model_package_access_granted(access_identifiers)
            if not _OTEL_AVAILABLE:
                return
            tracer = _get_tracer()
            if tracer is None:
                return
            self._span = tracer.start_span(
                "inference_models.load",
                attributes={
                    "model.id": access_identifiers.model_id,
                    "model.package_id": access_identifiers.package_id,
                },
            )
            from opentelemetry import context

            self._token = context.attach(trace.set_span_in_context(self._span))

        def on_file_created(
            self, file_path: str, access_identifiers: AccessIdentifiers
        ) -> None:
            super().on_file_created(file_path, access_identifiers)
            if self._span is not None and self._span.is_recording():
                self._span.add_event("file_created", {"file.path": file_path})

        def on_file_renamed(
            self,
            old_path: str,
            new_path: str,
            access_identifiers: AccessIdentifiers,
        ) -> None:
            super().on_file_renamed(old_path, new_path, access_identifiers)
            if self._span is not None and self._span.is_recording():
                self._span.add_event(
                    "file_renamed",
                    {"file.old_path": old_path, "file.new_path": new_path},
                )

        def on_model_alias_discovered(self, alias: str, model_id: str) -> None:
            super().on_model_alias_discovered(alias, model_id)
            if self._span is not None and self._span.is_recording():
                self._span.set_attribute("model.alias", alias)

        def on_model_dependency_discovered(
            self,
            base_model_id: str,
            base_model_package_id: Optional[str],
            dependent_model_id: str,
        ) -> None:
            super().on_model_dependency_discovered(
                base_model_id, base_model_package_id, dependent_model_id
            )
            if self._span is not None and self._span.is_recording():
                self._span.add_event(
                    "dependency_discovered",
                    {
                        "model.base_id": base_model_id,
                        "model.dependent_id": dependent_model_id,
                    },
                )

        def on_model_access_forbidden(
            self, model_id: str, api_key: Optional[str]
        ) -> None:
            super().on_model_access_forbidden(model_id, api_key)
            record_error(PermissionError(f"Model access forbidden: {model_id}"))

        def on_model_loaded(
            self,
            model: "AnyModel",
            access_identifiers: AccessIdentifiers,
            model_storage_path: str,
        ) -> None:
            super().on_model_loaded(model, access_identifiers, model_storage_path)
            if self._span is not None:
                self._span.set_attribute("model.storage_path", model_storage_path)
                self._span.end()
                if self._token is not None:
                    from opentelemetry import context

                    context.detach(self._token)
                self._span = None
                self._token = None


# ---------------------------------------------------------------------------
# Force-trace ASGI middleware + custom sampler
# ---------------------------------------------------------------------------


class _ForceTraceASGIMiddleware:
    """Lightweight ASGI middleware that detects X-Force-Trace header.

    Must wrap the app OUTSIDE the OTel instrumentor so the ContextVar is set
    before the instrumentor's should_sample() call.  We achieve this by adding
    it via app.add_middleware() AFTER FastAPIInstrumentor.instrument_app() —
    Starlette builds the middleware stack so the last-added middleware is
    outermost.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            for header_name, header_value in scope.get("headers", []):
                if header_name == FORCE_TRACE_HEADER:
                    if header_value.lower() == b"true":
                        _force_trace_flag.set(True)
                    break
        await self.app(scope, receive, send)


class _ForceTraceRootSampler:
    """Custom root sampler that force-samples when X-Force-Trace is set.

    Used as the ``root`` argument to ParentBased:
      - Parent exists  -> ParentBased honours the parent's decision (not us)
      - No parent, X-Force-Trace: true -> always sample
      - No parent, no header -> delegate to the ratio-based sampler
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def should_sample(
        self,
        parent_context: Any,
        trace_id: int,
        name: str,
        kind: Any = None,
        attributes: Any = None,
        links: Optional[Sequence[Any]] = None,
        trace_state: Any = None,
    ) -> Any:
        if _force_trace_flag.get(False):
            from opentelemetry.sdk.trace.sampling import Decision, SamplingResult

            return SamplingResult(
                decision=Decision.RECORD_AND_SAMPLE,
                attributes={"sampling.forced": True},
                trace_state=trace_state,
            )
        return self._delegate.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"ForceTraceRootSampler({self._delegate.get_description()})"


# ---------------------------------------------------------------------------
# Setup / shutdown — called only when OTEL_TRACING_ENABLED is True
# ---------------------------------------------------------------------------


def setup_telemetry(app: Any) -> None:
    """Initialize OTel TracerProvider, MeterProvider, and instrument the FastAPI app.

    Must be called before any middleware is added so the FastAPI instrumentor
    wraps at the outermost ASGI layer.
    """
    global _provider, _tracer, _meter_provider, _metrics

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_OFF,
        ALWAYS_ON,
        ParentBased,
        TraceIdRatioBased,
    )

    from inference.core.env import (
        OTEL_EXPORTER_ENDPOINT,
        OTEL_EXPORTER_PROTOCOL,
        OTEL_METRIC_EXPORT_INTERVAL_MS,
        OTEL_SAMPLING_RATE,
        OTEL_SERVICE_NAME,
        OTEL_TRACE_EXPORT_INTERVAL_MS,
    )

    # W3C TraceContext propagator — always set so extract/inject are safe
    set_global_textmap(CompositePropagator([TraceContextTextMapPropagator()]))

    # Build root sampler with force-trace override
    if OTEL_SAMPLING_RATE <= 0:
        root_sampler = ALWAYS_OFF
    elif OTEL_SAMPLING_RATE >= 1.0:
        root_sampler = ALWAYS_ON
    else:
        root_sampler = TraceIdRatioBased(OTEL_SAMPLING_RATE)

    # ParentBased: child spans honour parent decision, root spans use our
    # custom sampler that checks for X-Force-Trace before falling back to
    # the ratio-based sampler.
    sampler = ParentBased(root=_ForceTraceRootSampler(root_sampler))

    from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID

    resource = Resource.create(
        {
            "service.name": OTEL_SERVICE_NAME,
            "service.instance.id": GLOBAL_INFERENCE_SERVER_ID,
        }
    )

    if OTEL_EXPORTER_PROTOCOL == "http":
        exporter = HTTPExporter(
            endpoint=f"http://{OTEL_EXPORTER_ENDPOINT}/v1/traces",
        )
    else:
        exporter = GRPCExporter(
            endpoint=OTEL_EXPORTER_ENDPOINT,
            insecure=True,
        )

    _provider = TracerProvider(resource=resource, sampler=sampler)
    _provider.add_span_processor(
        BatchSpanProcessor(
            exporter, schedule_delay_millis=OTEL_TRACE_EXPORT_INTERVAL_MS
        )
    )
    trace.set_tracer_provider(_provider)

    _tracer = trace.get_tracer("inference")

    # --- Metrics ---
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as GRPCMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as HTTPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    if OTEL_EXPORTER_PROTOCOL == "http":
        metric_exporter = HTTPMetricExporter(
            endpoint=f"http://{OTEL_EXPORTER_ENDPOINT}/v1/metrics",
        )
    else:
        metric_exporter = GRPCMetricExporter(
            endpoint=OTEL_EXPORTER_ENDPOINT,
            insecure=True,
        )

    metric_reader = PeriodicExportingMetricReader(
        metric_exporter, export_interval_millis=OTEL_METRIC_EXPORT_INTERVAL_MS
    )
    _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    otel_metrics.set_meter_provider(_meter_provider)

    meter = _meter_provider.get_meter("inference")
    _metrics = {
        "models_loaded": meter.create_up_down_counter(
            "inference.models.loaded",
            description="Number of models currently loaded",
        ),
        "model_loads": meter.create_counter(
            "inference.model.loads",
            description="Total model loads (cold starts)",
        ),
        "model_unloads": meter.create_counter(
            "inference.model.unloads",
            description="Total model unloads",
        ),
        "model_load_duration": meter.create_histogram(
            "inference.model.load.duration",
            unit="s",
            description="Model load time in seconds",
        ),
        "model_infer_count": meter.create_counter(
            "inference.model.infer.count",
            description="Total inference requests",
        ),
        "model_infer_duration": meter.create_histogram(
            "inference.model.infer.duration",
            unit="s",
            description="Inference latency in seconds",
        ),
        "api_call_duration": meter.create_histogram(
            "inference.roboflow_api.duration",
            unit="s",
            description="Roboflow API call latency in seconds",
        ),
        "errors": meter.create_counter(
            "inference.errors",
            description="Total errors by type",
        ),
    }

    # Replace noisy connection-refused tracebacks with a single-line warning.
    _install_export_error_filter("opentelemetry.sdk.trace.export")
    _install_export_error_filter("opentelemetry.sdk.metrics._internal.export")

    # Auto-instrument FastAPI: creates server spans, extracts traceparent
    FastAPIInstrumentor.instrument_app(app)

    # Add force-trace middleware AFTER the instrumentor so it wraps outermost.
    # Starlette builds middleware last-added = outermost, so this runs BEFORE
    # the instrumentor's ASGI middleware, ensuring the ContextVar is set
    # before should_sample() is called.
    app.add_middleware(_ForceTraceASGIMiddleware)

    logger.info(
        "OpenTelemetry tracing enabled (service=%s, endpoint=%s, protocol=%s, sampling_rate=%s)",
        OTEL_SERVICE_NAME,
        OTEL_EXPORTER_ENDPOINT,
        OTEL_EXPORTER_PROTOCOL,
        OTEL_SAMPLING_RATE,
    )


def shutdown_telemetry() -> None:
    """Flush pending spans/metrics and shut down providers."""
    if _provider is not None and hasattr(_provider, "shutdown"):
        _provider.shutdown()
    if _meter_provider is not None and hasattr(_meter_provider, "shutdown"):
        _meter_provider.shutdown()


def _get_tracer():
    global _tracer
    if _tracer is None and _OTEL_AVAILABLE:
        _tracer = trace.get_tracer("inference")
    return _tracer


class _ExportErrorFilter(logging.Filter):
    """Replace noisy OTel export tracebacks with a single-line warning.

    The OTel SDK logs full connection-refused tracebacks every export cycle
    when the collector is down. This filter catches those ERROR-level records,
    logs a clean warning once, and suppresses duplicates.
    """

    def __init__(self):
        super().__init__()
        self._warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            if not self._warned:
                logger.warning(
                    "OTel exporter cannot reach collector — traces/metrics will be dropped "
                    "until the collector is available."
                )
                self._warned = True
            return False  # suppress the original noisy traceback
        # Reset warning flag when exports succeed again (logged at DEBUG/INFO)
        if self._warned and record.levelno <= logging.INFO:
            self._warned = False
        return True


def _install_export_error_filter(logger_name: str) -> None:
    """Attach the export error filter to an OTel SDK logger."""
    otel_logger = logging.getLogger(logger_name)
    otel_logger.addFilter(_ExportErrorFilter())
