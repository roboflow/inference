"""Optional OpenTelemetry tracing and metrics for the inference server.

Every public helper is safe to import and call when the ``opentelemetry``
packages are not installed or under ``OFFLINE_MODE``; they degrade to no-ops.
Otherwise they work on whatever span is active, with or without the providers
installed here, like the legacy helpers. ``setup_telemetry(app)``
installs the tracer provider, the OTLP span exporter, the FastAPI and
``requests`` instrumentors, the ``X-Trace-Id`` response middleware and the
``X-Force-Trace`` sampling override from ``OTEL_TRACING_ENABLED`` /
``OTEL_EXPORTER_PROTOCOL`` / ``OTEL_EXPORTER_ENDPOINT`` / ``OTEL_SAMPLING_RATE``,
and the meter provider with its periodic OTLP exporter when
``OTEL_METRICS_ENABLED`` is set. A missing package logs one warning and
leaves the application working without that feature. Install
``inference-server[otel]`` to enable it.
"""

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Sequence

from inference_server import configuration
from inference_server.middlewares.headers import TRACE_ID_HEADER

logger = logging.getLogger(__name__)

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.propagate import inject as otel_inject
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
    _FORCE_TRACE_KEY = otel_context.create_key("inference_server.force_trace")
except ImportError:
    _OTEL_AVAILABLE = False
    _FORCE_TRACE_KEY = None

FORCE_TRACE_HEADER = "x-force-trace"
TRACING_PACKAGES_MISSING_MESSAGE = (
    "OTEL_TRACING_ENABLED is set but the opentelemetry packages are not "
    "installed or incomplete (%s); tracing is disabled. Install "
    "'inference-server[otel]' to enable it."
)
METRICS_PACKAGES_MISSING_MESSAGE = (
    "OTEL_METRICS_ENABLED is set but the opentelemetry metrics packages are not "
    "installed or incomplete (%s); metrics are disabled. Install "
    "'inference-server[otel]' to enable them."
)

_TRACE_ID_HEADER_NAME = TRACE_ID_HEADER.lower().encode("latin-1")

_tracer = None
_provider = None
_meter_provider = None
_metrics: Optional[Dict[str, Any]] = None


def _otel_enabled() -> bool:
    return _OTEL_AVAILABLE and not configuration.OFFLINE_MODE


def _get_tracer() -> Any:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("inference_server")
    return _tracer


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Start a span as a child of the current context.

    Args:
        name: Span name.
        attributes: Attributes set on the span at creation.

    Yields:
        The span, or None when OpenTelemetry is unavailable or disabled.
    """
    if not _otel_enabled():
        yield None
        return

    tracer = _get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def record_error(error: Exception) -> None:
    """Record an exception on the current span and set its status to ERROR.

    Args:
        error: Exception to record; ignored without a recording span.
    """
    if not _otel_enabled():
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, str(error))


def trace_context_fields() -> Dict[str, str]:
    """Return the current trace and span ids as log fields.

    Returns:
        ``{"trace_id": ..., "span_id": ...}`` as hex strings when a span is
        active, an empty dict otherwise.
    """
    if not _otel_enabled():
        return {}

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        fields = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
        return fields
    return {}


def get_trace_id() -> Optional[str]:
    """Return the current span's trace id as a hex string, or None.

    Returns:
        The active trace id, or None when OpenTelemetry is unavailable,
        disabled, or no span is active.
    """
    trace_id = trace_context_fields().get("trace_id")

    return trace_id


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current span.

    Args:
        key: Attribute name.
        value: Attribute value; ignored without a recording span.
    """
    if not _otel_enabled():
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def capture_context() -> Any:
    """Capture the current context for propagation to another thread.

    Returns:
        An opaque token for ``attach_context``, or None without OpenTelemetry.
    """
    if not _otel_enabled():
        return None

    captured = otel_context.get_current()

    return captured


def attach_context(ctx: Any) -> Any:
    """Attach a context captured by ``capture_context`` in the current thread.

    Args:
        ctx: Token returned by ``capture_context``.

    Returns:
        A token for ``detach_context``, or None when nothing was attached.
    """
    if ctx is None or not _otel_enabled():
        return None

    token = otel_context.attach(ctx)

    return token


def detach_context(token: Any) -> None:
    """Detach a context attached by ``attach_context``.

    Args:
        token: Token returned by ``attach_context``; None is ignored.
    """
    if token is None or not _otel_enabled():
        return

    otel_context.detach(token)


def inject_trace_context(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Inject the W3C ``traceparent`` / ``tracestate`` headers into ``headers``.

    Args:
        headers: Outgoing request headers; None becomes an empty dict.

    Returns:
        The same headers, unchanged without OpenTelemetry.
    """
    if headers is None:
        headers = {}
    if not _otel_enabled():
        return headers

    otel_inject(headers)

    return headers


def record_model_loaded(model_id: str, load_time: float) -> None:
    """Record a model load: loaded gauge, load counter and load duration.

    Args:
        model_id: Loaded model.
        load_time: Load duration in seconds.
    """
    if _metrics is None:
        return

    _metrics["models_loaded"].add(1)
    _metrics["model_loads"].add(1, {"model.id": model_id})
    _metrics["model_load_duration"].record(load_time, {"model.id": model_id})


def record_model_unloaded(model_id: str) -> None:
    """Record a model unload.

    Args:
        model_id: Unloaded model.
    """
    if _metrics is None:
        return

    _metrics["models_loaded"].add(-1)
    _metrics["model_unloads"].add(1, {"model.id": model_id})


def record_inference(model_id: str, duration: float) -> None:
    """Record an inference execution.

    Args:
        model_id: Model that ran.
        duration: Inference duration in seconds.
    """
    if _metrics is None:
        return

    _metrics["model_infer_count"].add(1, {"model.id": model_id})
    _metrics["model_infer_duration"].record(duration, {"model.id": model_id})


def record_api_call(function_name: str, duration: float) -> None:
    """Record a Roboflow API call duration.

    Args:
        function_name: Name of the API function called.
        duration: Call duration in seconds.
    """
    if _metrics is None:
        return

    _metrics["api_call_duration"].record(
        duration, {"roboflow_api.function": function_name}
    )


def record_error_metric(error_type: str) -> None:
    """Increment the error counter for an error type.

    Args:
        error_type: Error class name or category.
    """
    if _metrics is None:
        return

    _metrics["errors"].add(1, {"error.type": error_type})


class TraceIdResponseMiddleware:
    """Raw ASGI middleware stamping ``X-Trace-Id`` when a span is active.

    A raw ASGI middleware, not ``BaseHTTPMiddleware``: Starlette's
    ``BaseHTTPMiddleware`` runs the inner chain in a separate task, which
    would hide the active span from ``get_trace_id()``. Added after every
    other application middleware and before the FastAPI instrumentor, so it
    stamps early responses of the inner middlewares while the server span is
    active.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def _send(message: dict) -> None:
            if message["type"] == "http.response.start":
                trace_id = get_trace_id()
                if trace_id:
                    headers = list(message.get("headers", []))
                    headers.append((_TRACE_ID_HEADER_NAME, trace_id.encode("latin-1")))
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, _send)


class _ForceTracePropagator:
    """Text-map propagator reading ``X-Force-Trace: true`` into the context.

    Registered next to the W3C propagator, so the FastAPI instrumentor's
    header extraction, which runs before the server span is sampled, carries
    the flag in the parent context the root sampler receives. Injects
    nothing.
    """

    def extract(self, carrier: Any, context: Any = None, getter: Any = None) -> Any:
        if context is None:
            context = otel_context.Context()
        if getter is None:
            values = carrier.get(FORCE_TRACE_HEADER)
        else:
            values = getter.get(carrier, FORCE_TRACE_HEADER)
        if isinstance(values, str):
            values = [values]
        if not values or values[0].strip().lower() != "true":
            return context

        forced = otel_context.set_value(_FORCE_TRACE_KEY, True, context)

        return forced

    def inject(self, carrier: Any, context: Any = None, setter: Any = None) -> None:
        return None

    @property
    def fields(self) -> set:
        return set()


class _ForceTraceRootSampler:
    """Root sampler that samples when the parent context carries the flag.

    Used as the ``root`` of ``ParentBased``, so a parent decision is honoured
    by ``ParentBased`` itself; a root span is sampled when
    ``_ForceTracePropagator`` set the flag and delegated otherwise.
    """

    def __init__(self, delegate: Any, sampling: Any) -> None:
        self._delegate = delegate
        self._sampling = sampling

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
        if otel_context.get_value(_FORCE_TRACE_KEY, context=parent_context):
            forced = self._sampling.SamplingResult(
                decision=self._sampling.Decision.RECORD_AND_SAMPLE,
                attributes={"sampling.forced": True},
                trace_state=trace_state,
            )
            return forced

        delegated = self._delegate.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

        return delegated

    def get_description(self) -> str:
        return f"ForceTraceRootSampler({self._delegate.get_description()})"


class _ExportErrorFilter(logging.Filter):
    """Replace the SDK's export tracebacks with one warning until exports recover."""

    def __init__(self) -> None:
        super().__init__()
        self._warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            if not self._warned:
                logger.warning(
                    "OTel exporter cannot reach the collector; traces/metrics are "
                    "dropped until it is available."
                )
                self._warned = True
            return False
        if self._warned and record.levelno <= logging.INFO:
            self._warned = False
        return True


def _install_export_error_filter(logger_name: str) -> None:
    logging.getLogger(logger_name).addFilter(_ExportErrorFilter())


def _import_tracing_dependencies(protocol: str) -> Optional[SimpleNamespace]:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositePropagator
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider, sampling
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        if protocol == "http":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
    except ImportError as error:
        logger.warning(TRACING_PACKAGES_MISSING_MESSAGE, error)
        return None

    dependencies = SimpleNamespace(
        FastAPIInstrumentor=FastAPIInstrumentor,
        RequestsInstrumentor=RequestsInstrumentor,
        set_global_textmap=set_global_textmap,
        CompositePropagator=CompositePropagator,
        TraceContextTextMapPropagator=TraceContextTextMapPropagator,
        Resource=Resource,
        TracerProvider=TracerProvider,
        sampling=sampling,
        BatchSpanProcessor=BatchSpanProcessor,
        OTLPSpanExporter=OTLPSpanExporter,
    )

    return dependencies


def _import_metrics_dependencies(protocol: str) -> Optional[SimpleNamespace]:
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        if protocol == "http":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
    except ImportError as error:
        logger.warning(METRICS_PACKAGES_MISSING_MESSAGE, error)
        return None

    dependencies = SimpleNamespace(
        otel_metrics=otel_metrics,
        MeterProvider=MeterProvider,
        PeriodicExportingMetricReader=PeriodicExportingMetricReader,
        OTLPMetricExporter=OTLPMetricExporter,
    )

    return dependencies


def _build_root_sampler(sampling: Any) -> Any:
    if configuration.OTEL_SAMPLING_RATE <= 0:
        return sampling.ALWAYS_OFF
    if configuration.OTEL_SAMPLING_RATE >= 1.0:
        return sampling.ALWAYS_ON

    ratio_sampler = sampling.TraceIdRatioBased(configuration.OTEL_SAMPLING_RATE)

    return ratio_sampler


def _build_span_exporter(dependencies: SimpleNamespace, protocol: str) -> Any:
    if protocol == "http":
        exporter = dependencies.OTLPSpanExporter(
            endpoint=f"http://{configuration.OTEL_EXPORTER_ENDPOINT}/v1/traces",
        )
    else:
        exporter = dependencies.OTLPSpanExporter(
            endpoint=configuration.OTEL_EXPORTER_ENDPOINT,
            insecure=True,
        )

    return exporter


def _install_tracer_provider(
    dependencies: SimpleNamespace, resource: Any, protocol: str
) -> None:
    global _provider

    dependencies.set_global_textmap(
        dependencies.CompositePropagator(
            [dependencies.TraceContextTextMapPropagator(), _ForceTracePropagator()]
        )
    )
    root_sampler = _build_root_sampler(dependencies.sampling)
    sampler = dependencies.sampling.ParentBased(
        root=_ForceTraceRootSampler(root_sampler, dependencies.sampling)
    )
    exporter = _build_span_exporter(dependencies, protocol)

    _provider = dependencies.TracerProvider(resource=resource, sampler=sampler)
    _provider.add_span_processor(
        dependencies.BatchSpanProcessor(
            exporter,
            schedule_delay_millis=configuration.OTEL_TRACE_EXPORT_INTERVAL_MS,
        )
    )
    trace.set_tracer_provider(_provider)
    _install_export_error_filter("opentelemetry.sdk.trace.export")


def _install_meter_provider(
    dependencies: SimpleNamespace, resource: Any, protocol: str
) -> None:
    global _meter_provider, _metrics

    metric_endpoint = (
        configuration.OTEL_METRIC_EXPORTER_ENDPOINT
        or configuration.OTEL_EXPORTER_ENDPOINT
    )
    if protocol == "http":
        metric_exporter = dependencies.OTLPMetricExporter(
            endpoint=f"http://{metric_endpoint}/v1/metrics",
        )
    else:
        metric_exporter = dependencies.OTLPMetricExporter(
            endpoint=metric_endpoint,
            insecure=True,
        )
    metric_reader = dependencies.PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=configuration.OTEL_METRIC_EXPORT_INTERVAL_MS,
    )

    _meter_provider = dependencies.MeterProvider(
        resource=resource, metric_readers=[metric_reader]
    )
    dependencies.otel_metrics.set_meter_provider(_meter_provider)
    meter = _meter_provider.get_meter("inference_server")
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
    _install_export_error_filter("opentelemetry.sdk.metrics._internal.export")


def setup_telemetry(app: Any) -> None:
    """Install the providers, exporters and instrumentation on the app.

    No-op when ``OTEL_TRACING_ENABLED`` is False. Logs one warning and stays a
    no-op when a required ``opentelemetry`` package is missing; a missing
    metrics package only disables metrics. Must be called after every other
    ``app.add_middleware`` call so the trace-id middleware wraps them all;
    the FastAPI instrumentor wraps the whole stack. The process-wide
    providers, propagators and the ``requests`` instrumentation are installed
    once; a later call instruments the new app with the same providers.

    Args:
        app: The FastAPI application to instrument.
    """
    if not configuration.OTEL_TRACING_ENABLED:
        return
    if not _OTEL_AVAILABLE:
        logger.warning(TRACING_PACKAGES_MISSING_MESSAGE, "opentelemetry")
        return

    protocol = configuration.OTEL_EXPORTER_PROTOCOL
    dependencies = _import_tracing_dependencies(protocol)
    if dependencies is None:
        return

    if _provider is None:
        resource = dependencies.Resource.create(
            {
                "service.name": configuration.OTEL_SERVICE_NAME,
                "service.instance.id": configuration.INFERENCE_SERVER_ID
                or configuration.SERVER_ID,
            }
        )
        _install_tracer_provider(dependencies, resource, protocol)
        if configuration.OTEL_METRICS_ENABLED:
            metrics_dependencies = _import_metrics_dependencies(protocol)
            if metrics_dependencies is not None:
                _install_meter_provider(metrics_dependencies, resource, protocol)
        dependencies.RequestsInstrumentor().instrument()

    app.add_middleware(TraceIdResponseMiddleware)
    dependencies.FastAPIInstrumentor.instrument_app(app)

    logger.info(
        "OpenTelemetry tracing enabled (service=%s, endpoint=%s, protocol=%s, "
        "sampling_rate=%s, metrics=%s)",
        configuration.OTEL_SERVICE_NAME,
        configuration.OTEL_EXPORTER_ENDPOINT,
        protocol,
        configuration.OTEL_SAMPLING_RATE,
        _metrics is not None,
    )


def shutdown_telemetry() -> None:
    """Flush pending spans and metrics and shut the providers down."""
    if _provider is not None and hasattr(_provider, "shutdown"):
        _provider.shutdown()
    if _meter_provider is not None and hasattr(_meter_provider, "shutdown"):
        _meter_provider.shutdown()
