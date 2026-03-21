"""OpenTelemetry tracing setup and helpers for the inference server.

Usage:
    # In HttpInterface.__init__(), before middleware:
    if OTEL_TRACING_ENABLED:
        setup_telemetry(app)

    # In handlers / managers:
    with start_span("model.infer", {"model.id": model_id}):
        result = do_inference(...)

    # Record errors on the current span:
    record_error(error)

When OTEL_TRACING_ENABLED is False, setup_telemetry() is never called and all
helper functions operate as noops via the OTel API's default noop tracer.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagation.composite import CompositePropagator
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation import TraceContextTextMapPropagator

logger = logging.getLogger("inference")

_tracer: Optional[trace.Tracer] = None
_provider = None


def setup_telemetry(app: Any) -> None:
    """Initialize OTel TracerProvider and instrument the FastAPI app.

    Must be called before any middleware is added so the FastAPI instrumentor
    wraps at the outermost ASGI layer.
    """
    global _provider, _tracer

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
        ParentBasedTraceIdRatio,
    )

    from inference.core.env import (
        OTEL_EXPORTER_ENDPOINT,
        OTEL_EXPORTER_PROTOCOL,
        OTEL_SAMPLING_RATE,
        OTEL_SERVICE_NAME,
    )

    # W3C TraceContext propagator — always set so extract/inject are safe
    set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator()])
    )

    # Sampler: honour parent decision, ratio-based for root spans
    if OTEL_SAMPLING_RATE <= 0:
        sampler = ALWAYS_OFF
    elif OTEL_SAMPLING_RATE >= 1.0:
        sampler = ALWAYS_ON
    else:
        sampler = ParentBasedTraceIdRatio(OTEL_SAMPLING_RATE)

    resource = Resource.create({"service.name": OTEL_SERVICE_NAME})

    if OTEL_EXPORTER_PROTOCOL == "http":
        exporter = HTTPExporter(
            endpoint=f"http://{OTEL_EXPORTER_ENDPOINT}/v1/traces",
            insecure=True,
        )
    else:
        exporter = GRPCExporter(
            endpoint=OTEL_EXPORTER_ENDPOINT,
            insecure=True,
        )

    _provider = TracerProvider(resource=resource, sampler=sampler)
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)

    _tracer = trace.get_tracer("inference")

    # Auto-instrument FastAPI: creates server spans, extracts traceparent
    FastAPIInstrumentor.instrument_app(app)

    logger.info(
        "OpenTelemetry tracing enabled (service=%s, endpoint=%s, protocol=%s, sampling_rate=%s)",
        OTEL_SERVICE_NAME,
        OTEL_EXPORTER_ENDPOINT,
        OTEL_EXPORTER_PROTOCOL,
        OTEL_SAMPLING_RATE,
    )


def shutdown_telemetry() -> None:
    """Flush pending spans and shut down the TracerProvider."""
    if _provider is not None and hasattr(_provider, "shutdown"):
        _provider.shutdown()


def _get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("inference")
    return _tracer


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a new span as a child of the current context.

    Usage:
        with start_span("model.infer", {"model.id": "abc/1"}) as span:
            ...
    """
    tracer = _get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def record_error(error: Exception) -> None:
    """Record an exception on the current active span and set ERROR status.

    Safe to call when there is no active span (noop).
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, str(error))


def get_trace_id() -> Optional[str]:
    """Return the current trace ID as a hex string, or None."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        return format(ctx.trace_id, "032x")
    return None
