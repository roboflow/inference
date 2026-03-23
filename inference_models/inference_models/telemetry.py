"""OpenTelemetry instrumentation for inference-models.

Depends only on opentelemetry-api (not the SDK). When no SDK is configured
by the application, all operations are noops with zero overhead.

Library users get instrumentation for free — the application configures
the SDK/exporters, the library just creates spans and records metrics.
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger("inference_models")

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is None and _OTEL_AVAILABLE:
        _tracer = trace.get_tracer("inference_models")
    return _tracer


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a new span as a child of the current context."""
    if not _OTEL_AVAILABLE:
        yield None
        return
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current active span."""
    if not _OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def record_error(error: Exception) -> None:
    """Record an exception on the current active span."""
    if not _OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, str(error))


# ---------------------------------------------------------------------------
# Model auto-instrumentation
# ---------------------------------------------------------------------------

_INSTRUMENTABLE_METHODS = ["infer", "pre_process", "forward", "post_process"]


def traced_from_pretrained(original_method):
    """Decorator for AutoModel.from_pretrained() that adds a span and instruments the result.

    Wraps the entire from_pretrained call in an inference_models.from_pretrained span,
    then auto-instruments the returned model's inference methods.
    """
    if not _OTEL_AVAILABLE:
        return original_method

    @functools.wraps(original_method)
    def wrapper(cls, model_id_or_path, *args, **kwargs):
        tracer = _get_tracer()
        if tracer is None:
            return original_method(cls, model_id_or_path, *args, **kwargs)
        with tracer.start_as_current_span(
            "inference_models.from_pretrained",
            attributes={"model.id": str(model_id_or_path)},
        ):
            model = original_method(cls, model_id_or_path, *args, **kwargs)
            return instrument_model(model)

    return wrapper


def instrument_model(model: Any) -> Any:
    """Wrap model inference methods with OTel spans.

    Called by AutoModel.from_pretrained() before returning the model.
    Model authors never need to add OTel code — this patches known methods
    automatically. Noop when opentelemetry-api is not installed.
    """
    if not _OTEL_AVAILABLE:
        return model
    tracer = _get_tracer()
    if tracer is None:
        return model
    for method_name in _INSTRUMENTABLE_METHODS:
        original = getattr(model, method_name, None)
        if original is None or not callable(original):
            continue
        span_name = f"inference_models.{method_name}"

        @functools.wraps(original)
        def traced_method(
            *args, _original=original, _span_name=span_name, **kwargs
        ):
            with tracer.start_as_current_span(_span_name):
                return _original(*args, **kwargs)

        setattr(model, method_name, traced_method)
    return model
