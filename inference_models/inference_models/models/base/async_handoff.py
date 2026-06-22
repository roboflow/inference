"""Explicit handoff state shared by async inference pipeline components.

The RF-DETR stream pipeline crosses package boundaries: inference-models owns
CUDA execution and sparse postprocess, while inference owns workflow response
assembly. This module centralizes the small amount of per-request state passed
between those layers so call sites do not depend on scattered private attribute
names.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

_ADAPTER_CONTEXT_ATTR = "_inference_adapter_future_context"
_ASYNC_RESPONSE_FUTURE_ATTR = "_async_response_future"
_ASYNC_RESPONSE_CONTEXT_ID_ATTR = "_async_response_context_id"
_DEFERRED_POSTPROCESS_ATTR = "_inference_deferred_postprocess_handoff"


STREAM_PIPELINE_CONTEXT_ID_KWARG = "stream_pipeline_context_id"


@dataclass(frozen=True)
class AdapterFutureContext:
    """State the inference adapter keeps on an in-flight model future."""

    mapped_kwargs: dict
    stream_pipeline_context_id: Optional[str] = None
    gpu_work_submitted: bool = False
    gpu_submit_generation: Optional[int] = None


@dataclass(frozen=True)
class DeferredPostprocessHandoff:
    """CUDA/postprocess state consumed later by response finalization."""

    done_event: Any
    trt_outputs_consumed_event: Any
    finalize: Callable[[], Any]


def attach_adapter_mapped_kwargs(
    future: Any,
    mapped_kwargs: dict,
    stream_pipeline_context_id: Optional[str] = None,
) -> None:
    """Store mapped inference kwargs on a model future."""
    setattr(
        future,
        _ADAPTER_CONTEXT_ATTR,
        AdapterFutureContext(
            mapped_kwargs=dict(mapped_kwargs),
            stream_pipeline_context_id=stream_pipeline_context_id,
        ),
    )


def get_adapter_mapped_kwargs(future: Any) -> dict:
    """Return mapped inference kwargs stored on a model future."""
    context = getattr(future, _ADAPTER_CONTEXT_ATTR, None)
    if not isinstance(context, AdapterFutureContext):
        return {}
    return context.mapped_kwargs


def get_adapter_stream_pipeline_context_id(future: Any) -> Optional[str]:
    """Return the stream-pipeline context id stored on a model future."""
    context = getattr(future, _ADAPTER_CONTEXT_ATTR, None)
    if not isinstance(context, AdapterFutureContext):
        return None
    context_id = context.stream_pipeline_context_id
    if isinstance(context_id, str):
        return context_id
    return None


def adapter_gpu_work_submitted(future: Any) -> bool:
    """Return whether eager GPU postprocess was submitted for a future."""
    context = getattr(future, _ADAPTER_CONTEXT_ATTR, None)
    return isinstance(context, AdapterFutureContext) and context.gpu_work_submitted


def mark_adapter_gpu_work_submitted(future: Any, generation: int) -> None:
    """Mark a future's eager GPU postprocess submission generation."""
    context = getattr(future, _ADAPTER_CONTEXT_ATTR, None)
    if not isinstance(context, AdapterFutureContext):
        context = AdapterFutureContext(mapped_kwargs={})
    setattr(
        future,
        _ADAPTER_CONTEXT_ATTR,
        replace(
            context,
            gpu_work_submitted=True,
            gpu_submit_generation=generation,
        ),
    )


def get_adapter_gpu_submit_generation(future: Any) -> Optional[int]:
    """Return the generation when eager GPU postprocess was submitted."""
    context = getattr(future, _ADAPTER_CONTEXT_ATTR, None)
    if not isinstance(context, AdapterFutureContext):
        return None
    return context.gpu_submit_generation


def attach_async_response_future(
    response: Any,
    response_future: Any,
    context_id: Optional[str] = None,
) -> None:
    """Attach a CPU response future to a placeholder workflow response."""
    setattr(response, _ASYNC_RESPONSE_FUTURE_ATTR, response_future)
    if context_id is not None:
        setattr(response, _ASYNC_RESPONSE_CONTEXT_ID_ATTR, context_id)


def get_async_response_future(response: Any) -> Any:
    """Return the CPU response future attached to a workflow response."""
    return getattr(response, _ASYNC_RESPONSE_FUTURE_ATTR, None)


def get_async_response_context_id(response: Any) -> Optional[str]:
    """Return the stream context id attached to a placeholder response."""
    context_id = getattr(response, _ASYNC_RESPONSE_CONTEXT_ID_ATTR, None)
    if isinstance(context_id, str):
        return context_id
    return None


def attach_deferred_postprocess_handoff(
    detections: Any,
    done_event: Any,
    trt_outputs_consumed_event: Any,
    finalize: Callable[[], Any],
) -> None:
    """Attach deferred sparse postprocess completion state to detections."""
    setattr(
        detections,
        _DEFERRED_POSTPROCESS_ATTR,
        DeferredPostprocessHandoff(
            done_event=done_event,
            trt_outputs_consumed_event=trt_outputs_consumed_event,
            finalize=finalize,
        ),
    )


def get_deferred_postprocess_handoff(
    detections: Any,
) -> Optional[DeferredPostprocessHandoff]:
    """Return deferred postprocess state attached to detections, if present."""
    handoff = getattr(detections, _DEFERRED_POSTPROCESS_ATTR, None)
    if not isinstance(handoff, DeferredPostprocessHandoff):
        return None
    return handoff


def get_deferred_postprocess_finalizer(detections: Any) -> Optional[Callable[[], Any]]:
    """Return a callable that materializes deferred detections, if present."""
    handoff = get_deferred_postprocess_handoff(detections)
    if handoff is None:
        return None
    return handoff.finalize


def get_deferred_postprocess_done_event(detections: Any) -> Any:
    """Return the CUDA event recorded after deferred postprocess DtoH copies."""
    handoff = get_deferred_postprocess_handoff(detections)
    if handoff is None:
        return None
    return handoff.done_event


def get_trt_outputs_consumed_event(detections: Any) -> Any:
    """Return the event proving TensorRT graph outputs are safe to reuse."""
    handoff = get_deferred_postprocess_handoff(detections)
    if handoff is None:
        return None
    return handoff.trt_outputs_consumed_event
