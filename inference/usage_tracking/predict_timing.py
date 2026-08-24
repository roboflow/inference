"""Timing a model's predict phase for usage telemetry.

The number we want is how long the model itself ran, separated from the pre-
and post-processing around it. Measuring that on the host is only correct when
the backend has finished its device work by the time ``predict()`` returns.
Most of them have: the legacy ONNX path binds its outputs to host buffers so
ONNX Runtime cannot return early, the ``inference_models`` ONNX helpers
synchronize their stream before handing tensors back, and TensorRT synchronizes
unless it is explicitly run in pipelined mode. The exceptions are backends that
enqueue CUDA work and return immediately, where a host clock would time the
launch instead of the inference.

CUDA events close that gap without slowing anything down. Recording an event is
just a marker enqueued on the stream, but reading the elapsed time between two
of them requires the second one to have completed - which is why the read
happens after post-processing rather than straight after ``predict()``. By the
time a call is ready to return real numbers to its caller, the device work it
describes must already be done, so the read finds both events complete and
never waits. If they are somehow not complete, the measurement is abandoned in
favour of the host clock instead of blocking; every path here degrades to the
host reading rather than costing latency or raising.

Deferred results are excluded entirely (see :func:`prediction_is_deferred`),
because a pipelined call has no predict phase to attribute in the first place.

Measurements are published through :class:`~contextvars.ContextVar` rather than
attributes on the model. Model instances are shared across the server's worker
threads, so an attribute would let one request overwrite the measurement of
another request that is still running.
"""

from __future__ import annotations

from contextvars import ContextVar
from time import perf_counter
from typing import Any, Optional, Tuple

_measured_predict_duration: ContextVar[Optional[float]] = ContextVar(
    "usage_measured_predict_duration",
    default=None,
)

_UNRESOLVED = object()
_torch_cuda: Any = _UNRESOLVED


def _get_torch_cuda() -> Any:
    """Return ``torch.cuda`` when a usable CUDA runtime is present, else None.

    Resolved once per process. Usage tracking is imported by CPU-only
    deployments that have no torch at all, so the import cannot be top-level.
    """
    global _torch_cuda
    if _torch_cuda is _UNRESOLVED:
        try:
            import torch

            _torch_cuda = torch.cuda if torch.cuda.is_available() else None
        except Exception:
            _torch_cuda = None
    return _torch_cuda


def resolve_inference_stream(model: Any) -> Any:
    """The CUDA stream a model's forward pass enqueues its work onto.

    ``inference_models`` backends expose the stream they run inference on as
    ``_inference_stream``, which is the reliable answer and is checked first,
    on the adapter and on the backend it wraps. Backends that manage no stream
    of their own - the RF-DETR torch path, for instance - enqueue onto the
    ambient current stream instead, so that is used when the model declares a
    CUDA device.

    Returns:
        The stream to record timing events on, or None when the model is not
        running on CUDA or its stream cannot be identified. None means the
        caller should time on the host.
    """
    candidates = (model, getattr(model, "_model", None))
    for candidate in candidates:
        if candidate is None:
            continue
        stream = getattr(candidate, "_inference_stream", None)
        if stream is not None:
            return stream

    torch_cuda = _get_torch_cuda()
    if torch_cuda is None:
        return None
    for candidate in candidates:
        if candidate is None:
            continue
        device = getattr(candidate, "_device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            return torch_cuda.current_stream(device)
    return None


def _create_timing_events(torch_cuda: Any) -> Optional[Tuple[Any, Any]]:
    """A fresh start/end event pair for one predict phase.

    Deliberately not pooled per thread: a model whose ``predict()`` runs
    another model would have the nested call re-record the events the outer
    call is still measuring with. The underlying CUDA event is created lazily
    on first record, so a pair costs a few microseconds against a forward pass
    measured in milliseconds.
    """
    try:
        return (
            torch_cuda.Event(enable_timing=True),
            torch_cuda.Event(enable_timing=True),
        )
    except Exception:
        return None


class PredictPhaseTimer:
    """Times one predict phase, on the device when the work runs there.

    Used across the predict and post-process boundary rather than as a context
    manager, because the device reading is only free once post-processing has
    forced the work to complete:

        timer = PredictPhaseTimer.start(model=self)
        predictions = self.predict(...)
        timer.finish(predictions=predictions)
        result = self.postprocess(predictions, ...)
        timer.publish()
    """

    __slots__ = (
        "_started_at",
        "_host_duration",
        "_events",
        "_stream",
        "_deferred",
    )

    def __init__(self) -> None:
        self._started_at: Optional[float] = None
        self._host_duration: Optional[float] = None
        self._events: Optional[Tuple[Any, Any]] = None
        self._stream: Any = None
        self._deferred = False

    @classmethod
    def start(cls, model: Any) -> "PredictPhaseTimer":
        """Begin timing, arming device events when the model runs on CUDA."""
        timer = cls()
        timer._started_at = perf_counter()
        try:
            stream = resolve_inference_stream(model)
            if stream is None:
                return timer
            torch_cuda = _get_torch_cuda()
            if torch_cuda is None:
                return timer
            events = _create_timing_events(torch_cuda)
            if events is None:
                return timer
            events[0].record(stream)
            timer._events = events
            timer._stream = stream
        except Exception:
            timer._events = None
        return timer

    def finish(self, predictions: Any) -> None:
        """Close the measured window, right after ``predict()`` returned."""
        if self._started_at is None:
            return
        self._host_duration = perf_counter() - self._started_at
        try:
            self._deferred = prediction_is_deferred(predictions)
            if self._events is not None and not self._deferred:
                self._events[1].record(self._stream)
        except Exception:
            self._events = None

    def publish(self) -> None:
        """Record the measurement, preferring the device reading.

        Deferred calls publish nothing, which leaves the bucket falling back to
        the decorator's full call duration.
        """
        if self._host_duration is None or self._deferred:
            return
        duration = self._host_duration
        device_duration = self._read_device_duration()
        if device_duration is not None:
            duration = device_duration
        record_measured_predict_duration(duration)

    def _read_device_duration(self) -> Optional[float]:
        """Seconds measured between the device events, or None to use the host.

        Never waits. An incomplete event means the assumption that
        post-processing forces completion did not hold for this backend, and an
        implausible reading means the events did not bracket the work - a
        stream this module failed to identify. Both fall back to the host.
        """
        if self._events is None:
            return None
        try:
            start_event, end_event = self._events
            if not start_event.query() or not end_event.query():
                return None
            seconds = start_event.elapsed_time(end_event) / 1000.0
            if seconds < 0 or seconds > perf_counter() - self._started_at:
                return None
            return seconds
        except Exception:
            return None


def prediction_is_deferred(predictions: Any) -> bool:
    """Whether ``predict()`` handed back a pending result instead of doing the work.

    The pipelined RF-DETR segmentation adapter returns a future from
    ``predict()`` and resolves it later - during ``postprocess()`` for an
    earlier frame, or after the call has returned entirely when the caller is a
    workflow. Timing ``predict()`` there would measure the launch, not the
    inference, so such a call has no separable predict phase and must fall back
    to the full call duration.

    Duck-typed rather than an ``isinstance`` check against the backend's future,
    so that this module stays import-light and later future-returning adapters
    are covered without a change here.

    Args:
        predictions: Whatever ``predict()`` returned.

    Returns:
        True when the real work has not happened yet.
    """
    predictions_are_deferred = callable(getattr(predictions, "result", None))

    return predictions_are_deferred


def clear_measured_predict_duration() -> None:
    """Drop any predict duration left over from an earlier call."""
    _measured_predict_duration.set(None)


def record_measured_predict_duration(duration: float) -> None:
    """Publish the time spent in a model's predict phase.

    Accumulates, because a single decorated call may run predict more than once
    (a model that chunks its own batch). Negative and non-numeric values are
    dropped rather than raised: usage tracking must never break inference.

    Args:
        duration: Seconds spent inside one predict call.
    """
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        return
    if seconds < 0:
        return

    recorded = _measured_predict_duration.get()
    _measured_predict_duration.set(
        seconds if recorded is None else recorded + seconds,
    )


def consume_measured_predict_duration() -> Optional[float]:
    """Read and clear the predict duration published by the current call.

    Returns:
        Seconds spent in predict, or None when the call had no separable
        predict phase and the caller should fall back to the full call
        duration.
    """
    duration = _measured_predict_duration.get()
    if duration is None:
        return None
    _measured_predict_duration.set(None)

    return duration
