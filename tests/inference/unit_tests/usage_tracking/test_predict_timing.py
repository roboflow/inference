import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference.usage_tracking import predict_timing
from inference.usage_tracking.decorator_helpers import get_model_megapixel_buckets
from inference.usage_tracking.megapixel_buckets import clear_measured_model_input
from inference.usage_tracking.predict_timing import (
    PredictPhaseTimer,
    clear_measured_predict_duration,
    consume_measured_predict_duration,
    prediction_is_deferred,
    record_measured_predict_duration,
    resolve_inference_stream,
)

_STREAM = object()


class _FakeEvent:
    """Stands in for a CUDA event, recording how the timer drives it."""

    def __init__(self, elapsed_ms: float = 5.0, complete: bool = True):
        self.elapsed_ms = elapsed_ms
        self.complete = complete
        self.recorded_on = []
        self.synchronize_calls = 0

    def record(self, stream):
        self.recorded_on.append(stream)

    def query(self):
        return self.complete

    def synchronize(self):
        self.synchronize_calls += 1

    def elapsed_time(self, other):
        return self.elapsed_ms


def _install_fake_cuda(monkeypatch, events):
    pending = list(events)
    monkeypatch.setattr(
        predict_timing,
        "_torch_cuda",
        SimpleNamespace(
            Event=lambda enable_timing=False: pending.pop(0),
            current_stream=lambda device: _STREAM,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_measurements():
    clear_measured_predict_duration()
    yield
    clear_measured_predict_duration()


def test_record_measured_predict_duration_accumulates_across_calls():
    # A model that chunks its own batch runs predict more than once per call.
    record_measured_predict_duration(0.2)
    record_measured_predict_duration(0.3)

    assert consume_measured_predict_duration() == pytest.approx(0.5)


def test_consume_measured_predict_duration_clears_value():
    record_measured_predict_duration(0.2)

    assert consume_measured_predict_duration() == pytest.approx(0.2)
    # A later call with no separable predict phase must not inherit this one.
    assert consume_measured_predict_duration() is None


def test_record_measured_predict_duration_ignores_unusable_values():
    record_measured_predict_duration(-1.0)
    record_measured_predict_duration("not-a-duration")
    record_measured_predict_duration(None)

    assert consume_measured_predict_duration() is None


def test_clear_measured_model_input_also_clears_predict_duration():
    record_measured_predict_duration(0.2)

    clear_measured_model_input()

    assert consume_measured_predict_duration() is None


def test_prediction_is_deferred_detects_future_like_results():
    assert prediction_is_deferred(SimpleNamespace(result=lambda: [1]))
    assert not prediction_is_deferred((np.zeros(1),))
    assert not prediction_is_deferred(np.zeros(1))
    assert not prediction_is_deferred({"logits": np.zeros(1)})
    # An attribute that merely shares the name is not a pending result.
    assert not prediction_is_deferred(SimpleNamespace(result=[1]))


def test_resolve_inference_stream_prefers_the_backend_stream(monkeypatch):
    _install_fake_cuda(monkeypatch, [])

    assert (
        resolve_inference_stream(SimpleNamespace(_inference_stream=_STREAM)) is _STREAM
    )
    # Adapters wrap the backend that owns the stream.
    adapter = SimpleNamespace(_model=SimpleNamespace(_inference_stream=_STREAM))
    assert resolve_inference_stream(adapter) is _STREAM


def test_resolve_inference_stream_uses_current_stream_for_unmanaged_backends(
    monkeypatch,
):
    # Backends like the RF-DETR torch path enqueue onto the ambient stream.
    _install_fake_cuda(monkeypatch, [])
    model = SimpleNamespace(_device=SimpleNamespace(type="cuda"))

    assert resolve_inference_stream(model) is _STREAM


def test_resolve_inference_stream_returns_none_off_cuda(monkeypatch):
    monkeypatch.setattr(predict_timing, "_torch_cuda", None)

    assert resolve_inference_stream(SimpleNamespace()) is None
    assert (
        resolve_inference_stream(SimpleNamespace(_device=SimpleNamespace(type="cpu")))
        is None
    )
    # A backend on CPU reports no stream of its own.
    assert resolve_inference_stream(SimpleNamespace(_inference_stream=None)) is None


def test_timer_uses_host_clock_when_model_is_not_on_cuda(monkeypatch):
    monkeypatch.setattr(predict_timing, "_torch_cuda", None)

    timer = PredictPhaseTimer.start(model=SimpleNamespace())
    time.sleep(0.01)
    timer.finish(predictions=(np.zeros(1),))
    timer.publish()

    assert consume_measured_predict_duration() >= 0.01


def test_timer_prefers_the_device_measurement_over_the_host_clock(monkeypatch):
    start_event, end_event = _FakeEvent(elapsed_ms=5.0), _FakeEvent(elapsed_ms=5.0)
    _install_fake_cuda(monkeypatch, [start_event, end_event])
    model = SimpleNamespace(_inference_stream=_STREAM)

    timer = PredictPhaseTimer.start(model=model)
    # The host clock sees far more than the device spent on the work.
    time.sleep(0.05)
    timer.finish(predictions=(np.zeros(1),))
    timer.publish()

    assert consume_measured_predict_duration() == pytest.approx(0.005)
    assert start_event.recorded_on == [_STREAM]
    assert end_event.recorded_on == [_STREAM]


def test_nested_timers_do_not_share_events(monkeypatch):
    # A model whose predict() runs another model nests one timer inside another.
    outer_start, outer_end = _FakeEvent(elapsed_ms=8.0), _FakeEvent(elapsed_ms=8.0)
    inner_start, inner_end = _FakeEvent(elapsed_ms=3.0), _FakeEvent(elapsed_ms=3.0)
    _install_fake_cuda(monkeypatch, [outer_start, outer_end, inner_start, inner_end])
    model = SimpleNamespace(_inference_stream=_STREAM)

    outer = PredictPhaseTimer.start(model=model)
    inner = PredictPhaseTimer.start(model=model)
    time.sleep(0.02)
    inner.finish(predictions=(np.zeros(1),))
    inner.publish()
    outer.finish(predictions=(np.zeros(1),))
    outer.publish()

    # The nested call must not have closed the outer call's window early.
    assert outer_end.recorded_on == [_STREAM]
    assert consume_measured_predict_duration() == pytest.approx(0.011)


def test_timer_never_waits_on_the_device(monkeypatch):
    start_event, end_event = _FakeEvent(), _FakeEvent()
    _install_fake_cuda(monkeypatch, [start_event, end_event])

    timer = PredictPhaseTimer.start(model=SimpleNamespace(_inference_stream=_STREAM))
    timer.finish(predictions=(np.zeros(1),))
    timer.publish()

    # Blocking on the event would put a device sync on the request path.
    assert start_event.synchronize_calls == 0
    assert end_event.synchronize_calls == 0


def test_timer_falls_back_to_host_clock_when_events_are_incomplete(monkeypatch):
    _install_fake_cuda(
        monkeypatch, [_FakeEvent(complete=False), _FakeEvent(complete=False)]
    )

    timer = PredictPhaseTimer.start(model=SimpleNamespace(_inference_stream=_STREAM))
    time.sleep(0.01)
    timer.finish(predictions=(np.zeros(1),))
    timer.publish()

    assert consume_measured_predict_duration() >= 0.01


def test_timer_rejects_a_device_reading_longer_than_the_call(monkeypatch):
    # Events that did not bracket the work can report an unrelated span.
    _install_fake_cuda(
        monkeypatch, [_FakeEvent(elapsed_ms=90_000.0), _FakeEvent(elapsed_ms=90_000.0)]
    )

    timer = PredictPhaseTimer.start(model=SimpleNamespace(_inference_stream=_STREAM))
    time.sleep(0.01)
    timer.finish(predictions=(np.zeros(1),))
    timer.publish()

    assert consume_measured_predict_duration() < 1.0


def test_timer_falls_back_to_host_clock_when_the_backend_raises(monkeypatch):
    exploding = _FakeEvent()
    exploding.record = lambda stream: (_ for _ in ()).throw(RuntimeError("no device"))
    _install_fake_cuda(monkeypatch, [exploding, _FakeEvent()])

    timer = PredictPhaseTimer.start(model=SimpleNamespace(_inference_stream=_STREAM))
    time.sleep(0.01)
    timer.finish(predictions=(np.zeros(1),))
    timer.publish()

    assert consume_measured_predict_duration() >= 0.01


def test_timer_publishes_nothing_for_deferred_work(monkeypatch):
    start_event, end_event = _FakeEvent(), _FakeEvent()
    _install_fake_cuda(monkeypatch, [start_event, end_event])

    timer = PredictPhaseTimer.start(model=SimpleNamespace(_inference_stream=_STREAM))
    timer.finish(predictions=SimpleNamespace(result=lambda: (np.zeros(1),)))
    timer.publish()

    assert consume_measured_predict_duration() is None
    # Closing the window around a launch would have measured nothing useful.
    assert end_event.recorded_on == []


def test_base_inference_bucket_duration_excludes_pre_and_postprocessing():
    from inference.core.models.base import BaseInference

    class SlowProcessingModel(BaseInference):
        def preprocess(self, image, **kwargs):
            time.sleep(0.05)
            return np.zeros((1, 3, 640, 640), dtype=np.float32), None

        def predict(self, img_in, **kwargs):
            time.sleep(0.01)
            return (np.zeros(1),)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            time.sleep(0.05)
            return predictions

    started_at = time.perf_counter()
    BaseInference.infer.__wrapped__(SlowProcessingModel(), object())
    call_duration = time.perf_counter() - started_at

    buckets = get_model_megapixel_buckets(
        frames=1,
        input_hw=(640, 640),
        execution_duration=call_duration,
    )

    bucket_duration = buckets["0.25-0.5"]["execution_duration"]
    assert bucket_duration >= 0.01
    assert bucket_duration < 0.05
    assert call_duration >= 0.11


def test_base_inference_skips_predict_timing_when_work_is_deferred():
    from inference.core.models.base import BaseInference

    class PipelinedModel(BaseInference):
        """Mirrors the RF-DETR stream pipeline: predict launches, postprocess resolves."""

        def preprocess(self, image, **kwargs):
            return np.zeros((1, 3, 640, 640), dtype=np.float32), None

        def predict(self, img_in, **kwargs):
            return SimpleNamespace(result=lambda: (np.zeros(1),))

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            time.sleep(0.05)
            return predictions.result()

    BaseInference.infer.__wrapped__(PipelinedModel(), object())

    # Timing the launch would have reported a near-zero predict phase, so the
    # call must publish nothing and fall back to the full duration instead.
    assert consume_measured_predict_duration() is None

    buckets = get_model_megapixel_buckets(
        frames=1,
        input_hw=(640, 640),
        execution_duration=1.5,
    )
    assert buckets["0.25-0.5"]["execution_duration"] == pytest.approx(1.5)
