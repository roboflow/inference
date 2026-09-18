from datetime import datetime
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
)
from inference.core.interfaces.camera.video_source import (
    FRAME_CAPTURED_EVENT,
    StreamState,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    InitialisePipelinePayload,
)
from inference.core.interfaces.stream_manager.manager_app.file_jobs import (
    FileJobResults,
)


def frame(n):
    return VideoFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        frame_id=n,
        frame_timestamp=datetime.now(),
        source_id=0,
    )


def captured(sink, n):
    sink.on_status_update(
        StatusUpdate(
            timestamp=datetime.now(),
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_CAPTURED_EVENT,
            payload={"frame_id": n},
            context="video_source",
        )
    )


def finish(sink, declared):
    sink._video_sources = [
        SimpleNamespace(
            describe_source=lambda: SimpleNamespace(
                state=StreamState.ENDED,
                source_properties=SimpleNamespace(is_file=True, total_frames=declared),
            )
        )
    ]
    sink.wait_for_completion(SimpleNamespace(join=lambda: None))


def test_slow_consumer_gets_all_results_with_bounded_memory():
    sink = FileJobResults(queue_size=1, frame_stride=5)
    captured(sink, 11)
    sink.on_prediction({}, frame(1))
    done = Event()
    thread = Thread(target=lambda: (sink.on_prediction({}, frame(6)), done.set()))
    thread.start()
    assert not done.wait(0.05)
    assert sink.snapshot()["pending_results"] == 1
    assert sink.consume_prediction()[1][0].frame_id == 1
    assert done.wait(1)
    assert sink.consume_prediction()[1][0].frame_id == 6
    sink.on_prediction({}, frame(11))
    sink.consume_prediction()
    finish(sink, 11)
    assert sink.snapshot() == dict(
        state="completed",
        source_frames=11,
        frame_stride=5,
        results_produced=3,
        results_consumed=3,
        pending_results=0,
        error=None,
    )
    thread.join()


def test_cancel_unblocks_full_queue():
    sink = FileJobResults(queue_size=1, frame_stride=1)
    sink.on_prediction({}, frame(1))
    thread = Thread(target=lambda: sink.on_prediction({}, frame(2)))
    thread.start()
    sink.cancel()
    thread.join(timeout=1)
    assert not thread.is_alive()
    finish(sink, 2)
    assert sink.snapshot()["state"] == "cancelled"


@pytest.mark.parametrize(
    "captured_count,produced_count,declared", [(3, 2, 3), (2, 2, 3), (0, 0, 0)]
)
def test_missing_tail_or_decode_truncation_cannot_succeed(
    captured_count, produced_count, declared
):
    sink = FileJobResults(queue_size=4, frame_stride=1)
    captured(sink, captured_count)
    for n in range(1, produced_count + 1):
        sink.on_prediction({}, frame(n))
    finish(sink, declared)
    assert sink.snapshot()["state"] == "failed"


def test_error_remains_latched_after_watchdog_history_rolls_over():
    sink = FileJobResults(queue_size=1, frame_stride=1)
    sink.on_status_update(
        StatusUpdate(
            timestamp=datetime.now(),
            severity=UpdateSeverity.ERROR,
            event_type="INFERENCE_ERROR",
            payload={},
            context="inference",
        )
    )
    for _ in range(600):
        sink.on_status_update(
            StatusUpdate(
                timestamp=datetime.now(),
                severity=UpdateSeverity.INFO,
                event_type="UPDATE",
                payload={},
                context="inference",
            )
        )
    captured(sink, 1)
    sink.on_prediction({}, frame(1))
    finish(sink, 1)
    assert sink.snapshot()["state"] == "failed"


def test_file_job_requires_local_retained_single_source():
    base = dict(
        video_configuration=dict(
            type="VideoConfiguration", video_reference="/video.mp4"
        ),
        processing_configuration=dict(type="WorkflowConfiguration"),
        file_job=dict(frame_stride=5),
        retain_results_on_eof=True,
        consumption_timeout=60,
    )
    model = InitialisePipelinePayload(**base)
    assert model.video_configuration.source_buffer_filling_strategy == "WAIT"
    assert model.video_configuration.source_buffer_consumption_strategy == "LAZY"
    for bad in ("https://blob/video.mp4", ["/a.mp4", "/b.mp4"], 0):
        with pytest.raises(ValidationError):
            InitialisePipelinePayload(
                **{
                    **base,
                    "video_configuration": dict(
                        type="VideoConfiguration", video_reference=bad
                    ),
                }
            )
    for stride in (0, -1, 1.5, True):
        with pytest.raises(ValidationError):
            InitialisePipelinePayload(**{**base, "file_job": dict(frame_stride=stride)})
