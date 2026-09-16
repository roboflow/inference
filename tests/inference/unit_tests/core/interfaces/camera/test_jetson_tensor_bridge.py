from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference.core.interfaces.camera.jetson_tensor_bridge import (
    NativeJetsonTensorPipeline,
)


def queued_pipeline(*frames):
    queue = deque(frames)
    pipeline = NativeJetsonTensorPipeline.__new__(NativeJetsonTensorPipeline)
    pipeline._handle = 1
    pipeline._grabbed_tensor = None
    pipeline._library = SimpleNamespace(
        rf_jetson_pipeline_grab=lambda *args: 1 if queue else 0,
    )
    pipeline._take_ready_tensor = Mock(side_effect=queue.popleft)
    return pipeline, queue


def test_repeated_grabs_advance_without_retrieve():
    first, second, third = object(), object(), object()
    pipeline, queue = queued_pipeline(first, second, third)

    assert pipeline.grab()
    assert pipeline.grab()

    assert pipeline.retrieve() is second
    assert list(queue) == [third]


def test_retrieve_keeps_selected_frame_when_live_queue_changes():
    selected, newer = object(), object()
    pipeline, queue = queued_pipeline(selected)
    assert pipeline.grab()
    queue.append(newer)

    assert pipeline.retrieve() is selected
    assert pipeline.retrieve() is selected
    assert pipeline._take_ready_tensor.call_count == 1
    assert list(queue) == [newer]


def test_eos_clears_the_previous_frame():
    pipeline, _ = queued_pipeline(object())
    assert pipeline.grab()
    assert not pipeline.grab()

    with pytest.raises(RuntimeError, match="No grabbed frame"):
        pipeline.retrieve()


def test_timeout_clears_the_previous_frame():
    pipeline, _ = queued_pipeline(object())
    assert pipeline.grab()

    with pytest.raises(TimeoutError):
        pipeline.grab(timeout_ns=0)
    with pytest.raises(RuntimeError, match="No grabbed frame"):
        pipeline.retrieve()


def test_retrieve_requires_a_successful_grab():
    pipeline, queue = queued_pipeline(object())

    with pytest.raises(RuntimeError, match="No grabbed frame"):
        pipeline.retrieve()
    assert len(queue) == 1
