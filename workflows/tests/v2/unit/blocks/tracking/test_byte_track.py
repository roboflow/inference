"""ByteTrack's per-source state and direct-call contract."""

import pytest
from roboflow_workflows.v2.blocks.tracking.byte_tracker.v3 import ByteTrackerBlock
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.errors import EventEmissionError

from .test_base import _image, _prediction


def _run(block, *, source="a", prediction=None, **overrides):
    params = dict(
        track_activation_threshold=0.25,
        lost_track_buffer=1,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
        max_sources=2,
        frame_rate=30.0,
        instances_cache_size=10,
    )
    params.update(overrides)
    result = block.run(
        image=_image(source),
        predictions=_prediction() if prediction is None else prediction,
        **params
    )

    return result


def _ids(prediction):
    ids = [row["tracker_id"] for row in prediction.bboxes_metadata or []]

    return ids


def test_interleaved_sources_have_independent_new_seen_and_fixed_parameters():
    block = ByteTrackerBlock()
    for source in ("b", "a"):
        first = _run(block, source=source)
        assert _ids(first["tracked_detections"]) == [1, 2, 3]
        assert len(first["new_instances"]) == 3
        assert len(first["already_seen_instances"]) == 0
    for source in ("a", "b"):
        later = _run(
            block,
            source=source,
            track_activation_threshold=1.0,
            minimum_consecutive_frames=10,
            instances_cache_size=1,
        )
        assert _ids(later["tracked_detections"]) == [1, 2, 3]
        assert len(later["already_seen_instances"]) == 3
        assert len(later["new_instances"]) == 0
        assert block._trackers[source].instances.size == 10


def test_cache_eviction_makes_still_tracked_id_new_again():
    block = ByteTrackerBlock()
    _run(block, instances_cache_size=1)
    later = _run(block, instances_cache_size=1)
    assert _ids(later["tracked_detections"]) == [1, 2, 3]
    assert len(later["new_instances"]) == 3
    assert len(later["already_seen_instances"]) == 0


def test_empty_frames_age_tracker_and_new_id_is_reported_on_return():
    from roboflow_workflows.v2.data.prediction_ops import select_predictions

    block = ByteTrackerBlock()
    first = _run(block)
    empty = select_predictions(_prediction(), indices=[])
    for _ in range(3):
        outputs = _run(block, prediction=empty)
        assert all(len(value) == 0 for value in outputs.values())
    _run(block)  # A returning track needs confirmation after the first frame.
    later = _run(block)
    assert _ids(later["tracked_detections"])
    assert not set(_ids(first["tracked_detections"])) & set(
        _ids(later["tracked_detections"])
    )


def test_direct_run_only_falls_back_for_public_missing_context(monkeypatch):
    from roboflow_workflows.v2.blocks.tracking.byte_tracker import v3 as byte_track

    block = ByteTrackerBlock()

    def absent():
        raise NoExecutionContextError("plain call")

    monkeypatch.setattr(byte_track, "get_execution_context", absent)
    assert set(_run(block)) == set(block.outputs)

    def broken():
        raise EventEmissionError("bad engine context")

    monkeypatch.setattr(byte_track, "get_execution_context", broken)
    with pytest.raises(EventEmissionError, match="bad engine context"):
        _run(block)


def test_run_advances_tracker_and_fifo_even_without_output_demand(monkeypatch):
    block = ByteTrackerBlock()
    monkeypatch.setattr(block, "_wanted_outputs", lambda: set())
    assert _run(block) == {}
    state = block._trackers["a"]
    assert len(state.instances) == 3
    assert state.instances.split([1, 2, 3]) == ([], [0, 1, 2])
    monkeypatch.setattr(block, "_wanted_outputs", lambda: set(block.outputs))
    later = _run(block)
    assert _ids(later["tracked_detections"]) == [1, 2, 3]
    assert len(later["new_instances"]) == 0
    assert len(later["already_seen_instances"]) == 3
