import importlib
import threading

import pytest
from roboflow_workflows.core_steps.common import video_clips
from roboflow_workflows.core_steps.common.video_clips.admission import (
    REJECT_BUDGET_EXCEEDED,
    REJECT_CLIP_TOO_LARGE,
    REJECT_TOO_MANY_PENDING_CAPTURES,
    Budget,
    default_budget,
)


class _Capture:
    def __init__(self, estimated_size_bytes: int = 0):
        self.estimated_size_bytes = estimated_size_bytes
        self.released = False

    def release(self) -> None:
        self.released = True


class _Source:
    def __init__(self, held: int):
        self.held = held

    def budget_bytes(self) -> int:
        return self.held


def test_admit_returns_ticket_and_counts_pending_captures() -> None:
    # given
    budget = Budget(budget_bytes=1000, max_pending_captures=2, max_clip_bytes=500)

    # when
    first = budget.admit(_Capture())
    second = budget.admit(_Capture())
    third_capture = _Capture()
    third = budget.admit(third_capture)

    # then
    assert first.admitted and second.admitted
    assert third.admitted is False
    assert third.reason == REJECT_TOO_MANY_PENDING_CAPTURES
    assert third.ticket is None
    assert third_capture.released is True
    assert budget.usage().pending_captures == 2

    first.ticket.release()
    first.ticket.release()
    assert budget.admit(_Capture()).admitted is True


def test_clip_larger_than_max_clip_bytes_is_rejected() -> None:
    # given
    budget = Budget(budget_bytes=10_000, max_clip_bytes=500)
    capture = _Capture(estimated_size_bytes=501)

    # when
    decision = budget.admit(capture)

    # then
    assert decision.reason == REJECT_CLIP_TOO_LARGE
    assert capture.released is True


def test_byte_cap_counts_admitted_clips_and_the_new_estimate() -> None:
    # given
    budget = Budget(budget_bytes=1000, max_clip_bytes=1000)
    admitted = budget.admit(_Capture(estimated_size_bytes=100))
    admitted.ticket.set_bytes(700)

    # when
    rejected = budget.admit(_Capture(estimated_size_bytes=400))
    fits = budget.admit(_Capture(estimated_size_bytes=300))

    # then
    assert admitted.admitted is True
    assert rejected.reason == REJECT_BUDGET_EXCEEDED
    assert fits.admitted is True
    assert budget.usage().ticket_bytes == 1000


def test_captures_admitted_together_cannot_use_the_same_free_bytes() -> None:
    # given: room for one clip of this size, not for two. Neither clip is muxed
    # yet, so only the estimates can tell.
    budget = Budget(budget_bytes=150, max_clip_bytes=150)
    first_capture = _Capture(estimated_size_bytes=112)
    second_capture = _Capture(estimated_size_bytes=112)

    # when
    first = budget.admit(first_capture)
    second = budget.admit(second_capture)

    # then
    assert first.admitted is True
    assert first.ticket.bytes == 112
    assert second.reason == REJECT_BUDGET_EXCEEDED
    assert second_capture.released is True

    # and: the bytes are free again once the first clip reaches a terminal status
    first.ticket.release()
    assert budget.admit(_Capture(estimated_size_bytes=112)).admitted is True


def test_recording_memory_does_not_count_against_the_byte_cap() -> None:
    # given: the rings alone hold more than the cap (long pre-rolls, many streams)
    budget = Budget(budget_bytes=1000, max_clip_bytes=1000)
    source = _Source(held=5000)
    budget.register_source(source)

    # when
    decision = budget.admit(_Capture(estimated_size_bytes=900))

    # then: the clip is admitted, and recording memory is still reported
    assert decision.admitted is True
    assert budget.usage().source_bytes == 5000
    assert budget.usage().ticket_bytes == 900

    budget.unregister_source(source)
    assert budget.usage().source_bytes == 0


def test_sources_are_held_weakly() -> None:
    # given
    budget = Budget()
    budget.register_source(_Source(held=10))

    # when
    usage = budget.usage()

    # then
    assert usage.source_bytes == 0


def test_upload_slots_bound_concurrency() -> None:
    # given
    budget = Budget(max_upload_concurrency=2)
    active = []
    peak = []
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def _upload() -> None:
        with budget.upload_slot():
            with lock:
                active.append(1)
                peak.append(len(active))
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                pass
            with lock:
                active.pop()

    # when
    threads = [threading.Thread(target=_upload) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # then
    assert max(peak) == 2


def test_invalid_caps_are_rejected() -> None:
    with pytest.raises(ValueError):
        Budget(budget_bytes=0)


def test_default_budget_is_a_singleton() -> None:
    assert default_budget() is default_budget()


def test_caps_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # given
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES", "2048")
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES", "not-a-number")
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES", "-5")

    # when
    reloaded = importlib.reload(video_clips)

    # then
    try:
        assert reloaded.WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES == 2048
        assert reloaded.WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES == 16
        assert reloaded.WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES == 512 * 1024 * 1024
    finally:
        monkeypatch.undo()
        importlib.reload(video_clips)


def test_env_positive_float_clamps_to_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    # given
    monkeypatch.setenv("SOME_VIDEO_CLIP_SETTING", "900")

    # when
    value = video_clips.env_positive_float(
        "SOME_VIDEO_CLIP_SETTING", default=60.0, maximum=300.0
    )

    # then
    assert value == 300.0
