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


def test_byte_cap_counts_sources_tickets_and_the_new_estimate() -> None:
    # given
    budget = Budget(budget_bytes=1000, max_clip_bytes=1000)
    source = _Source(held=600)
    budget.register_source(source)
    admitted = budget.admit(_Capture(estimated_size_bytes=100))
    admitted.ticket.set_bytes(250)

    # when
    rejected = budget.admit(_Capture(estimated_size_bytes=200))
    fits = budget.admit(_Capture(estimated_size_bytes=150))

    # then
    assert admitted.admitted is True
    assert rejected.reason == REJECT_BUDGET_EXCEEDED
    assert fits.admitted is True
    assert budget.usage().total_bytes == 850

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
