from datetime import datetime, timedelta, timezone

import pytest
from roboflow_workflows.core_steps.common.video_clips.artifact import ClipArtifact
from roboflow_workflows.core_steps.common.video_clips.capture import (
    ClipAnchor,
    ClipUnavailableError,
    PendingCapture,
    bind_capture,
)
from roboflow_workflows.core_steps.common.video_clips.report import ClipReport
from roboflow_workflows.core_steps.common.video_clips.ring import (
    EncodedPacket,
    PacketStore,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _packet(t: float, *, key: bool, size: int = 100) -> EncodedPacket:
    return EncodedPacket(
        frame_timestamp=T0 + timedelta(seconds=t),
        t=t,
        frame_number=int(t * 10),
        payload=b"\x00" * size,
        is_keyframe=key,
        width=4,
        height=4,
    )


def _store(seconds: int = 5, fps: int = 10) -> PacketStore:
    store = PacketStore()
    for index in range(seconds * fps):
        store.append(_packet(index / fps, key=index % fps == 0))

    return store


def _pending() -> PendingCapture:
    return PendingCapture(
        capture_id="c",
        video_identifier="v",
        backend="x264",
        requested_window=(T0, T0 + timedelta(seconds=1)),
        available_from=None,
        estimated_size_bytes=0,
    )


def test_snapshot_starts_at_latest_keyframe_and_includes_window_end() -> None:
    # given
    store = _store()

    # when
    packets = store.snapshot(2.5, 3.0, max_lead_seconds=1.0)

    # then
    assert packets[0].t == 2.0 and packets[0].is_keyframe
    assert packets[-1].t == pytest.approx(3.0)


def test_snapshot_lead_is_capped_at_one_output_second() -> None:
    # given: keyframes every 2 s, lead cap 1 s
    store = PacketStore()
    for index in range(60):
        store.append(_packet(index / 10, key=index % 20 == 0))

    # when
    packets = store.snapshot(3.5, 5.0, max_lead_seconds=1.0)

    # then: keyframe 2.0 leads by 1.5 s, so the clip starts at keyframe 4.0
    assert packets[0].t == pytest.approx(4.0)


def test_snapshot_without_earlier_keyframe_starts_at_first_window_keyframe() -> None:
    # given
    store = PacketStore()
    for index in range(10, 30):
        store.append(_packet(index / 10, key=index % 10 == 0))

    # when
    packets = store.snapshot(0.0, 2.5, max_lead_seconds=1.0)

    # then
    assert packets[0].t == pytest.approx(1.0)
    assert packets[-1].t == pytest.approx(2.5)


def test_time_eviction_is_gop_wise_and_records_evicted_time() -> None:
    # given
    store = _store()

    # when
    evicted = store.evict_older_than(2.5)

    # then
    assert len(evicted) == 20
    assert store.oldest.t == 2.0
    assert store.evicted_until_t == pytest.approx(1.9)


def test_byte_cap_keeps_the_newest_gop() -> None:
    # given
    store = _store()

    # when
    evicted = store.enforce_cap(1)

    # then
    assert len(evicted) == 40
    assert len(store) == 10
    assert store.bytes == 1000


def test_referenced_packets_stay_held_until_released() -> None:
    # given
    store = _store()
    packets = store.snapshot(0.0, 0.5, max_lead_seconds=1.0)
    store.acquire(packets)
    store.acquire(packets)

    # when
    store.evict_older_than(3.0)

    # then
    assert store.held_bytes == 600
    store.release(packets)
    assert store.held_bytes == 600
    store.release(packets)
    assert store.held_bytes == 0
    assert all(packet.refs == 0 for packet in packets)


def test_byte_rate_needs_a_second_of_span() -> None:
    store = PacketStore()
    store.append(_packet(0.0, key=True))
    store.append(_packet(0.5, key=False))
    assert store.byte_rate() is None

    store.append(_packet(2.0, key=False))
    assert store.byte_rate() == pytest.approx(150.0)


def test_unavailable_anchor_raises_with_its_reason() -> None:
    # given
    anchor = ClipAnchor(
        recorder_step="recorder",
        video_identifier="v",
        frame_timestamp=None,
        frame_number=1,
        availability="unavailable:no_timestamp",
        label="Evidence",
        pre_roll_seconds=5.0,
        post_roll_seconds=5.0,
        _capture=None,
    )

    # when / then
    assert anchor.available is False
    with pytest.raises(ClipUnavailableError) as error:
        anchor.capture(5.0, 5.0)
    assert error.value.reason == "unavailable:no_timestamp"


def test_anchor_metadata_has_no_frames_or_packets() -> None:
    # given
    anchor = ClipAnchor(
        recorder_step="recorder",
        video_identifier="v",
        frame_timestamp=T0,
        frame_number=7,
        availability="ok",
        label="Evidence",
        pre_roll_seconds=5.0,
        post_roll_seconds=5.0,
        _capture=lambda pre, post: _pending(),
    )

    # when
    metadata = anchor.to_metadata()

    # then
    assert metadata == {
        "kind": "video_clip_anchor",
        "recorder": "recorder",
        "video_identifier": "v",
        "frame_timestamp": T0.isoformat(),
        "frame_number": 7,
        "availability": "ok",
    }


def test_bind_capture_maps_rolls_to_an_inclusive_window() -> None:
    # given
    calls = []

    class _Ring:
        def capture(self, start, end_inclusive):
            calls.append((start, end_inclusive))
            return _pending()

    capture = bind_capture(_Ring(), T0)

    # when
    capture(2.0, 0.0)

    # then
    assert calls == [(T0 - timedelta(seconds=2), T0)]


def test_release_before_resolution_cancels_and_calls_back_once() -> None:
    # given
    cancelled = []
    pending = PendingCapture(
        capture_id="c",
        video_identifier="v",
        backend="x264",
        requested_window=(T0, T0 + timedelta(seconds=1)),
        available_from=T0,
        estimated_size_bytes=10,
        on_cancel=lambda: cancelled.append(True),
    )

    # when
    pending.release()
    pending.release()

    # then
    assert pending.future.cancelled()
    assert cancelled == [True]
    assert pending.requested_window_dict() == {
        "start": T0.isoformat(),
        "end": (T0 + timedelta(seconds=1)).isoformat(),
    }


def test_release_after_resolution_releases_the_artifact() -> None:
    # given
    released = []
    pending = _pending()
    pending.future.set_running_or_notify_cancel()
    artifact = ClipArtifact(
        b"x",
        size_bytes=1,
        duration_seconds=0.1,
        report=ClipReport(requested_window=(T0, T0)),
        on_release=lambda: released.append(True),
    )

    # when: released while the finalizer is still running
    pending.release()
    pending.future.set_result(artifact)

    # then
    assert released == [True]
    assert artifact.released is True
