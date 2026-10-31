import io
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, List

import av
import cv2
import numpy as np
import pytest
from roboflow_workflows.core_steps.common.video_clips.admission import Budget
from roboflow_workflows.core_steps.common.video_clips.artifact import ClipArtifact
from roboflow_workflows.core_steps.common.video_clips.capture import (
    PARTIAL_RESOLUTION_CHANGE,
    PARTIAL_RING_EVICTED,
    PARTIAL_SHUTDOWN,
    PARTIAL_STALLED,
    ClipAnchor,
    ClipEncodeError,
    ClipUnavailableError,
    bind_capture,
)
from roboflow_workflows.core_steps.common.video_clips.grid import PTS_CLOCK_RATE
from roboflow_workflows.core_steps.common.video_clips.ring import Ring
from roboflow_workflows.core_steps.common.video_clips.x264_ring import X264Ring
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FPS = 30


def _pixels(index: int, *, width: int = 320, height: int = 240) -> np.ndarray:
    pixels = np.full((height, width, 3), (index * 7) % 200, dtype=np.uint8)
    cv2.putText(
        pixels, str(index), (10, height // 2), 0, height / 120, (255, 255, 255), 2
    )

    return pixels


def _image(index: int, *, width: int = 320, height: int = 240) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=None,
        numpy_image=_pixels(index, width=width, height=height),
    )


def _ts(index: int, fps: float = FPS) -> datetime:
    return T0 + timedelta(seconds=index / fps)


def _ring(**kwargs) -> X264Ring:
    settings = {"pre_roll_seconds": 2.0, "budget": Budget(), "queue_depth": 1000}
    settings.update(kwargs)

    return X264Ring("camera-1", **settings)


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.005)


def _feed(ring: X264Ring, indices, **image_kwargs) -> None:
    for index in indices:
        ring.submit(_ts(index), index, _image(index, **image_kwargs))
    _wait_for(lambda: ring.stats().queue_depth == 0)


def _decode(artifact: ClipArtifact):
    container = av.open(io.BytesIO(artifact.read_bytes()))
    stream = container.streams.video[0]
    frames = list(container.decode(stream))
    duration = float(stream.duration * stream.time_base)
    container.close()

    return frames, duration


@pytest.fixture
def ring():
    created: List[X264Ring] = []

    def _factory(**kwargs) -> X264Ring:
        instance = _ring(**kwargs)
        created.append(instance)
        return instance

    yield _factory
    for instance in created:
        instance.close()


def test_x264_ring_implements_the_ring_protocol(ring) -> None:
    assert isinstance(ring(), Ring)


def test_post_roll_zero_clip_contains_the_trigger_frame(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(61))
    anchor = ClipAnchor(
        recorder_step="recorder",
        video_identifier="camera-1",
        frame_timestamp=_ts(60),
        frame_number=60,
        availability="ok",
        label="Evidence",
        pre_roll_seconds=1.0,
        post_roll_seconds=0.0,
        _capture=bind_capture(recorder, _ts(60)),
    )

    # when
    capture = anchor.capture(anchor.pre_roll_seconds, anchor.post_roll_seconds)
    _feed(recorder, range(61, 63))
    artifact = capture.future.result(timeout=10)

    # then
    report = artifact.report
    assert report.frames == 31
    assert report.actual_start == _ts(30)
    assert report.lead_seconds == 0.0
    assert report.partial is None
    frames, _ = _decode(artifact)
    assert len(frames) == 31
    assert frames[0].pts == 0
    artifact.release()


@pytest.mark.parametrize(
    "playback_speed, start_index",
    [(1.0, 44), (1.0, 59), (1.0, 31), (2.0, 89), (2.0, 61)],
)
def test_gop_lead_never_exceeds_one_output_second(
    ring, playback_speed: float, start_index: int
) -> None:
    # given
    recorder = ring(playback_speed=playback_speed, pre_roll_seconds=4.0)
    _feed(recorder, range(150))

    # when
    capture = recorder.capture(_ts(start_index), _ts(start_index + 15))
    artifact = capture.future.result(timeout=10)

    # then
    report = artifact.report
    assert 0.0 <= report.lead_seconds <= playback_speed + 1e-6
    assert report.actual_start <= _ts(start_index)
    frames, _ = _decode(artifact)
    assert frames[0].key_frame
    assert len(frames) == report.frames
    artifact.release()


def test_future_resolves_on_the_ring_thread_and_packets_are_released_once_muxed(
    ring,
) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(60))
    resolved_on = []
    first = recorder.capture(_ts(30), _ts(70))
    second = recorder.capture(_ts(35), _ts(70))
    first.future.add_done_callback(
        lambda _: resolved_on.append(threading.current_thread().name)
    )

    # when: before completion both captures reference the same packets
    pending = list(recorder._pending)
    shared = set(map(id, pending[0].packets)) & set(map(id, pending[1].packets))
    assert shared
    assert all(packet.refs == 2 for packet in pending[1].packets)
    captured_packets = pending[0].packets + pending[1].packets
    _feed(recorder, range(60, 75))
    artifacts = [first.future.result(timeout=10), second.future.result(timeout=10)]

    # then
    assert resolved_on and "finalizer" in resolved_on[0]
    assert resolved_on[0] != threading.current_thread().name
    # the muxed MP4s own their bytes, so no packet stays referenced by them
    assert all(packet.refs == 0 for packet in captured_packets)
    _feed(recorder, range(75, 400))
    stats = recorder.stats()
    assert stats.held_bytes == 0
    assert all(packet.refs == 0 for packet in recorder._store._packets)
    assert all(artifact.read_bytes() for artifact in artifacts)
    for artifact in artifacts:
        artifact.release()


def test_byte_cap_eviction_marks_capture_ring_evicted(ring) -> None:
    # given: noisy frames and a tiny cap force evictions
    recorder = ring(byte_cap_bytes=200_000, pre_roll_seconds=5.0)
    rng = np.random.default_rng(0)
    for index in range(150):
        noisy = WorkflowImageData(
            parent_metadata=None,
            numpy_image=rng.integers(0, 255, (240, 320, 3), dtype=np.uint8),
        )
        recorder.submit(_ts(index), index, noisy)
    _wait_for(lambda: recorder.stats().queue_depth == 0)
    assert recorder.stats().ring_drops > 0
    assert recorder.stats().packets <= 2 * FPS

    # when
    capture = recorder.capture(_ts(0), _ts(140))
    artifact = capture.future.result(timeout=10)

    # then
    report = artifact.report
    assert report.partial == PARTIAL_RING_EVICTED
    assert report.actual_start > _ts(0)
    assert report.loss.ring_drops > 0
    assert capture.available_from == report.actual_start
    artifact.release()


def test_stalled_capture_is_finalised_as_partial(ring) -> None:
    # given
    recorder = ring(stall_grace_seconds=0.2)
    _feed(recorder, range(30))

    # when: post-roll ends 0.3 s after the newest frame, then frames stop
    started = time.monotonic()
    capture = recorder.capture(_ts(0), _ts(29) + timedelta(seconds=0.3))
    artifact = capture.future.result(timeout=5)

    # then
    assert artifact.report.partial == PARTIAL_STALLED
    assert artifact.report.frames == 30
    assert time.monotonic() - started >= 0.45
    artifact.release()


def test_source_slower_than_real_time_is_not_finalised_as_stalled(ring) -> None:
    # given: a 10 fps source processed at 1/3 real time and a 0.5 s grace
    recorder = ring(stall_grace_seconds=0.5)
    source_fps = 10
    for index in range(20):
        recorder.submit(_ts(index, source_fps), index, _image(index))
    _wait_for(lambda: recorder.stats().queue_depth == 0)

    # when: a 1 s post-roll needs 3 s of wall time to arrive
    capture = recorder.capture(
        _ts(10, source_fps), _ts(19, source_fps) + timedelta(seconds=1)
    )
    for index in range(20, 31):
        time.sleep(0.3)
        recorder.submit(_ts(index, source_fps), index, _image(index))
    artifact = capture.future.result(timeout=10)

    # then: the whole post-roll is in the clip
    report = artifact.report
    assert report.partial is None
    assert report.actual_end == _ts(30, source_fps)
    assert report.frames == 20
    artifact.release()


class _SlowX264Ring(X264Ring):
    def _encode_frame(self, frame, *, pts, force_keyframe):
        time.sleep(0.02)
        return super()._encode_frame(frame, pts=pts, force_keyframe=force_keyframe)


def test_prepare_close_drops_queue_and_finalises_eight_captures_fast() -> None:
    # given: a slow encoder with a long queue and 8 pending captures
    recorder = _SlowX264Ring(
        "camera-1", pre_roll_seconds=2.0, budget=Budget(), queue_depth=200
    )
    try:
        _feed(recorder, range(10))
        for index in range(10, 200):
            recorder.submit(_ts(index), index, _image(index))
        captures = [recorder.capture(_ts(0), _ts(100 + index)) for index in range(8)]
        assert recorder.stats().queue_depth > 50

        # when
        started = time.monotonic()
        recorder.prepare_close(5.0)
        artifacts = [capture.future.result(timeout=5) for capture in captures]
        elapsed = time.monotonic() - started

        # then
        assert elapsed < 0.5
        assert all(a.report.partial == PARTIAL_SHUTDOWN for a in artifacts)
        assert all(a.report.frames >= 10 for a in artifacts)
        stats = recorder.stats()
        assert stats.queue_depth == 0
        assert stats.queue_bytes == 0
        assert stats.queue_drops > 50
        assert artifacts[0].report.loss.queue_drops > 50
        for artifact in artifacts:
            artifact.release()
    finally:
        recorder.close()
    assert recorder.stats().held_bytes == 0


def test_concurrent_close_from_two_threads_is_safe(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(20))
    capture = recorder.capture(_ts(0), _ts(100))
    errors = []

    def _close() -> None:
        try:
            recorder.close()
        except Exception as error:
            errors.append(error)

    # when
    threads = [threading.Thread(target=_close) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # then
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert capture.future.result(timeout=1).report.partial == PARTIAL_SHUTDOWN
    assert not recorder._encoder_thread.is_alive()
    assert not recorder._finalizer_thread.is_alive()
    recorder.close()
    capture.release()


def test_closed_ring_refuses_frames_and_captures(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(5))
    recorder.close()

    # when
    recorder.submit(_ts(6), 6, _image(6))

    # then
    assert recorder.stats().frames_submitted == 5
    with pytest.raises(ClipUnavailableError) as error:
        recorder.capture(_ts(0), _ts(1))
    assert error.value.reason == "unavailable:closing"


def test_resolution_change_finalises_pending_capture_as_partial(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(30))
    capture = recorder.capture(_ts(10), _ts(60))

    # when
    _feed(recorder, range(30, 40), width=160, height=120)
    artifact = capture.future.result(timeout=10)

    # then
    assert artifact.report.partial == PARTIAL_RESOLUTION_CHANGE
    assert artifact.report.width == 320
    later = recorder.capture(_ts(31), _ts(35))
    _feed(recorder, range(40, 42), width=160, height=120)
    later_artifact = later.future.result(timeout=10)
    assert later_artifact.report.width == 160
    artifact.release()
    later_artifact.release()


def test_retire_finalises_pending_with_reason_and_stops_threads(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(30))
    capture = recorder.capture(_ts(10), _ts(60))

    # when
    recorder.retire(PARTIAL_RESOLUTION_CHANGE)
    artifact = capture.future.result(timeout=5)

    # then
    assert artifact.report.partial == PARTIAL_RESOLUTION_CHANGE
    _wait_for(lambda: not recorder._encoder_thread.is_alive())
    _wait_for(lambda: not recorder._finalizer_thread.is_alive())
    artifact.release()


def test_retired_ring_releases_itself_once_its_last_capture_is_finalised(
    ring,
) -> None:
    # given
    budget = Budget()
    recorder = ring(budget=budget)
    _feed(recorder, range(30))
    capture = recorder.capture(_ts(10), _ts(60))
    released = threading.Event()

    # when
    recorder.retire(PARTIAL_RESOLUTION_CHANGE, on_released=released.set)
    artifact = capture.future.result(timeout=5)

    # then: packets and the budget share are gone without close()
    assert released.wait(timeout=5)
    assert recorder.budget_bytes() == 0
    assert budget.usage().source_bytes == 0
    assert recorder not in budget._sources
    artifact.release()
    recorder.close()


def test_retired_ring_without_frames_releases_itself_at_once(ring) -> None:
    # given
    budget = Budget()
    recorder = ring(budget=budget)
    released = threading.Event()

    # when
    recorder.retire(PARTIAL_RESOLUTION_CHANGE, on_released=released.set)

    # then
    assert released.is_set()
    assert recorder not in budget._sources


def test_submit_never_keeps_a_reference_to_the_callers_frame() -> None:
    # given: the encoder is slow, so the frame is still queued when mutated
    recorder = _SlowX264Ring(
        "camera-1", pre_roll_seconds=2.0, budget=Budget(), queue_depth=100
    )
    try:
        pixels = np.zeros((240, 320, 3), dtype=np.uint8)
        image = WorkflowImageData(parent_metadata=None, numpy_image=pixels)
        for index in range(20):
            recorder.submit(_ts(index), index, image)
        pixels[:] = 255
        capture = recorder.capture(_ts(0), _ts(18))
        _feed(recorder, range(20, 22))

        # when
        artifact = capture.future.result(timeout=10)
        frames, _ = _decode(artifact)

        # then
        assert all(frame.to_ndarray(format="gray").mean() < 40 for frame in frames)
        artifact.release()
    finally:
        recorder.close()


def test_full_queue_counts_queue_drops() -> None:
    # given
    recorder = _SlowX264Ring(
        "camera-1", pre_roll_seconds=2.0, budget=Budget(), queue_depth=2
    )
    try:
        # when
        for index in range(20):
            recorder.submit(_ts(index), index, _image(index))

        # then
        assert recorder.stats().queue_drops >= 15
    finally:
        recorder.close()


def test_released_pending_capture_frees_its_packets(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(60))
    capture = recorder.capture(_ts(10), _ts(1000))
    _feed(recorder, range(60, 400))
    assert recorder.stats().held_bytes > 0

    # when
    capture.release()
    capture.release()

    # then
    assert capture.future.cancelled()
    assert recorder.stats().held_bytes == 0
    assert recorder.stats().pending_captures == 0


def test_window_with_no_frames_fails_with_clip_encode_error(ring) -> None:
    # given
    recorder = ring(stall_grace_seconds=0.1)

    # when
    capture = recorder.capture(_ts(0), _ts(1))

    # then
    with pytest.raises(ClipEncodeError):
        capture.future.result(timeout=5)


def test_speed_and_clip_fps_decimate_and_report(ring) -> None:
    # given: 30 fps source, 15x speed, 30 fps output -> one frame per 0.5 s
    recorder = ring(playback_speed=15.0, clip_fps=30.0, pre_roll_seconds=10.0)
    _feed(recorder, range(20 * FPS))

    # when
    capture = recorder.capture(_ts(5 * FPS), _ts(15 * FPS))
    _feed(recorder, range(20 * FPS, 21 * FPS))
    artifact = capture.future.result(timeout=10)

    # then
    report = artifact.report
    assert report.loss.decimated == pytest.approx(10 * FPS - 21, abs=2)
    assert report.lead_seconds <= 15.0
    frames, duration = _decode(artifact)
    assert len(frames) == report.frames
    assert all(
        b.pts - a.pts == PTS_CLOCK_RATE // 30 for a, b in zip(frames, frames[1:])
    )
    assert duration == pytest.approx(report.frames / 30, abs=1 / 30)
    artifact.release()


def test_stale_frames_are_counted_in_the_window(ring) -> None:
    # given
    recorder = ring()
    _feed(recorder, range(30))
    recorder.record_stale(_ts(15))
    recorder.record_stale(_ts(100))
    capture = recorder.capture(_ts(0), _ts(29))
    _feed(recorder, range(30, 32))

    # when
    artifact = capture.future.result(timeout=10)

    # then
    assert artifact.report.loss.stale_dropped == 1
    assert recorder.stats().stale_dropped == 2
    artifact.release()


def test_ring_bytes_count_against_the_budget() -> None:
    # given
    budget = Budget()
    recorder = X264Ring("camera-1", pre_roll_seconds=2.0, budget=budget)
    try:
        # when
        _feed(recorder, range(30))

        # then
        assert budget.usage().source_bytes == recorder.stats().bytes > 0
    finally:
        recorder.close()
    assert budget.usage().source_bytes == 0


def test_max_width_downscales_with_even_dimensions(ring) -> None:
    # given
    recorder = ring(max_width=200)
    _feed(recorder, range(20), width=321, height=241)

    # when
    capture = recorder.capture(_ts(0), _ts(18))
    _feed(recorder, range(20, 22), width=321, height=241)
    artifact = capture.future.result(timeout=10)

    # then
    assert (artifact.report.width, artifact.report.height) == (200, 150)
    artifact.release()


def test_300_frame_1080p_clip_decodes_with_exact_timing(ring) -> None:
    # given: 300 frames of 1080p at a jittered ~30 fps
    recorder = ring(pre_roll_seconds=10.0)
    jitter_us = [0, 4000, -3000, 2000, -5000, 1000]
    stamps = [
        T0 + timedelta(microseconds=33333 * i + jitter_us[i % len(jitter_us)])
        for i in range(300)
    ]
    for index, stamp in enumerate(stamps):
        recorder.submit(stamp, index, _image(index, width=1920, height=1080))
    _wait_for(lambda: recorder.stats().frames_encoded == 300, timeout=30)

    # when
    capture = recorder.capture(stamps[0], stamps[298])
    recorder.submit(
        stamps[299] + timedelta(seconds=1 / 30),
        300,
        _image(300, width=1920, height=1080),
    )
    artifact = capture.future.result(timeout=30)
    frames, duration = _decode(artifact)

    # then
    assert artifact.report.frames == 299
    assert len(frames) == 299
    expected_pts = [
        round((stamp - stamps[0]).total_seconds() * PTS_CLOCK_RATE)
        for stamp in stamps[:299]
    ]
    assert all(abs(f.pts - e) <= 1 for f, e in zip(frames, expected_pts))
    expected_duration = (stamps[298] - stamps[0]).total_seconds() + 1 / 30
    assert duration == pytest.approx(expected_duration, abs=1 / 30)
    assert (frames[0].width, frames[0].height) == (1920, 1080)
    artifact.release()
