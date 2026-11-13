from datetime import datetime, timedelta, timezone

import pytest
from roboflow_workflows.core_steps.common.video_clips.grid import (
    PTS_CLOCK_RATE,
    GridSelector,
    count_inference_gaps,
    final_frame_duration_ticks,
    grid_slot,
    plan_pts,
    plan_timeline,
    to_seconds,
)


def _kept(times, *, playback_speed=1.0, clip_fps=None):
    selector = GridSelector(playback_speed=playback_speed, clip_fps=clip_fps)

    return [t for t in times if selector.accept(t)]


def test_to_seconds_reads_naive_timestamps_as_utc() -> None:
    # given
    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = naive.replace(tzinfo=timezone.utc)

    # when / then
    assert to_seconds(naive) == to_seconds(aware)


def test_grid_without_cap_keeps_every_frame() -> None:
    # given
    times = [0.0, 0.01, 0.02, 0.5, 0.5]

    # when
    kept = _kept(times)

    # then
    assert kept == times


def test_grid_keeps_first_frame_of_each_slot_with_cap() -> None:
    # given: 30 fps source, cap 10 fps -> slots of 0.1 s
    times = [i / 30 for i in range(30)]

    # when
    kept = _kept(times, clip_fps=10.0)

    # then
    assert len(kept) == 10
    assert kept == pytest.approx([i / 10 for i in range(10)])


def test_grid_slots_scale_with_playback_speed_15() -> None:
    # given: 60 s of 30 fps at speed 15 with cap 30 -> 0.5 s source slots
    times = [i / 30 for i in range(60 * 30)]

    # when
    kept = _kept(times, playback_speed=15.0, clip_fps=30.0)

    # then
    assert len(kept) == 120
    assert all(b - a == pytest.approx(0.5) for a, b in zip(kept, kept[1:]))


def test_grid_slots_scale_with_playback_speed_half() -> None:
    # given: 10 fps source, speed 0.5, cap 30 -> slots of 1/60 s keep everything
    times = [i / 10 for i in range(20)]

    # when
    kept = _kept(times, playback_speed=0.5, clip_fps=30.0)

    # then
    assert kept == times


def test_grid_handles_jittered_timestamps() -> None:
    # given: 30 fps with +-8 ms jitter, cap 15 fps
    jitter = [0.008, -0.006, 0.004, -0.008, 0.002, 0.0]
    times = [i / 30 + jitter[i % len(jitter)] for i in range(90)]

    # when
    kept = _kept(times, clip_fps=15.0)

    # then: never two frames in one slot, every slot of the 3 s represented
    slots = [
        grid_slot(t, grid_origin=times[0], playback_speed=1.0, clip_fps=15.0)
        for t in kept
    ]
    assert len(set(slots)) == len(slots)
    assert 43 <= len(kept) <= 45


def test_grid_keeps_sparse_frames() -> None:
    # given: a frame every 2 s with a 10 fps cap
    times = [0.0, 2.0, 4.0, 6.0]

    # when
    kept = _kept(times, clip_fps=10.0)

    # then
    assert kept == times


def test_grid_decimates_duplicate_and_backward_frames_with_cap() -> None:
    # given
    times = [0.0, 0.0, 0.2, 0.1, 0.4]

    # when
    kept = _kept(times, clip_fps=10.0)

    # then
    assert kept == [0.0, 0.2, 0.4]


def test_grid_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        GridSelector(playback_speed=0.0, clip_fps=None)
    with pytest.raises(ValueError):
        GridSelector(playback_speed=1.0, clip_fps=0.0)


def test_pts_start_at_zero_and_follow_source_time() -> None:
    # given
    times = [10.0, 10.5, 11.0]

    # when
    pts = plan_pts(times, actual_start=10.0, playback_speed=1.0)

    # then
    assert pts == [0, 45000, 90000]


def test_pts_are_divided_by_playback_speed() -> None:
    # given
    times = [0.0, 1.0, 2.0]

    # when
    fast = plan_pts(times, actual_start=0.0, playback_speed=15.0)
    slow = plan_pts(times, actual_start=0.0, playback_speed=0.5)

    # then
    assert fast == [0, 6000, 12000]
    assert slow == [0, 180000, 360000]


def test_duplicate_and_backward_timestamps_are_repaired_and_counted() -> None:
    # given
    times = [0.0, 0.1, 0.1, 0.05, 0.2]

    # when
    plan = plan_timeline(times, playback_speed=1.0, clip_fps=None, declared_fps=30.0)

    # then
    assert plan.pts == [0, 9000, 9001, 9002, 18000]
    assert plan.pts_clamped == 2
    assert all(b > a for a, b in zip(plan.pts, plan.pts[1:]))


def test_large_forward_gap_stays_a_gap_and_is_counted() -> None:
    # given: a 43 s model-load stall between two frames
    times = [0.0, 1 / 30, 43.0, 43.0 + 1 / 30]

    # when
    plan = plan_timeline(times, playback_speed=1.0, clip_fps=None, declared_fps=30.0)

    # then: the late frame keeps its own time, it is not moved forward/back
    assert plan.pts[2] == 43 * PTS_CLOCK_RATE
    assert plan.inference_gaps == 1
    assert plan.pts_clamped == 0


def test_backward_jump_above_a_second_is_flagged() -> None:
    # given
    times = [100.0, 100.1, 98.0, 98.1]

    # when
    plan = plan_timeline(times, playback_speed=1.0, clip_fps=None, declared_fps=None)

    # then
    assert plan.wall_clock_jump_flagged is True
    assert plan.pts_clamped == 2


def test_final_duration_uses_clip_fps_when_set() -> None:
    assert (
        final_frame_duration_ticks(
            [0.0, 0.1], playback_speed=1.0, clip_fps=10.0, declared_fps=30.0
        )
        == 9000
    )


def test_final_duration_uses_median_of_last_five_intervals_over_speed() -> None:
    # given: intervals 1, 1, 1, 1, 1 (outlier 5 s earlier is outside the tail)
    times = [0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # when
    ticks = final_frame_duration_ticks(
        times, playback_speed=2.0, clip_fps=None, declared_fps=30.0
    )

    # then
    assert ticks == 45000


def test_final_duration_falls_back_to_declared_fps_then_30() -> None:
    assert (
        final_frame_duration_ticks(
            [1.0], playback_speed=1.0, clip_fps=None, declared_fps=25.0
        )
        == 3600
    )
    assert (
        final_frame_duration_ticks(
            [1.0], playback_speed=1.0, clip_fps=None, declared_fps=None
        )
        == 3000
    )


def test_timeline_duration_and_end_offset() -> None:
    # given
    times = [i / 30 for i in range(30)]

    # when
    plan = plan_timeline(times, playback_speed=1.0, clip_fps=None, declared_fps=30.0)

    # then
    assert plan.duration_ticks == 90000
    assert plan.actual_end_offset_seconds == pytest.approx(1.0)


def test_inference_gaps_respect_grid_slot_length() -> None:
    # given: frames 0.5 s apart because of a 2 fps cap, one 2 s hole
    times = [0.0, 0.5, 1.0, 3.0, 3.5]

    # when
    gaps = count_inference_gaps(
        times, playback_speed=1.0, clip_fps=2.0, declared_fps=30.0
    )

    # then
    assert gaps == 1


def test_empty_timeline() -> None:
    plan = plan_timeline([], playback_speed=1.0, clip_fps=None, declared_fps=None)

    assert plan.pts == []
    assert plan.duration_ticks == 0


def test_timestamps_from_datetimes_round_trip_within_one_tick() -> None:
    # given
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamps = [start + timedelta(microseconds=33333 * i) for i in range(100)]
    times = [to_seconds(stamp) for stamp in stamps]

    # when
    pts = plan_pts(times, actual_start=times[0], playback_speed=1.0)

    # then
    for index, value in enumerate(pts):
        assert abs(value - round(0.033333 * index * PTS_CLOCK_RATE)) <= 1
