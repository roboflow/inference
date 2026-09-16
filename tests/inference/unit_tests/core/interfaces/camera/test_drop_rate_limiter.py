"""Deterministic arrival traces, including bursts and scheduler jitter."""

import math
import random

import pytest

from inference.core.interfaces.camera import utils
from inference.core.interfaces.camera.utils import FPSLimiterStrategy, limit_frame_rate


def run_trace(
    monkeypatch, arrivals, fps, strategy=FPSLimiterStrategy.DROP, processing_seconds=0.0
):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(utils.time, "sleep", sleep)

    def frames():
        for index, arrival in enumerate(arrivals):
            clock[0] = max(clock[0], arrival)
            yield index

    accepted = []
    for index in limit_frame_rate(frames(), fps, strategy):
        accepted.append((index, clock[0]))
        clock[0] += processing_seconds
    return accepted, sleeps


@pytest.mark.parametrize("fps", [15, 30])
def test_drop_tracks_30fps_jittered_source_without_phase_reset(monkeypatch, fps):
    rng = random.Random(47)
    arrivals = [i / 30 + rng.uniform(-0.006, 0.006) for i in range(3600)]
    accepted, sleeps = run_trace(monkeypatch, arrivals, fps)
    duration = arrivals[-1] - arrivals[0]
    assert (
        fps * duration - 2
        <= len(accepted)
        <= fps * duration + max(2, math.ceil(fps * 0.2))
    )
    assert not sleeps
    if fps == 30:
        assert len(accepted) == len(arrivals)


def test_drop_exact_nominal_30fps_does_not_lose_rounding_boundary_frames(monkeypatch):
    arrivals = [i / 30 for i in range(9001)]
    accepted, _ = run_trace(monkeypatch, arrivals, 30)
    assert len(accepted) == len(arrivals)


def test_drop_long_stall_cannot_accumulate_catch_up_burst(monkeypatch):
    arrivals = [0.0] * 100 + [3600.0] * 100
    accepted, sleeps = run_trace(monkeypatch, arrivals, 30)
    assert [index for index, _ in accepted] == list(range(6)) + list(range(100, 106))
    assert not sleeps


def test_drop_bursty_trace_obeys_rate_bound_in_every_accepted_window(monkeypatch):
    arrivals = [i / 120 for i in range(1201) for _ in range(3)]
    accepted, _ = run_trace(monkeypatch, arrivals, 30)
    for start, (_, start_time) in enumerate(accepted):
        for end in range(start, len(accepted)):
            elapsed = accepted[end][1] - start_time
            # Tiny assertion tolerance accounts only for float arithmetic;
            # the limiter itself grants no arbitrary timing epsilon.
            assert end - start + 1 <= 6 + elapsed * 30 + 1e-9
    assert 299 <= len(accepted) <= 306


def test_drop_slow_source_preserves_order_and_all_frames(monkeypatch):
    accepted, sleeps = run_trace(monkeypatch, [i / 10 for i in range(100)], 30)
    assert [index for index, _ in accepted] == list(range(100))
    assert not sleeps


def test_wait_retains_minimum_spacing_without_drops_or_jitter_credit(monkeypatch):
    accepted, sleeps = run_trace(monkeypatch, [0.0] * 100, 30, FPSLimiterStrategy.WAIT)
    assert [index for index, _ in accepted] == list(range(100))
    assert len(sleeps) == 99
    assert all(math.isclose(delay, 1 / 30) for delay in sleeps)
    assert math.isclose(accepted[-1][1], 99 / 30)


@pytest.mark.parametrize("group_size", [3, 4])
@pytest.mark.parametrize("processing_seconds", [0.0, 0.009, 0.010, 0.011])
def test_drop_grouped_arrivals_fit_bounded_burst_window(
    monkeypatch, group_size, processing_seconds
):
    # The 200 ms window at 30 FPS accommodates these bounded decoder bursts.
    groups = 300
    arrivals = [
        group * group_size / 30 for group in range(groups) for _ in range(group_size)
    ]
    accepted, sleeps = run_trace(
        monkeypatch, arrivals, 30, processing_seconds=processing_seconds
    )
    assert len(accepted) == len(arrivals)
    assert not sleeps


@pytest.mark.parametrize(
    "fps,capacity", [(0.5, 2), (5, 2), (15, 3), (30, 6), (100, 20)]
)
def test_drop_capacity_scales_with_rate_and_stays_bounded(monkeypatch, fps, capacity):
    accepted, _ = run_trace(monkeypatch, [0.0] * 100, fps)
    assert len(accepted) == capacity


def test_drop_callback_reports_only_discarded_items(monkeypatch):
    monkeypatch.setattr(utils.time, "monotonic", lambda: 0.0)
    discarded = []
    accepted = list(
        limit_frame_rate(
            range(10), 30, FPSLimiterStrategy.DROP, on_frame_dropped=discarded.append
        )
    )
    assert accepted == list(range(6))
    assert discarded == list(range(6, 10))


def test_multiplex_reports_each_frame_in_discarded_batch(monkeypatch):
    from types import SimpleNamespace

    reported = []
    sources = [
        SimpleNamespace(
            source_id=source_id,
            record_frame_dropped=lambda frame, cause: reported.append(
                (frame.source_id, frame.frame_id, cause)
            ),
        )
        for source_id in (0, 1)
    ]
    batches = [
        [SimpleNamespace(source_id=s, frame_id=i) for s in (0, 1)] for i in range(5)
    ]
    monkeypatch.setattr(utils.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        utils,
        "_prepare_video_sources",
        lambda **kwargs: utils.VideoSources(sources, [False, False], []),
    )
    monkeypatch.setattr(utils, "_multiplex_videos", lambda **kwargs: iter(batches))
    accepted = list(
        utils.multiplex_videos(
            sources,
            max_fps=30,
            limiter_strategy=FPSLimiterStrategy.DROP,
        )
    )
    # Existing aggregate-rate contract divides 30 FPS over the two sources.
    assert len(accepted) == 3
    assert reported == [(s, i, "FPS limiter") for i in (3, 4) for s in (0, 1)]
