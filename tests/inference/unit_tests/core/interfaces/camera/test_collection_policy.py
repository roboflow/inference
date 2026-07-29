"""Unit tests for AUTO/EVERY_FRAME collection policies and their dispatch.

Covers: mode resolution (explicit > tensor-flag default > legacy), the
EMA-driven adaptive window, bounded-staleness FIFO reads with the file
exemption, and the `_multiplex_videos` dispatch guarantee that policy=None
uses the legacy retrieval path untouched.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inference.core import env as core_env
from inference.core.interfaces.camera import collection_policy as cp
from inference.core.interfaces.camera import utils as camera_utils
from inference.core.interfaces.camera.collection_policy import (
    AdaptiveWindowController,
    CollectionPolicy,
    VideoProcessingMode,
    resolve_video_processing_mode,
)
from inference.core.interfaces.camera.exceptions import EndOfStreamError
from inference.core.interfaces.camera.utils import VideoSources, _multiplex_videos


def test_resolve_mode_explicit_argument_wins(monkeypatch) -> None:
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", True)

    resolved = resolve_video_processing_mode(explicit_mode="every_frame")

    assert resolved is VideoProcessingMode.EVERY_FRAME


def test_resolve_mode_defaults_to_auto_for_tensor_cohort(monkeypatch) -> None:
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", True)

    resolved = resolve_video_processing_mode(explicit_mode=None)

    assert resolved is VideoProcessingMode.AUTO


def test_resolve_mode_preserves_legacy_behavior_outside_tensor_cohort(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", False)

    resolved = resolve_video_processing_mode(explicit_mode=None)

    assert resolved is None


def test_resolve_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        resolve_video_processing_mode(explicit_mode="turbo")


@pytest.mark.parametrize("alias", ["legacy", "none", "LEGACY", "None"])
def test_resolve_mode_legacy_alias_overrides_tensor_cohort_default(
    monkeypatch, alias: str
) -> None:
    # given - the tensor cohort, where the implicit default is AUTO
    monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", True)

    resolved = resolve_video_processing_mode(explicit_mode=alias)

    # then - the escape hatch forces the legacy path anyway
    assert resolved is None


def test_adaptive_window_starts_at_initial_value() -> None:
    controller = AdaptiveWindowController(initial_window=0.005, clock=lambda: 0.0)

    assert controller.on_collection_start() == 0.005


def test_adaptive_window_tracks_execution_gap() -> None:
    # given - a fake clock advanced manually
    now = {"value": 0.0}
    controller = AdaptiveWindowController(clock=lambda: now["value"])

    # when - first round collects frames, execution takes 100 ms
    controller.on_collection_start()
    controller.on_collection_end(collected_any_frame=True)
    now["value"] += 0.1
    window = controller.on_collection_start()

    # then - first gap sample seeds the EMA directly: 0.2 * 100 ms = 20 ms
    assert controller.execution_gap_ema == pytest.approx(0.1)
    assert window == pytest.approx(0.02)


def test_adaptive_window_is_clamped_to_bounds() -> None:
    now = {"value": 0.0}
    controller = AdaptiveWindowController(clock=lambda: now["value"])

    controller.on_collection_end(collected_any_frame=True)
    now["value"] += 10.0  # absurdly slow execution
    upper = controller.on_collection_start()
    controller.on_collection_end(collected_any_frame=True)
    now["value"] += 0.000001  # near-instant execution
    controller.on_collection_end(collected_any_frame=True)
    # drive the EMA down with repeated near-zero gaps
    for _ in range(64):
        controller.on_collection_start()
        controller.on_collection_end(collected_any_frame=True)
        now["value"] += 0.000001
    lower = controller.on_collection_start()

    assert upper == pytest.approx(cp.MAX_COLLECTION_WINDOW_SECONDS)
    assert lower == pytest.approx(cp.MIN_COLLECTION_WINDOW_SECONDS)


def test_adaptive_window_ignores_empty_rounds() -> None:
    now = {"value": 0.0}
    controller = AdaptiveWindowController(clock=lambda: now["value"])

    controller.on_collection_start()
    controller.on_collection_end(collected_any_frame=False)
    now["value"] += 5.0  # long idle gap after an EMPTY round
    controller.on_collection_start()

    # then - idle time never contaminates the execution-time estimate
    assert controller.execution_gap_ema is None


def test_arrival_estimator_uniform_arrivals() -> None:
    estimator = cp._SourceArrivalEstimator()
    start = datetime(2026, 1, 1, 12, 0, 0)

    for index in range(32):
        estimator.observe(
            frame_timestamp=start + timedelta(seconds=index / 15.0), now=100.0
        )

    assert estimator.period(now=100.0) == pytest.approx(1 / 15.0, rel=1e-6)


def test_arrival_estimator_bursty_arrivals_converge_on_mean_rate() -> None:
    # given - the consumer-camera pattern: clusters of 50 ms spacing with a
    # 450 ms encoder pause every 30 frames (true mean rate 15 fps-ish)
    estimator = cp._SourceArrivalEstimator()
    start = datetime(2026, 1, 1, 12, 0, 0)
    timestamp = start
    for index in range(64):
        gap = 0.450 if index % 30 == 29 else 0.050
        timestamp = timestamp + timedelta(seconds=gap)
        estimator.observe(frame_timestamp=timestamp, now=100.0)

    period = estimator.period(now=100.0)

    # then - close to the mean period, NOT the intra-burst 50 ms spacing
    mean_gap = (0.050 * 29 + 0.450) / 30.0
    assert period == pytest.approx(mean_gap, rel=0.15)
    assert period > 0.055


def test_arrival_estimator_resets_after_reconnect_gap() -> None:
    estimator = cp._SourceArrivalEstimator()
    start = datetime(2026, 1, 1, 12, 0, 0)
    for index in range(32):
        estimator.observe(
            frame_timestamp=start + timedelta(seconds=index / 15.0), now=100.0
        )

    # when - a 30 s reconnect gap, then only a few fresh samples
    rejoined = start + timedelta(seconds=40)
    for index in range(4):
        estimator.observe(
            frame_timestamp=rejoined + timedelta(seconds=index / 15.0), now=141.0
        )

    # then - the pre-gap history is discarded, too few samples to trust
    assert estimator.period(now=141.0) is None


def test_arrival_estimator_dormant_source_reports_no_period() -> None:
    estimator = cp._SourceArrivalEstimator()
    start = datetime(2026, 1, 1, 12, 0, 0)
    for index in range(32):
        estimator.observe(
            frame_timestamp=start + timedelta(seconds=index / 15.0), now=100.0
        )

    assert estimator.period(now=100.0) is not None
    # then - nothing seen for longer than the activity horizon -> excluded
    assert estimator.period(now=103.0) is None


def test_controller_rate_matches_when_arrival_period_known() -> None:
    now = {"value": 0.0}
    controller = AdaptiveWindowController(clock=lambda: now["value"])

    controller.on_collection_start()
    controller.on_collection_end(collected_any_frame=True)
    now["value"] += 0.042  # execution takes 42 ms
    window = controller.on_collection_start(minimum_arrival_period=1 / 15.0)

    # then - window = frame period - exec: rounds lock to the arrival rate
    assert window == pytest.approx(1 / 15.0 - 0.042, rel=1e-6)


def test_controller_floors_under_saturation_with_known_period() -> None:
    now = {"value": 0.0}
    controller = AdaptiveWindowController(clock=lambda: now["value"])

    controller.on_collection_end(collected_any_frame=True)
    now["value"] += 0.129  # x-model regime: exec far above the frame period
    window = controller.on_collection_start(minimum_arrival_period=1 / 15.0)

    assert window == pytest.approx(cp.MIN_COLLECTION_WINDOW_SECONDS)


def test_controller_caps_rate_matched_window() -> None:
    now = {"value": 0.0}
    controller = AdaptiveWindowController(clock=lambda: now["value"])

    controller.on_collection_end(collected_any_frame=True)
    now["value"] += 0.005  # near-instant execution, very slow source (2 fps)
    window = controller.on_collection_start(minimum_arrival_period=0.5)

    assert window == pytest.approx(cp.RATE_MATCHED_WINDOW_CAP_SECONDS)


def test_policy_minimum_period_tracks_fastest_source_and_skips_files() -> None:
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO, max_staleness=0.4)
    start = datetime.now()
    fast_frames = [
        SimpleNamespace(
            frame_id=index,
            source_id=0,
            frame_timestamp=start + timedelta(seconds=index / 30.0),
        )
        for index in range(32)
    ]
    slow_frames = [
        SimpleNamespace(
            frame_id=index,
            source_id=1,
            frame_timestamp=start + timedelta(seconds=index / 5.0),
        )
        for index in range(32)
    ]
    file_frames = [
        SimpleNamespace(
            frame_id=index,
            source_id=2,
            frame_timestamp=start + timedelta(seconds=index / 120.0),
        )
        for index in range(32)
    ]
    fast = _FakeSource(frames=fast_frames, is_file=False)
    slow = _FakeSource(frames=slow_frames, is_file=False)
    file_source = _FakeSource(frames=file_frames, is_file=True)

    for _ in range(32):
        policy.read_frame(source_ord=0, source=fast, timeout=0.1)
        policy.read_frame(source_ord=1, source=slow, timeout=0.1)
        policy.read_frame(source_ord=2, source=file_source, timeout=0.1)

    # then - the fastest LIVE source binds; the 120 fps file never counts
    assert policy.minimum_live_arrival_period() == pytest.approx(
        1 / 30.0, rel=0.05
    )


def test_policy_feeds_estimator_with_staleness_drained_frames() -> None:
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO, max_staleness=0.4)
    start = datetime.now() - timedelta(seconds=10)
    stale_frames = [
        SimpleNamespace(
            frame_id=index,
            source_id=0,
            frame_timestamp=start + timedelta(seconds=index / 15.0),
        )
        for index in range(32)
    ]
    source = _FakeSource(frames=stale_frames, is_file=False)

    frame = policy.read_frame(source_ord=0, source=source, timeout=0.1)

    # then - every drained arrival counted: period known despite 0 returns
    assert frame is None
    assert policy.minimum_live_arrival_period() == pytest.approx(
        1 / 15.0, rel=0.05
    )


def _fake_frame(age_seconds: float, frame_id: int = 1, source_id: int = 0):
    return SimpleNamespace(
        frame_id=frame_id,
        source_id=source_id,
        frame_timestamp=datetime.now() - timedelta(seconds=age_seconds),
    )


class _FakeSource:
    def __init__(self, frames, is_file):
        self._frames = list(frames)
        self._is_file = is_file

    def read_frame(self, timeout=None):
        if not self._frames:
            return None
        item = self._frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def describe_source(self):
        properties = (
            None if self._is_file is None else SimpleNamespace(is_file=self._is_file)
        )
        return SimpleNamespace(source_properties=properties)


def test_auto_policy_drops_stale_frames_for_live_sources() -> None:
    # given - two stale frames queued ahead of a fresh one
    dropped = []
    policy = CollectionPolicy(
        mode=VideoProcessingMode.AUTO,
        max_staleness=0.4,
        on_frame_dropped=dropped.append,
    )
    source = _FakeSource(
        frames=[
            _fake_frame(1.2, frame_id=1),
            _fake_frame(0.9, frame_id=2),
            _fake_frame(0.05, frame_id=3),
        ],
        is_file=False,
    )

    frame = policy.read_frame(source_ord=0, source=source, timeout=0.1)

    assert frame.frame_id == 3
    assert [f.frame_id for f in dropped] == [1, 2]
    assert policy.frames_dropped_on_staleness == {0: 2}


def test_auto_policy_never_drops_file_frames() -> None:
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO, max_staleness=0.4)
    source = _FakeSource(frames=[_fake_frame(10.0, frame_id=1)], is_file=True)

    frame = policy.read_frame(source_ord=0, source=source, timeout=0.1)

    assert frame.frame_id == 1
    assert policy.frames_dropped_on_staleness == {}


def test_auto_policy_treats_unknown_source_properties_as_file() -> None:
    # given - source metadata not resolved yet (initialising source)
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO, max_staleness=0.4)
    source = _FakeSource(frames=[_fake_frame(10.0, frame_id=1)], is_file=None)

    frame = policy.read_frame(source_ord=0, source=source, timeout=0.1)

    # then - never drop while liveness is unknown
    assert frame.frame_id == 1
    assert policy.frames_dropped_on_staleness == {}


def test_every_frame_policy_disables_staleness_budget() -> None:
    policy = CollectionPolicy(mode=VideoProcessingMode.EVERY_FRAME, max_staleness=0.4)
    source = _FakeSource(frames=[_fake_frame(10.0, frame_id=1)], is_file=False)

    frame = policy.read_frame(source_ord=0, source=source, timeout=0.1)

    assert frame.frame_id == 1
    assert policy.max_staleness is None


def test_policy_rejects_freshest_mode() -> None:
    with pytest.raises(ValueError):
        CollectionPolicy(mode=VideoProcessingMode.FRESHEST)


def test_policy_propagates_end_of_stream() -> None:
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO)
    source = _FakeSource(frames=[EndOfStreamError()], is_file=False)

    with pytest.raises(EndOfStreamError):
        policy.read_frame(source_ord=0, source=source, timeout=0.1)


def test_dropped_frame_callback_errors_never_break_reads() -> None:
    def raising_callback(frame):
        raise RuntimeError("sink exploded")

    policy = CollectionPolicy(
        mode=VideoProcessingMode.AUTO,
        max_staleness=0.4,
        on_frame_dropped=raising_callback,
    )
    source = _FakeSource(
        frames=[_fake_frame(1.0, frame_id=1), _fake_frame(0.0, frame_id=2)],
        is_file=False,
    )

    frame = policy.read_frame(source_ord=0, source=source, timeout=0.1)

    assert frame.frame_id == 2


def _ended_video_sources() -> VideoSources:
    # single fake source that ends immediately - drives the multiplex loop to
    # exactly one retrieval round
    source = _FakeSource(frames=[EndOfStreamError()], is_file=False)
    return VideoSources(
        all_sources=[source], allow_reconnection=[False], managed_sources=[]
    )


def test_multiplex_dispatch_uses_legacy_path_without_policy(monkeypatch) -> None:
    # given - spies on both retrieval methods
    legacy_spy = MagicMock(return_value=None)
    policy_spy = MagicMock(return_value=None)
    monkeypatch.setattr(
        camera_utils.VideoSourcesManager,
        "retrieve_frames_from_sources",
        legacy_spy,
    )
    monkeypatch.setattr(
        camera_utils.VideoSourcesManager,
        "retrieve_frames_from_sources_with_policy",
        policy_spy,
    )

    # when
    list(
        _multiplex_videos(
            video_sources=_ended_video_sources(),
            batch_collection_timeout=0.123,
            should_stop=lambda: False,
            on_reconnection_error=lambda *_: None,
        )
    )

    # then - policy machinery is not even touched, timeout forwarded verbatim
    legacy_spy.assert_called_with(batch_collection_timeout=0.123)
    policy_spy.assert_not_called()


def test_multiplex_dispatch_uses_policy_path_when_policy_given(monkeypatch) -> None:
    legacy_spy = MagicMock(return_value=None)
    policy_spy = MagicMock(return_value=None)
    monkeypatch.setattr(
        camera_utils.VideoSourcesManager,
        "retrieve_frames_from_sources",
        legacy_spy,
    )
    monkeypatch.setattr(
        camera_utils.VideoSourcesManager,
        "retrieve_frames_from_sources_with_policy",
        policy_spy,
    )
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO)

    list(
        _multiplex_videos(
            video_sources=_ended_video_sources(),
            batch_collection_timeout=None,
            should_stop=lambda: False,
            on_reconnection_error=lambda *_: None,
            collection_policy=policy,
        )
    )

    policy_spy.assert_called_with(collection_policy=policy)
    legacy_spy.assert_not_called()


def test_policy_retrieval_round_collects_and_registers_eos() -> None:
    # given - one live source with a fresh frame, one source at EOS
    fresh = _fake_frame(0.01, frame_id=7, source_id=0)
    source_with_frame = _FakeSource(frames=[fresh], is_file=False)
    ended_source = _FakeSource(frames=[EndOfStreamError()], is_file=False)
    video_sources = VideoSources(
        all_sources=[source_with_frame, ended_source],
        allow_reconnection=[False, False],
        managed_sources=[],
    )
    manager = camera_utils.VideoSourcesManager.init(
        video_sources=video_sources,
        should_stop=lambda: False,
        on_reconnection_error=lambda *_: None,
    )
    policy = CollectionPolicy(mode=VideoProcessingMode.AUTO, max_staleness=0.4)

    batch = manager.retrieve_frames_from_sources_with_policy(collection_policy=policy)

    assert batch == [fresh]
    assert manager.all_sources_ended() is False  # only one of two sources ended
    second_batch = manager.retrieve_frames_from_sources_with_policy(
        collection_policy=policy
    )
    assert second_batch == []  # ended source inactive, first source drained
