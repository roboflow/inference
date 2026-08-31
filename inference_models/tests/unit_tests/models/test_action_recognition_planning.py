import pytest

from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction,
    VideoSampling,
    merge_segment,
    plan_windows,
)


class _Segment:
    def __init__(self, start, end, class_name):
        self.start_frame_idx = start
        self.end_frame_idx = end
        self.class_name = class_name

    def __eq__(self, other):
        return (
            self.start_frame_idx,
            self.end_frame_idx,
            self.class_name,
        ) == (other.start_frame_idx, other.end_frame_idx, other.class_name)


def test_windows_tile_from_the_start_and_drop_the_remainder() -> None:
    # 30 s at 10 fps, 8 s windows -> three whole windows, 6 s left over.
    windows = plan_windows(
        frame_count=300,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0),
    )

    assert len(windows) == 3
    assert windows[0].frame_indices[0] == 0
    assert windows[1].frame_indices[0] == 80
    assert windows[2].frame_indices[0] == 160
    # The tail beyond the last whole window is not classified.
    assert max(windows[-1].frame_indices) < 300


def test_each_window_samples_at_the_declared_rate() -> None:
    windows = plan_windows(
        frame_count=300,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0),
    )

    # 8 s at 2 fps is 16 frames, every fifth source frame.
    assert len(windows[0].frame_indices) == 16
    gaps = {
        later - earlier
        for earlier, later in zip(
            windows[0].frame_indices, windows[0].frame_indices[1:]
        )
    }
    assert gaps == {5}


def test_a_clip_shorter_than_one_window_stays_whole() -> None:
    windows = plan_windows(
        frame_count=40,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )

    assert len(windows) == 1
    assert windows[0].frame_indices[0] == 0
    assert len(windows[0].frame_indices) == 8


def test_a_clip_below_the_minimum_frame_count_yields_nothing() -> None:
    windows = plan_windows(
        frame_count=5,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )

    assert windows == []


def test_sample_rate_never_exceeds_the_source_rate() -> None:
    windows = plan_windows(
        frame_count=60,
        source_fps=2.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=30.0),
    )

    assert len(windows[0].frame_indices) == len(set(windows[0].frame_indices))


@pytest.mark.parametrize("source_fps", [0.0, -1.0])
def test_an_unusable_frame_rate_yields_no_windows(source_fps) -> None:
    assert (
        plan_windows(frame_count=100, source_fps=source_fps, sampling=VideoSampling())
        == []
    )


def test_merge_unions_the_same_class_within_the_stride() -> None:
    timeline = [_Segment(0, 10, "walk")]

    merge_segment(timeline=timeline, segment=_Segment(12, 20, "walk"), stride=4)

    assert timeline == [_Segment(0, 20, "walk")]


def test_merge_keeps_ranges_separated_by_more_than_the_stride() -> None:
    timeline = [_Segment(0, 10, "walk")]

    merge_segment(timeline=timeline, segment=_Segment(40, 50, "walk"), stride=4)

    assert timeline == [_Segment(0, 10, "walk"), _Segment(40, 50, "walk")]


def test_merge_keeps_other_classes_apart() -> None:
    timeline = [_Segment(0, 10, "walk")]

    merge_segment(timeline=timeline, segment=_Segment(2, 8, "run"), stride=4)

    assert timeline == [_Segment(0, 10, "walk"), _Segment(2, 8, "run")]


def test_merge_never_shrinks_a_range() -> None:
    timeline = [_Segment(0, 100, "walk")]

    merge_segment(timeline=timeline, segment=_Segment(10, 20, "walk"), stride=4)

    assert timeline == [_Segment(0, 100, "walk")]
