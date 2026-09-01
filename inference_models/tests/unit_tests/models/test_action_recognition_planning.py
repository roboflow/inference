from dataclasses import dataclass

import pytest

from inference_models.models.base.action_recognition import (
    WHOLE_VIDEO_MODE,
    ActionRecognitionPrediction,
    VideoSampling,
    merge_segment,
    plan_windows,
)


@dataclass
class _Segment:
    start_frame_idx: int
    end_frame_idx: int
    class_name: str


def test_windows_tile_from_the_start_and_cover_the_remainder() -> None:
    # 30 s at 10 fps, 8 s windows -> three whole windows and a 6 s tail.
    windows = plan_windows(
        frame_count=300,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0),
    )

    assert len(windows) == 4
    assert [window.frame_indices[0] for window in windows] == [0, 80, 160, 240]
    # The tail is shorter than a window but still read, so the last frame of
    # the clip falls inside the last window.
    assert max(windows[-1].frame_indices) >= 290
    assert len(windows[-1].frame_indices) < len(windows[0].frame_indices)


def test_each_window_samples_at_the_declared_rate() -> None:
    windows = plan_windows(
        frame_count=300,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0),
    )

    # 8 s at 2 fps is 16 frames, every fifth source frame.
    assert len(windows[0].frame_indices) == 16
    assert [window.sample_fps for window in windows] == [2.0, 2.0, 2.0, 2.0]
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


def test_a_clip_below_the_minimum_frame_count_is_read_at_a_higher_rate() -> None:
    # Training clamps the count up to the floor rather than refusing, so a
    # clip it trained on stays servable.
    windows = plan_windows(
        frame_count=5,
        source_fps=10.0,
        sampling=VideoSampling(
            window_seconds=8.0, sample_fps=2.0, min_frames=4, max_frames=16
        ),
    )

    assert len(windows) == 1
    assert len(windows[0].frame_indices) == 4
    assert windows[0].sample_fps == pytest.approx(4 / 0.5)


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


def _whole_video(**kwargs) -> VideoSampling:
    return VideoSampling(mode=WHOLE_VIDEO_MODE, **kwargs)


def test_whole_video_mode_never_cuts_a_long_clip() -> None:
    # 60 s at 10 fps with a declared budget: the budget holds and the step
    # stretches, rather than the clip being tiled.
    windows = plan_windows(
        frame_count=600,
        source_fps=10.0,
        sampling=_whole_video(window_seconds=16.0, sample_fps=4.0, max_frames=64),
    )

    assert len(windows) == 1
    assert len(windows[0].frame_indices) == 64
    # The single window spans the clip rather than its first 16 s.
    assert windows[0].frame_indices[0] == 0
    assert windows[0].frame_indices[-1] >= 580
    # 64 frames over 60 s is well under the nominal 4 fps.
    assert windows[0].sample_fps == pytest.approx(64 / 60.0)


def test_whole_video_mode_holds_the_minimum_frame_count() -> None:
    windows = plan_windows(
        frame_count=4,
        source_fps=10.0,
        sampling=_whole_video(window_seconds=16.0, sample_fps=4.0, min_frames=4),
    )

    assert len(windows[0].frame_indices) == 4


def test_whole_video_mode_never_draws_more_frames_than_the_clip_holds() -> None:
    # A 2 fps source cannot supply 4 fps worth of distinct frames.
    windows = plan_windows(
        frame_count=20,
        source_fps=2.0,
        sampling=_whole_video(window_seconds=16.0, sample_fps=4.0),
    )

    assert len(windows[0].frame_indices) == 20
    assert len(set(windows[0].frame_indices)) == 20


def test_whole_video_holds_the_rate_when_no_budget_is_declared() -> None:
    # Zero-shot never trained on a budget, so the rate is what must hold: the
    # frame timestamps are built from it.
    windows = plan_windows(
        frame_count=600,
        source_fps=10.0,
        sampling=_whole_video(window_seconds=16.0, sample_fps=4.0),
    )

    assert len(windows) == 1
    # 60 s at 4 fps, not the 64 a trained window would have capped it to.
    assert len(windows[0].frame_indices) == 240
    # Exactly the nominal rate, not a rate re-derived from the frame count:
    # the model builds its frame timestamps from this number.
    assert windows[0].sample_fps == 4.0


def test_whole_video_reports_the_nominal_rate_when_the_count_rounds() -> None:
    # 18.77 s at 4 fps rounds to 75 frames, and 75 / 18.77 is not 4.0. The
    # window still reports 4.0, because that is the rate it sampled at.
    windows = plan_windows(
        frame_count=563,
        source_fps=30.0,
        sampling=_whole_video(sample_fps=4.0),
    )

    assert len(windows[0].frame_indices) == 75
    assert windows[0].sample_fps == 4.0


def test_a_trained_model_reads_the_grid_training_sampled_on() -> None:
    # roboflow-train fixes the count, divides the interval by it, and takes
    # the first frame at or after each timestamp. At 29.97 fps that lands a
    # frame later than a rounded nominal grid does.
    windows = plan_windows(
        frame_count=600,
        source_fps=29.97,
        sampling=VideoSampling(max_frames=64),
    )

    assert windows[0].frame_indices[:5] == (0, 8, 15, 23, 30)
    assert windows[0].sample_fps == pytest.approx(4.0)


def test_a_trained_model_spans_a_short_clip_and_reports_the_rate_it_used() -> None:
    # 127 frames at 30 fps is 4.23 s, which rounds to 17 samples. Training
    # spreads them over the whole clip, so the rate is not the nominal 4.0.
    windows = plan_windows(
        frame_count=127,
        source_fps=30.0,
        sampling=VideoSampling(max_frames=64),
    )

    assert len(windows) == 1
    assert len(windows[0].frame_indices) == 17
    assert windows[0].sample_fps == pytest.approx(17 / (127 / 30.0))


def test_a_model_without_a_budget_keeps_the_nominal_grid() -> None:
    # Zero-shot has only its pretraining to match, and that used a plain 4 fps.
    # Reporting anything else moves every frame timestamp off what it saw.
    windows = plan_windows(
        frame_count=563,
        source_fps=30.0,
        sampling=_whole_video(sample_fps=4.0),
    )

    assert windows[0].frame_indices[:5] == (0, 7, 15, 22, 30)
    assert windows[0].sample_fps == 4.0


def test_a_trained_model_keeps_its_rate_when_the_source_is_slower() -> None:
    # Training picks the count from the recorded rate alone and repeats frames
    # a slow source cannot supply distinctly. Capping at the source rate would
    # hand the model a different count and a different stamp on every frame.
    windows = plan_windows(
        frame_count=20,
        source_fps=2.0,
        sampling=VideoSampling(max_frames=64),
    )

    assert len(windows[0].frame_indices) == 40
    assert windows[0].frame_indices[:8] == (0, 1, 1, 2, 2, 3, 3, 4)
    assert windows[0].sample_fps == pytest.approx(4.0)


def test_a_model_without_a_budget_still_caps_at_the_source_rate() -> None:
    # Zero-shot has no training to reproduce, so repeating frames buys nothing.
    windows = plan_windows(
        frame_count=20,
        source_fps=2.0,
        sampling=_whole_video(sample_fps=4.0),
    )

    assert len(windows[0].frame_indices) == 20
    assert windows[0].sample_fps == 2.0


def test_the_trained_grid_does_not_drift_on_binary_floats() -> None:
    # 0.4 s over 4 samples steps by 0.1 s, and (0.1 * 3) * 30 lands just above
    # 9.0 in binary floating point, which rounds the fourth frame up to 10.
    windows = plan_windows(
        frame_count=12,
        source_fps=30.0,
        sampling=VideoSampling(max_frames=64),
    )

    assert windows[0].frame_indices == (0, 3, 6, 9)


def test_a_clip_that_divides_evenly_gets_no_extra_window() -> None:
    # 24 s at 10 fps against 8 s windows leaves nothing over.
    windows = plan_windows(
        frame_count=240,
        source_fps=10.0,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0),
    )

    assert len(windows) == 3


def test_a_trained_model_reads_the_remainder_on_its_own_grid() -> None:
    # 20 s at 30 fps against a 16 s contract: one whole window and a 4 s
    # tail, which the trainer's grid spans at the rate it implies.
    windows = plan_windows(
        frame_count=600,
        source_fps=30.0,
        sampling=VideoSampling(max_frames=64),
    )

    assert len(windows) == 2
    assert len(windows[0].frame_indices) == 64
    assert windows[0].sample_fps == pytest.approx(4.0)
    assert len(windows[1].frame_indices) == 16
    assert windows[1].sample_fps == pytest.approx(4.0)


@pytest.mark.parametrize("window_seconds", [0.0, -4.0])
def test_a_sliding_model_without_a_window_is_rejected(window_seconds) -> None:
    # Reading it as one whole-clip sample would guess a window the model
    # never declared, and the guess decides what the model sees.
    with pytest.raises(ValueError, match="positive window"):
        plan_windows(
            frame_count=300,
            source_fps=10.0,
            sampling=VideoSampling(window_seconds=window_seconds),
        )


def test_whole_video_mode_ignores_the_window_entirely() -> None:
    # That mode never consults a window, so an absent one is not a problem.
    windows = plan_windows(
        frame_count=300,
        source_fps=10.0,
        sampling=_whole_video(window_seconds=0.0, sample_fps=2.0),
    )

    assert len(windows) == 1
