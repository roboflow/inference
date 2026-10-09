from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction,
    VideoSampling,
    merge_segment,
    plan_windows,
)


def test_overlapping_end_aligned_windows_keep_short_clip_fps():
    sampling = VideoSampling(
        window_seconds=1,
        sample_fps=4,
        min_frames=1,
        max_frames=4,
        overlap_frames=2,
        end_aligned=True,
        fixed_sample_fps=True,
    )
    windows = plan_windows(9, 4, sampling)
    assert [window.frame_indices for window in windows] == [
        (0, 1, 2, 3),
        (2, 3, 4, 5),
        (4, 5, 6, 7),
        (5, 6, 7, 8),
    ]
    short = plan_windows(1, 30, sampling)
    assert short[0].frame_indices == (0,)
    assert short[0].sample_fps == 4


def test_class_union_keeps_maximum_confidence():
    timeline = [
        ActionRecognitionPrediction(0, 2, "a", 0.9),
        ActionRecognitionPrediction(1, 3, "b", 0.7),
    ]
    merge_segment(timeline, ActionRecognitionPrediction(2, 4, "a", 0.6), stride=0)
    assert len(timeline) == 2
    assert timeline[-1] == ActionRecognitionPrediction(0, 4, "a", 0.9)
