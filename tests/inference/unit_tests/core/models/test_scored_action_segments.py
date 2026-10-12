from inference.core.models.action_recognition import merge_window_segments
from inference_models.models.base.action_recognition import ActionRecognitionPrediction


def test_scored_half_open_spans_cover_source_frames_without_gap_filling():
    timeline = []
    candidates = [
        ActionRecognitionPrediction(0, 1, "a", 0.9, True),
        ActionRecognitionPrediction(2, 4, "a", 0.8, True),
    ]
    merge_window_segments(
        timeline, [0, 8, 15, 23], candidates, ["a"], stride=7.5, frame_limit=30
    )
    assert [(row.start_frame_idx, row.end_frame_idx) for row in timeline] == [
        (0, 7),
        (15, 29),
    ]
    assert [row.confidence for row in timeline] == [0.9, 0.8]
    raw = []
    merge_window_segments(
        raw,
        [0, 8, 15, 23],
        candidates * 2,
        ["a"],
        stride=7.5,
        frame_limit=30,
        merge=False,
    )
    assert len(raw) == 4
