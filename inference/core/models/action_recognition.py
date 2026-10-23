"""Adapt shared package temporal conversion to existing response entities."""

from typing import Any, List, Optional, Sequence

from inference.core.entities.responses.action_recognition import (
    ActionRecognitionPrediction,
)
from inference_models.video.results import (
    merge_window_segments as _merge_window_segments,
)


def merge_window_segments(
    timeline: List[ActionRecognitionPrediction],
    frame_numbers: Sequence[int],
    segments: List[Any],
    id_vocabulary: Optional[List[str]],
    stride: float,
    class_filter: Optional[List[str]] = None,
    frame_limit: Optional[int] = None,
    sample_stride: Optional[float] = None,
    merge: bool = True,
    sample_start_frame: Optional[float] = None,
) -> None:
    """Union one window's segments into ``timeline``, in place.

    ``frame_numbers`` holds the source frame each sampled index came from, in
    the order the model saw them. ``class_filter`` drops classes the caller
    did not ask for. Classes outside ``id_vocabulary`` report ``-1``, which is
    what an open-vocabulary answer gets.

    ``sample_start_frame`` anchors continuous spans to the intended sample
    clock when streaming delivery selects a slightly later source frame.

    Args:
        timeline (List[ActionRecognitionPrediction]): Timeline updated in place.
        frame_numbers (Sequence[int]): Selected source indices in sample order.
        segments (List[Any]): Model spans in sampled-frame coordinates.
        id_vocabulary (Optional[List[str]]): Model classes used to assign IDs.
        stride (float): Maximum source-frame gap for merging discrete spans.
        class_filter (Optional[List[str]]): Classes retained in the timeline.
        frame_limit (Optional[int]): Exclusive source-frame endpoint.
        sample_stride (Optional[float]): Source-frame interval for continuous spans.
        merge (bool): Merge overlapping spans instead of appending each span.
        sample_start_frame (Optional[float]): Intended origin for continuous spans.
    """
    _merge_window_segments(
        timeline=timeline,
        frame_numbers=frame_numbers,
        segments=segments,
        id_vocabulary=id_vocabulary,
        stride=stride,
        class_filter=class_filter,
        frame_limit=frame_limit,
        sample_stride=sample_stride,
        merge=merge,
        sample_start_frame=sample_start_frame,
        prediction_factory=ActionRecognitionPrediction,
    )
