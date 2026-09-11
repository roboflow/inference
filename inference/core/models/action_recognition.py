"""Turning a model's window-local answer into a clip-wide timeline.

A model indexes its segments into the list of frames it was handed, never
into the video. Both callers of an action recognition model — the HTTP
adapter and the streaming workflow block — therefore run the same four steps
before they can merge: clamp the indices into the window, map each one to the
frame number it came from, resolve a class id, and union the result. That is
this module.
"""

from typing import Any, List, Optional, Sequence

from inference.core.entities.responses.action_recognition import (
    ActionRecognitionPrediction,
)
from inference_models.models.base.action_recognition import merge_segment


def merge_window_segments(
    timeline: List[ActionRecognitionPrediction],
    frame_numbers: Sequence[int],
    segments: List[Any],
    id_vocabulary: Optional[List[str]],
    stride: float,
    class_filter: Optional[List[str]] = None,
) -> None:
    """Union one window's segments into ``timeline``, in place.

    ``frame_numbers`` holds the source frame each sampled index came from, in
    the order the model saw them. ``class_filter`` drops classes the caller
    did not ask for. Classes outside ``id_vocabulary`` report ``-1``, which is
    what an open-vocabulary answer gets.
    """
    sample_count = len(frame_numbers)
    if sample_count == 0:
        return
    class_ids = (
        {class_name: index for index, class_name in enumerate(id_vocabulary)}
        if id_vocabulary is not None
        else {}
    )
    for segment in segments:
        class_name = segment.class_name
        if class_filter is not None and class_name not in class_filter:
            continue
        start_index = min(sample_count - 1, max(0, int(segment.start_frame_idx)))
        end_index = min(sample_count - 1, max(0, int(segment.end_frame_idx)))
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        merge_segment(
            timeline=timeline,
            segment=ActionRecognitionPrediction(
                start_frame_idx=frame_numbers[start_index],
                end_frame_idx=frame_numbers[end_index],
                class_name=class_name,
                class_id=class_ids.get(class_name, -1),
            ),
            stride=stride,
        )
