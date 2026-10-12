"""Copied verbatim from ``inference.core.models.action_recognition``.

Pure logic: it unions one sliding window's segments into a timeline. Its only
dependencies are ``ActionRecognitionPrediction`` (re-exported by
``execution_engine.entities.base``) and ``inference_models``' ``merge_segment``
- an allowed dependency - so the copy carries no server coupling.

Pinned to the original by
``tests/workflows/unit_tests/utils/test_action_recognition_copy.py``, which
drives BOTH implementations over the same inputs: if you change one, change
both.
"""

import math
from typing import Any, List, Optional, Sequence

from roboflow_workflows.execution_engine.entities.base import (
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
        if getattr(segment, "end_exclusive", False):
            # Neural span heads return continuous, half-open sample coordinates.
            # The public timeline uses inclusive source-frame indices.
            step = stride if sample_stride is None else sample_stride
            origin = (
                frame_numbers[0] if sample_start_frame is None else sample_start_frame
            )
            sample_limit = (
                sample_count if frame_limit is None else (frame_limit - origin) / step
            )
            start = max(0.0, min(float(segment.start_frame_idx), sample_limit))
            end = max(0.0, min(float(segment.end_frame_idx), sample_limit))
            limit = (
                frame_limit
                if frame_limit is not None
                else math.ceil(
                    frame_numbers[-1] + step
                    if sample_start_frame is None
                    else origin + sample_count * step
                )
            )
            start_frame = max(math.floor(origin), math.floor(origin + start * step))
            end_frame = min(limit - 1, math.ceil(origin + end * step) - 1)
            if end <= start or end_frame < start_frame:
                continue
            merge_stride = 1
        else:
            start_index = min(sample_count - 1, max(0, int(segment.start_frame_idx)))
            end_index = min(sample_count - 1, max(0, int(segment.end_frame_idx)))
            if start_index > end_index:
                start_index, end_index = end_index, start_index
            start_frame, end_frame = (
                frame_numbers[start_index],
                frame_numbers[end_index],
            )
            merge_stride = stride
        prediction = ActionRecognitionPrediction(
            start_frame_idx=start_frame,
            end_frame_idx=end_frame,
            class_name=class_name,
            class_id=class_ids.get(class_name, -1),
            confidence=getattr(segment, "confidence", None),
        )
        if merge:
            merge_segment(timeline=timeline, segment=prediction, stride=merge_stride)
        else:
            timeline.append(prediction)
