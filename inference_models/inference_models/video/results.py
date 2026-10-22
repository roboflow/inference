"""Package-owned video results and shared temporal coordinate conversion."""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from inference_models.errors import VideoProcessingLimitError
from inference_models.models.base.action_recognition import merge_segment


@dataclass(frozen=True)
class ActionRecognitionVideoPrediction:
    """One action in inclusive, zero-based source-video frame coordinates.

    Attributes:
        start_frame_idx (int): First source frame in the range.
        end_frame_idx (int): Last source frame in the range.
        class_name (str): Predicted label.
        class_id (int): Model vocabulary index, or -1 for open vocabulary.
        confidence (Optional[float]): Score, absent for unscored models.
    """

    start_frame_idx: int
    end_frame_idx: int
    class_name: str
    class_id: int
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        """Return a plain mapping, omitting absent confidence.

        Returns:
            dict: Source coordinates, label, class ID and optional score.
        """
        result = asdict(self)
        if self.confidence is None:
            result.pop("confidence")
        return result


@dataclass(frozen=True)
class ActionRecognitionProgress:
    """Snapshot after one processed window, including skipped short windows.

    Attributes:
        windows_processed (int): Number of windows decoded or skipped so far.
        windows_total (int): Number of planned windows.
        windows_classified (int): Number of completed model calls.
        frames_decoded (int): Observed sequential source-frame count.
        source_fps (float): Nominal source FPS.
        frame_count (int): Header count, corrected when early EOF is observed.
    """

    windows_processed: int
    windows_total: int
    windows_classified: int
    frames_decoded: int
    source_fps: float
    frame_count: int


@dataclass(frozen=True)
class ActionRecognitionWindowResult:
    """Incremental output for one window; no history or frame pixels retained.

    Attributes:
        predictions (Tuple[ActionRecognitionVideoPrediction, ...]): Filtered,
            unmerged source-video spans. Windows can overlap.
        candidates (Optional[Tuple[ActionRecognitionVideoPrediction, ...]]):
            Unfiltered, unmerged scored spans when requested; otherwise None.
        progress (ActionRecognitionProgress): Processing counts and source facts.
        confidence_threshold (Optional[float]): Effective global threshold.
        per_class_confidence_thresholds (Optional[Dict[str, float]]): Effective
            thresholds when per-class recommendations apply.
        span_semantics (str): Model-declared meaning of the spans.
    """

    predictions: Tuple[ActionRecognitionVideoPrediction, ...]
    candidates: Optional[Tuple[ActionRecognitionVideoPrediction, ...]]
    progress: ActionRecognitionProgress
    confidence_threshold: Optional[float]
    per_class_confidence_thresholds: Optional[Dict[str, float]]
    span_semantics: str
    _merge_strides: Tuple[float, ...] = field(repr=False)

    def merge_into(self, timeline: List[ActionRecognitionVideoPrediction]) -> None:
        """Merge filtered spans into a caller-owned timeline in place.

        Args:
            timeline (List[ActionRecognitionVideoPrediction]): Accumulated spans.
                The caller is responsible for bounding retained history.
        """
        for prediction, stride in zip(self.predictions, self._merge_strides):
            merge_segment(timeline=timeline, segment=prediction, stride=stride)


@dataclass(frozen=True)
class ActionRecognitionVideoResult:
    """Collected video timeline and the metadata needed to interpret it.

    Attributes:
        timeline (List[ActionRecognitionVideoPrediction]): Sorted, merged spans.
        source_fps (float): Nominal source FPS for converting indices to seconds.
        frame_count (int): Source count, corrected when early EOF is observed.
        windows_classified (int): Completed model calls.
        windows_processed (int): Processed windows, including skipped windows.
        span_semantics (str): Model-declared meaning of the spans.
        confidence_threshold (Optional[float]): Effective global threshold.
        per_class_confidence_thresholds (Optional[Dict[str, float]]): Effective
            per-class thresholds, when applicable.
        candidates (Optional[List[ActionRecognitionVideoPrediction]]): Unmerged
            scored spans, including those below the selected threshold.
    """

    timeline: List[ActionRecognitionVideoPrediction]
    source_fps: float
    frame_count: int
    windows_classified: int
    windows_processed: int
    span_semantics: str
    confidence_threshold: Optional[float]
    per_class_confidence_thresholds: Optional[Dict[str, float]]
    candidates: Optional[List[ActionRecognitionVideoPrediction]]


def _map_window_segments(
    *,
    frame_numbers: Sequence[int],
    segments: Sequence[Any],
    id_vocabulary: Optional[List[str]],
    stride: float,
    class_filter: Optional[List[str]] = None,
    frame_limit: Optional[int] = None,
    sample_stride: Optional[float] = None,
    sample_start_frame: Optional[float] = None,
) -> Iterator[Tuple[ActionRecognitionVideoPrediction, float]]:
    """Map each span and preserve its continuous/discrete merge-gap rule."""
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
        prediction = ActionRecognitionVideoPrediction(
            start_frame_idx=start_frame,
            end_frame_idx=end_frame,
            class_name=class_name,
            class_id=class_ids.get(class_name, -1),
            confidence=getattr(segment, "confidence", None),
        )
        yield prediction, merge_stride


def merge_window_segments(
    timeline: list,
    *,
    frame_numbers: Sequence[int],
    segments: Sequence[Any],
    id_vocabulary: Optional[List[str]],
    stride: float,
    class_filter: Optional[List[str]] = None,
    frame_limit: Optional[int] = None,
    sample_stride: Optional[float] = None,
    merge: bool = True,
    sample_start_frame: Optional[float] = None,
    prediction_factory: Callable = ActionRecognitionVideoPrediction,
) -> None:
    """Map a model window into a timeline using caller-compatible entities.

    Args:
        timeline (list): Source-video timeline updated in place.
        frame_numbers (Sequence[int]): Actual selected source-frame indices.
        segments (Sequence[Any]): Model spans in sampled-window coordinates.
        id_vocabulary (Optional[List[str]]): Model vocabulary for assigning IDs.
        stride (float): Source-frame merge gap for discrete spans.
        class_filter (Optional[List[str]]): Optional allowed labels.
        frame_limit (Optional[int]): Exclusive observed source endpoint.
        sample_stride (Optional[float]): Continuous sample-clock interval.
        merge (bool): Union compatible spans; False appends raw candidates.
        sample_start_frame (Optional[float]): Intended streaming clock origin.
        prediction_factory (Callable): Boundary conversion to a caller's entity.
            The default produces package-owned video predictions.
    """
    mapped = _map_window_segments(
        frame_numbers=frame_numbers,
        segments=segments,
        id_vocabulary=id_vocabulary,
        stride=stride,
        class_filter=class_filter,
        frame_limit=frame_limit,
        sample_stride=sample_stride,
        sample_start_frame=sample_start_frame,
    )
    for prediction, merge_stride in mapped:
        converted = prediction_factory(**prediction.to_dict())
        if merge:
            merge_segment(timeline=timeline, segment=converted, stride=merge_stride)
        else:
            timeline.append(converted)


def _json_size(value) -> int:
    size = sum(
        len(chunk) for chunk in json.JSONEncoder(ensure_ascii=True).iterencode(value)
    )
    return size


class _ResultBudget:
    """Bound retained candidates, timeline entries and serialized result data."""

    def __init__(self, limits):
        self.limits = limits
        self.candidate_count = 0
        self.candidate_bytes = 2

    def check_candidate_count(self, additional: int) -> None:
        if self.candidate_count + additional > self.limits.max_candidates:
            raise VideoProcessingLimitError(
                f"Action-recognition candidates exceed the limit of {self.limits.max_candidates}. "
                "Use fewer classes, disable candidates, consume windows incrementally, "
                "or raise max_candidates."
            )

    def add_candidates(self, candidates) -> None:
        self.check_candidate_count(len(candidates))
        for candidate in candidates:
            self.candidate_bytes += _json_size(candidate.to_dict()) + 2
            self._check_bytes(self.candidate_bytes)
        self.candidate_count += len(candidates)

    def check_response(self, timeline, metadata) -> None:
        if len(timeline) > self.limits.max_timeline_entries:
            raise VideoProcessingLimitError(
                f"Action-recognition timeline exceeds the limit of {self.limits.max_timeline_entries} entries. "
                "Consume windows incrementally or raise max_timeline_entries."
            )
        size = self.candidate_bytes + _json_size(metadata) + 64
        self._check_bytes(size)
        for prediction in timeline:
            size += _json_size(prediction.to_dict()) + 2
            self._check_bytes(size)

    def _check_bytes(self, size):
        if size > self.limits.max_result_bytes:
            raise VideoProcessingLimitError(
                f"Action-recognition result exceeds the limit of {self.limits.max_result_bytes} bytes. "
                "Consume windows incrementally or raise max_result_bytes."
            )
