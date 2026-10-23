"""Run action recognition on a local video using only inference-models."""

import math
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Union

import torch

from inference_models.entities import Confidence
from inference_models.errors import VideoInputError
from inference_models.models.base.action_recognition import (
    ActionRecognitionModel,
    effective_max_frame_side,
    plan_windows,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.video.control import VideoProcessingControl, VideoProcessingLimits
from inference_models.video.io import VideoDecodeState, probe_video, read_frame_windows
from inference_models.video.results import (
    ActionRecognitionProgress,
    ActionRecognitionVideoResult,
    ActionRecognitionWindowResult,
    _map_window_segments,
    _ResultBudget,
)

if TYPE_CHECKING:
    from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


class ActionRecognitionVideoPipeline:
    """Apply a loaded action-recognition model to local video files.

    The pipeline reuses the model's sampling and confidence contracts. It does
    not own or unload the model. Each run owns its decoder and result state.
    Source timing uses nominal FPS, not variable-rate frame timestamps.

    Args:
        model (Union[ActionRecognitionModel, Cosmos3EdgeReasoner]): Loaded V-JEPA
            or Cosmos action model; a bare Cosmos reasoner is wrapped using its
            package vocabulary and sampling metadata.
        limits (Optional[VideoProcessingLimits]): Explicit resource caps; omitted
            values use the package defaults. Limits never silently truncate data.
    """

    def __init__(
        self,
        model: "Union[ActionRecognitionModel, Cosmos3EdgeReasoner]",
        *,
        limits: Optional[VideoProcessingLimits] = None,
    ):
        if not hasattr(model, "video_sampling"):
            from inference_models.models.cosmos3.cosmos3_action_recognition import (
                Cosmos3EdgeActionRecognition,
            )
            from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
                Cosmos3EdgeReasoner,
            )

            if not isinstance(model, Cosmos3EdgeReasoner):
                raise ValueError(
                    "The loaded model does not support action recognition."
                )
            model = Cosmos3EdgeActionRecognition.from_reasoner(reasoner=model)

        self.model = model
        self.limits = limits or VideoProcessingLimits()

    @contextmanager
    def stream(
        self,
        path: Union[str, Path],
        *,
        class_names: Optional[List[str]] = None,
        confidence: Optional[Confidence] = None,
        include_candidates: bool = False,
        control: Optional[VideoProcessingControl] = None,
        on_progress: Optional[Callable[[ActionRecognitionProgress], None]] = None,
        on_source: Optional[Callable[[float, int], None]] = None,
    ) -> Iterator[Iterator[ActionRecognitionWindowResult]]:
        """Yield a context-managed iterator of independent window results.

        Use ``with pipeline.stream(path) as windows:`` so breaking iteration or
        a failing sink closes the decoder immediately. Results contain no pixels
        or prior history. Predictions are filtered but unmerged; use ``run`` to
        collect a final timeline or ``window.merge_into`` with bounded own state.

        Args:
            path (Union[str, Path]): Local video file, never a URL.
            class_names (Optional[List[str]]): Model vocabulary selection. An
                open-vocabulary model retains its existing interpretation.
            confidence (Optional[Confidence]): Numeric, "best" or "default";
                None selects "default". Ignored by unscored models.
            include_candidates (bool): Include unfiltered scored predictions.
            control (Optional[VideoProcessingControl]): Cancellation/deadline
                controller, or any object exposing the same ``check()`` method.
            on_progress (Optional[Callable]): Called after each processed window.
            on_source (Optional[Callable]): Receives nominal FPS and declared
                frame count before planning; may raise to enforce caller policy.

        Yields:
            Iterator[ActionRecognitionWindowResult]: Window events, including
            empty events for windows skipped because too few frames decoded.

        Raises:
            VideoInputError: The file or its source metadata is unusable.
            VideoProcessingLimitError: Planning, buffers or an event exceed limits.
            VideoProcessingCancelledError: Cooperative cancellation was requested.
            VideoProcessingTimeoutError: The processing deadline expired.
            ValueError: A supported confidence option is invalid.
        """
        iterator = self._iterate(
            path,
            class_names=class_names,
            confidence=confidence,
            include_candidates=include_candidates,
            control=control,
            on_progress=on_progress,
            on_source=on_source,
            collection_budget=None,
        )
        with closing(iterator):
            yield iterator

    def run(
        self,
        path: Union[str, Path],
        *,
        class_names: Optional[List[str]] = None,
        confidence: Optional[Confidence] = None,
        include_candidates: bool = False,
        control: Optional[VideoProcessingControl] = None,
        on_progress: Optional[Callable[[ActionRecognitionProgress], None]] = None,
        on_source: Optional[Callable[[float, int], None]] = None,
    ) -> ActionRecognitionVideoResult:
        """Collect a bounded, video-wide timeline from a local file.

        Args:
            path (Union[str, Path]): Local video file.
            class_names (Optional[List[str]]): Model vocabulary selection.
            confidence (Optional[Confidence]): Threshold, "best" or "default";
                None selects "default". Ignored by unscored models.
            include_candidates (bool): Retain unfiltered, unmerged scored spans.
            control (Optional[VideoProcessingControl]): Cooperative control.
            on_progress (Optional[Callable]): Receives progress per window.
            on_source (Optional[Callable]): Receives FPS/count before planning.

        Returns:
            ActionRecognitionVideoResult: Timeline, optional candidates, source
            facts, processing counts, effective thresholds and span semantics.

        Raises:
            VideoInputError: No usable source metadata or windows are available.
            VideoProcessingLimitError: Planning, buffers or collected output
                exceed limits; no partial collected result is returned.
            VideoProcessingCancelledError: The caller requests cancellation.
            VideoProcessingTimeoutError: The deadline expires.
            ValueError: A supported confidence option is invalid.
        """
        control = control or VideoProcessingControl()
        budget = _ResultBudget(self.limits)
        timeline = []
        candidates = (
            [] if include_candidates and self.model.supports_confidence else None
        )
        last = None
        iterator = self._iterate(
            path,
            class_names=class_names,
            confidence=confidence,
            include_candidates=include_candidates,
            control=control,
            on_progress=on_progress,
            on_source=on_source,
            collection_budget=budget,
        )
        with closing(iterator):
            for window in iterator:
                control.check()
                if candidates is not None:
                    budget.add_candidates(window.candidates)
                    candidates.extend(window.candidates)
                window.merge_into(timeline)
                budget.check_response(
                    timeline,
                    {
                        "per_class_confidence_thresholds": window.per_class_confidence_thresholds,
                    },
                )
                last = window

        if last is None:
            raise VideoInputError("The video produced no usable window plans.")

        timeline.sort(key=lambda entry: (entry.start_frame_idx, entry.class_id))
        progress = last.progress
        metadata = dict(
            source_fps=progress.source_fps,
            frame_count=progress.frame_count,
            windows_classified=progress.windows_classified,
            windows_processed=progress.windows_processed,
            confidence_threshold=last.confidence_threshold,
            per_class_confidence_thresholds=last.per_class_confidence_thresholds,
            span_semantics=last.span_semantics,
        )
        budget.check_response(timeline, metadata)
        control.check()
        result = ActionRecognitionVideoResult(
            timeline=timeline, candidates=candidates, **metadata
        )
        return result

    def _iterate(
        self,
        path,
        *,
        class_names,
        confidence,
        include_candidates,
        control,
        on_progress,
        on_source,
        collection_budget,
    ):
        control = control or VideoProcessingControl()
        control.check()
        model = self.model
        confidence = (
            _normalize_confidence(confidence) if model.supports_confidence else None
        )
        class_names = list(class_names) if class_names else None
        vocabulary = model.class_names or None
        threshold, per_class = _thresholds(
            model, confidence=confidence, vocabulary=vocabulary
        )
        source_fps, frame_count = probe_video(path=path, check_cancelled=control.check)
        if on_source is not None:
            on_source(source_fps, frame_count)
        control.check()
        sampling = model.video_sampling
        windows = plan_windows(
            frame_count=frame_count,
            source_fps=source_fps,
            sampling=sampling,
            max_windows=self.limits.max_windows,
            max_sampled_frames=self.limits.max_sampled_frames,
            check_cancelled=control.check,
        )
        scored_candidates = include_candidates and model.supports_confidence
        if collection_budget is not None and scored_candidates:
            estimate = model.estimate_candidate_count(
                sum(len(window.frame_indices) for window in windows),
                class_names=class_names,
            )
            if estimate is not None:
                collection_budget.check_candidate_count(estimate)

        limits = [
            min(
                frame_count,
                round(
                    window.frame_indices[0]
                    + (
                        window.duration_seconds
                        if window.duration_seconds is not None
                        else len(window.frame_indices) / window.sample_fps
                    )
                    * source_fps
                ),
            )
            for window in windows
        ]
        transform = getattr(model, "frame_storage_transform", None)
        decode_options = (
            {"frame_transform": transform}
            if transform is not None
            else {"max_frame_side": effective_max_frame_side(sampling)}
        )
        state = VideoDecodeState()
        classified = 0
        frames_iterator = read_frame_windows(
            path=path,
            windows=[window.frame_indices for window in windows],
            decode_state=state,
            window_end_frames=limits,
            check_cancelled=control.check,
            max_retained_frame_bytes=self.limits.max_retained_frame_bytes,
            **decode_options,
        )
        with closing(frames_iterator):
            for index, (window, planned_limit, frames) in enumerate(
                zip(windows, limits, frames_iterator)
            ):
                control.check()
                frame_limit = min(
                    planned_limit,
                    state.frame_count if state.reached_end else frame_count,
                )
                event_budget = _ResultBudget(self.limits)
                segments = []
                if len(frames) >= max(1, sampling.min_frames):
                    if scored_candidates:
                        estimate = model.estimate_candidate_count(
                            len(frames), class_names=class_names
                        )
                        if estimate is not None:
                            event_budget.check_candidate_count(estimate)
                    kwargs = {}
                    if (
                        model.supports_observed_duration
                        and window.duration_seconds is not None
                    ):
                        kwargs["duration_seconds"] = min(
                            window.duration_seconds,
                            (frame_limit - window.frame_indices[0]) / source_fps,
                        )
                    if model.supports_confidence:
                        kwargs["confidence"] = 0.0 if scored_candidates else confidence
                    segments = model.infer(
                        frames=frames,
                        class_names=class_names,
                        fps=window.sample_fps,
                        check_cancelled=control.check,
                        **kwargs,
                    )
                    classified += 1
                    control.check()

                event_budget.check_candidate_count(len(segments))
                mapping = dict(
                    frame_numbers=window.frame_indices[: len(frames)],
                    id_vocabulary=vocabulary,
                    stride=max(1.0, source_fps / window.sample_fps),
                    frame_limit=frame_limit,
                    sample_stride=source_fps / window.sample_fps,
                )
                candidates = None
                if scored_candidates:
                    if collection_budget is not None:
                        collection_budget.check_candidate_count(len(segments))
                    candidates = tuple(
                        item[0]
                        for item in _map_window_segments(segments=segments, **mapping)
                    )
                    event_budget.add_candidates(candidates)
                    segments = _filter_segments(
                        segments, threshold=threshold, per_class=per_class
                    )
                mapped = list(_map_window_segments(segments=segments, **mapping))
                predictions = tuple(item[0] for item in mapped)
                progress = ActionRecognitionProgress(
                    windows_processed=index + 1,
                    windows_total=len(windows),
                    windows_classified=classified,
                    frames_decoded=state.frame_count,
                    source_fps=source_fps,
                    frame_count=state.frame_count if state.reached_end else frame_count,
                )
                event_budget.check_response(
                    predictions,
                    {
                        "progress": asdict(progress),
                        "confidence_threshold": threshold,
                        "per_class_confidence_thresholds": per_class,
                    },
                )
                event = ActionRecognitionWindowResult(
                    predictions=predictions,
                    candidates=candidates,
                    progress=progress,
                    confidence_threshold=threshold,
                    per_class_confidence_thresholds=(
                        None if per_class is None else dict(per_class)
                    ),
                    span_semantics=getattr(model, "span_semantics", "instances"),
                    _merge_strides=tuple(item[1] for item in mapped),
                )
                del frames
                if on_progress is not None:
                    on_progress(progress)
                control.check()
                yield event
        control.check()


def _normalize_confidence(confidence):
    if confidence is None:
        return "default"
    if isinstance(confidence, str) and confidence in ("best", "default"):
        return confidence
    if not isinstance(confidence, bool) and isinstance(confidence, (int, float)):
        if math.isfinite(confidence) and 0 <= confidence <= 1:
            normalized = float(confidence)
            return normalized
    raise ValueError(
        'confidence must be "default", "best", or a number between 0 and 1'
    )


def _thresholds(model, *, confidence, vocabulary):
    if not model.supports_confidence:
        return None, None

    values = ConfidenceFilter(
        confidence=confidence,
        recommended_parameters=getattr(model, "recommended_parameters", None),
        default_confidence=getattr(model, "confidence_threshold", None),
    ).get_threshold(vocabulary or [])
    if isinstance(values, torch.Tensor):
        per_class = dict(zip(vocabulary, values.tolist()))
        return None, per_class
    return values, None


def _filter_segments(segments, *, threshold, per_class):
    thresholds = (
        torch.tensor(
            [per_class[segment.class_name] for segment in segments], dtype=torch.float32
        )
        if per_class is not None
        else threshold
    )
    keep = torch.ge(
        torch.tensor([segment.confidence for segment in segments], dtype=torch.float32),
        torch.as_tensor(thresholds, dtype=torch.float32),
    ).tolist()
    filtered = [segment for segment, accepted in zip(segments, keep) if accepted]
    return filtered
