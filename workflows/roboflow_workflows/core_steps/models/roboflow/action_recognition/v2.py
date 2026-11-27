"""Stateful action recognition workflow block."""

import math
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4
from weakref import finalize

from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    normalise_class_names,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.entities import (
    _SharedTimeline,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    MAX_TRACKED_VIDEOS,
    ActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest as V1Manifest,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    _ActionRecognitionBookkeeping as V1Bookkeeping,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    _SampledFrame,
    logger,
)
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionResult,
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_RESULT_KIND,
    BOOLEAN_KIND,
    FLOAT_KIND,
    Selector,
)
from roboflow_workflows.prototypes.block import BlockResult, WorkflowBlockManifest
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from inference_models.entities import Confidence
from inference_models.models.base.action_recognition import WHOLE_VIDEO_MODE
from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models.models.base.action_recognition import VideoSampling

COVERAGE_HISTORY_SECONDS = 900

LONG_DESCRIPTION = """
Classify actions and events in a video stream. The block continuously samples
each stream into a sliding window and adds each model result to a cumulative
timeline. The first scheduled classification is due one stride after the stream
starts. By default, scheduled windows use the overlap recorded by the model.
With zero overlap, the stride equals the window length. Set a smaller stride
to overlap scheduled windows for finer range boundaries. Ranges can
overlap. Timeline ranges only grow when the block merges model evidence. The
block does not extend ranges provisionally. The timeline holds at most 5000
ranges per stream. A stream that passes that ceiling loses the ranges that
ended earliest, so its timeline covers recent history rather than all of it. When a stream provides no source
FPS, the block assumes 30 FPS and logs a warning.

The window length and the sample rate are part of the model, not block
settings: a fine-tuned model declares the values its training used, and other
models declare their defaults (a 16 second window at 4 frames per second).

predictions bundles timeline, analyzed_from_frame, analyzed_until_frame, and
error_status into a single action_recognition_result output for downstream
blocks such as Action Recognition Visualization. analyzed_from_frame and
analyzed_until_frame are the inclusive first and last source frames for which
the latest successful call could report an action, or null before the first
success. For models like Cosmos that return sampled-frame indices, they are the
first and last sampled frames. For models like V-JEPA that use the observed
window duration, analyzed_from_frame is the floored intended position of the
first sample and analyzed_until_frame is the source frame that triggered the
call. Models can therefore report different values for the same input frame. To
display file-processing progress when the frame count is known, use frame_number
and total_frames from the video metadata. Both marks update even if no actions
are detected. Earlier gaps can remain after failed, skipped, or non-overlapping
windows (continuous coverage without sub-sample gaps requires stride < window),
and overlapping windows can still revise earlier predictions. covered_frame_ranges
records recent successful windows, including quiet ones. source_fps is the
pinned source rate, and stream_generation changes when model state resets.

On a video file that reports its frame count, the last frame can trigger a
classification before the next scheduled call. This call uses the current
sliding window, so it can overlap the previous call more than scheduled windows
do. For example, with a 16-second window and a 16-second stride, a 20-second
file has calls at about 16 and 20 seconds. Their windows share about 12 seconds.
A file shorter than one stride can also trigger its first call on the last
frame. The buffer must contain the model's minimum number of sampled frames,
and the last frame must reach the block. A source such as InferencePipeline
with max_fps can drop it. Live streams and files without a usable frame count
get no extra call, so frames after the final scheduled call receive no result.

By default a classification runs inside the step, so the video waits for
the model. With background_inference, a live stream hands each window to a
worker thread and the step returns at once. The result reaches the timeline
on the stream's next frame after the call ends, and predictions.analyzed_until_frame
shows how far results go. The last call of a stream that stops is therefore
not reported. One call runs per stream at a time: a call that comes due
while the previous one has not finished waits, then takes the newest window.
When a call starts late, predictions.error_status names the frames no window covers
between it and the previous call. The step has one worker, so its streams
take turns: with several cameras on one step, each camera's results arrive
less often. Video files always classify inside the step, so every window is
analyzed and results are repeatable. A stream reads background_inference on
its first frame and keeps that mode until its state resets, for example when
a window input changes.

Use this block with InferencePipeline for full temporal behavior. Still-image
and HTTP execution do not provide a continuous stream. A single frame has no
temporal content, so these paths return an empty timeline.

This block serves fine-tuned models only. A model trained on whole videos
spans each clip in one call, and a stream has no end to span, so the block
refuses it and names the action recognition endpoint instead.

The class vocabulary is optional. Leave it empty to report every class the
model carries, or list classes to report a subset of them. When a model call
fails, predictions.error_status carries the error text for that frame and the stream
continues.

If required samples are missing, the block skips that window and reports
their source-frame indices in predictions.error_status. Later complete windows resume
inference. Source frames intentionally skipped by sampling are not missing.
"""


def _first_covered_frame(
    sample: _SampledFrame,
    *,
    supports_observed_duration: bool,
) -> int:
    if supports_observed_duration:
        return math.floor(sample.intended_frame_number)

    if sample.source_frame_number is not None:
        return sample.source_frame_number

    return math.ceil(sample.intended_frame_number - 1e-9)


@dataclass
class _WindowCall:
    """One window handed to the model, with what its merge needs later."""

    sampled: List[_SampledFrame]
    frame_limit: Optional[int]
    block_filter: Optional[List[str]]
    id_vocabulary: Optional[List[str]]
    sampling_stride: float
    supports_observed_duration: bool
    future: Optional[Future] = None

    def first_covered_frame(self) -> int:
        first_frame = _first_covered_frame(
            self.sampled[0],
            supports_observed_duration=self.supports_observed_duration,
        )

        return first_frame

    def last_covered_frame(self) -> int:
        if self.supports_observed_duration and self.frame_limit is not None:
            return self.frame_limit - 1

        return self.sampled[-1].source_frame_number


_WindowResult = List[ModelActionRecognitionPrediction]


class _WindowWorker:
    """Keep windows in FIFO order until one worker claims them.

    Cancelling a future submitted directly to ThreadPoolExecutor leaves its
    arguments in the pool's queue until the worker reaches it. Only submit
    the drain to the pool so discarding a window also releases its frames.
    """

    def __init__(self):
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="action-recognition"
        )
        self._lock = threading.Lock()
        self._waiting: OrderedDict[Future, Callable[[], _WindowResult]] = OrderedDict()
        self._draining = False
        self._closed = False

    def submit(self, classify: Callable[[], _WindowResult]) -> Future:
        future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("The action recognition worker is closed.")

            if not self._draining:
                self._pool.submit(self._drain)
                self._draining = True
            self._waiting[future] = classify

        return future

    def discard(self, future: Future) -> None:
        with self._lock:
            if self._waiting.pop(future, None) is not None:
                future.cancel()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._waiting:
                    self._draining = False
                    return

                future, classify = self._waiting.popitem(last=False)
                should_run = future.set_running_or_notify_cancel()

            try:
                if not should_run:
                    continue

                result = classify()
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)
            finally:
                # A failed future keeps this loop's frame through its traceback.
                # Drop each window's references before moving on or going idle.
                future = classify = result = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for future in self._waiting:
                future.cancel()
            self._waiting.clear()
            self._pool.shutdown(wait=False, cancel_futures=True)


@dataclass
class _ActionRecognitionBookkeeping(V1Bookkeeping):
    analyzed_from_frame: Optional[int] = None
    analyzed_until_frame: Optional[int] = None
    covered_frame_ranges: Tuple[Tuple[int, int], ...] = ()
    stream_generation: str = field(default_factory=lambda: uuid4().hex)
    tail_flushed: bool = False
    pending: Optional[_WindowCall] = None
    deferred_window_end: Optional[int] = None
    background: bool = False

    def record_successful_window(self, *, start: int, end: int) -> None:
        """Retain successful coverage across the display history and maximum delay."""
        self.analyzed_from_frame = start
        self.analyzed_until_frame = end
        assert self.source_fps is not None
        oldest = end - math.ceil(COVERAGE_HISTORY_SECONDS * self.source_fps)
        ranges = [pair for pair in self.covered_frame_ranges if pair[1] >= oldest]
        # Windows for each stream complete in source order.
        while ranges and start <= ranges[-1][1] + 1:
            start = min(start, ranges.pop()[0])

        self.covered_frame_ranges = (*ranges, (start, end))


class BlockManifest(V1Manifest):
    # The optional worker changes when results arrive. Both versions buffer
    # video and load the same local model with the same runtime restrictions.
    discover_work_operations = V1Manifest.discover_work_operations
    discover_dependent_resources = V1Manifest.discover_dependent_resources
    get_actual_restrictions = V1Manifest.get_actual_restrictions

    model_config = ConfigDict(
        json_schema_extra={
            **V1Manifest.model_config["json_schema_extra"],
            "version": "v2",
            "long_description": LONG_DESCRIPTION,
        }
    )
    type: Literal["roboflow_core/roboflow_action_recognition_model@v2"]
    stride_seconds: Union[Optional[float], Selector(kind=[FLOAT_KIND])] = Field(
        default=None,
        description=(
            "Time between scheduled classification calls. Leave empty to use "
            "the model's recorded overlap. A smaller stride increases overlap "
            "and runs the model more often. A video file's last frame can "
            "trigger an earlier call with more overlap."
        ),
        examples=[None, 2.0],
    )
    background_inference: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description=(
            "Classify live-stream windows on a worker thread, so the step "
            "returns at once and the video does not wait for the model. "
            "Results reach the timeline on a later frame. Video files always "
            "classify inside the step. A stream keeps the mode it starts with "
            "until its state resets."
        ),
        examples=[False, "$inputs.background_inference"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> str:
        # The engine closes the live worker on workflow shutdown and failed init.
        return ">=1.17.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[ACTION_RECOGNITION_RESULT_KIND],
            ),
        ]


class ActionRecognitionModelBlockV2(ActionRecognitionModelBlockV1):
    """Classify scheduled and final file windows, with optional live background inference."""

    def __init__(
        self,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
        model_manager: Optional[ModelsProvider] = None,
    ):
        super().__init__(
            api_key=api_key,
            step_execution_mode=step_execution_mode,
            model_manager=model_manager,
        )
        # A file stream classifies on the step thread while a live stream's
        # worker may be inside the same model.
        self._model_lock = threading.Lock()
        self._worker: Optional[_WindowWorker] = None
        self._worker_finalizer = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _clear_streams(self) -> None:
        for bookkeeping in self._video_bookkeeping.values():
            self._discard_pending_call(bookkeeping)
        super()._clear_streams()

    def _discard_pending_call(self, bookkeeping: _ActionRecognitionBookkeeping) -> None:
        if bookkeeping.pending is not None and self._worker is not None:
            self._worker.discard(bookkeeping.pending.future)

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_filter: Optional[List[str]] = None,
        stride_seconds: Optional[float] = None,
        confidence: Optional[Confidence] = None,
        background_inference: bool = False,
    ) -> BlockResult:
        """Process video frames using detection's confidence modes.

        Args:
            images: Batch of frames carrying video metadata.
            model_id: Action-recognition model identifier.
            class_filter: Optional subset of model classes.
            stride_seconds: Time between model calls.
            confidence: Numeric override, "best" recommendations, or the
                built-in default with "default" or None.
            background_inference: Classify live-stream windows on a worker
                thread instead of inside the step.

        Returns:
            One bundled ``predictions`` result per input frame.
        """
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        model = self._get_model(model_id=model_id)
        block_filter = normalise_class_names(class_filter) or None
        # A filter is not a vocabulary; only the model's own class list
        # carries ids. See the adapter for the zero-shot case this avoids.
        id_vocabulary = getattr(model, "class_names", None) or None
        video_sampling = getattr(model, "video_sampling", None) or VideoSampling()
        if video_sampling.mode == WHOLE_VIDEO_MODE:
            # Whole-video training fed one sample spanning each clip, and a
            # stream has no end to span. Any window this block picked would
            # sample differently from training, so it refuses instead.
            raise ValueError(
                f"Model {model_id} was trained on whole videos, so it cannot run "
                f"on a stream. Send whole clips to the action recognition "
                f"endpoint instead, or train the model with sliding windows."
            )
        results = []
        for image in images:
            results.append(
                self._process_frame(
                    model=model,
                    image=image,
                    block_filter=block_filter,
                    id_vocabulary=id_vocabulary,
                    video_sampling=video_sampling,
                    stride_seconds=stride_seconds,
                    confidence=confidence,
                    background_inference=bool(background_inference),
                )
            )
        return results

    def _process_frame(
        self,
        model,
        image: WorkflowImageData,
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        video_sampling: VideoSampling,
        stride_seconds: Optional[float],
        confidence: Optional[Confidence] = None,
        background_inference: bool = False,
    ) -> dict:
        metadata = image.video_metadata
        requested_window_seconds = float(video_sampling.window_seconds)
        requested_stride_seconds = (
            requested_window_seconds
            - getattr(video_sampling, "overlap_frames", 0) / video_sampling.sample_fps
            if stride_seconds is None
            else float(stride_seconds)
        )
        if requested_stride_seconds <= 0 or not math.isfinite(requested_stride_seconds):
            raise ValueError("Stride must be positive and finite.")

        signature = (
            tuple(block_filter or ()),
            requested_window_seconds,
            requested_stride_seconds,
            confidence,
        )
        video_id = metadata.video_identifier
        frame_number = metadata.frame_number
        bookkeeping = self._video_bookkeeping.get(video_id)
        if bookkeeping is not None:
            self._video_bookkeeping.move_to_end(video_id)
        if (
            bookkeeping is None
            or bookkeeping.signature != signature
            or (
                bookkeeping.last_frame_number >= 0
                and frame_number < bookkeeping.last_frame_number
            )
        ):
            if bookkeeping is not None:
                self._discard_pending_call(bookkeeping)
            # The mode holds for the stream's life, so a call left running by
            # one mode never meets the other. A file is not paced by the clock,
            # so a worker would fall behind it and leave most windows unclassified.
            bookkeeping = _ActionRecognitionBookkeeping(
                signature=signature,
                background=background_inference and not metadata.comes_from_video_file,
            )
            self._video_bookkeeping[video_id] = bookkeeping
            while len(self._video_bookkeeping) > MAX_TRACKED_VIDEOS:
                evicted_video_id, evicted = self._video_bookkeeping.popitem(last=False)
                self._discard_pending_call(evicted)
                self._warned_fps_video_ids.discard(evicted_video_id)
                logger.warning(
                    "Action Recognition Model tracks at most %s videos and "
                    "dropped the state of %s. A step that gives every frame a "
                    "new video identifier, such as a crop, causes this.",
                    MAX_TRACKED_VIDEOS,
                    evicted_video_id,
                )

        effective_sample_fps, sampling_stride, stride_frames = self._sample_frame(
            image=image,
            model=model,
            video_sampling=video_sampling,
            bookkeeping=bookkeeping,
            requested_window_seconds=requested_window_seconds,
            requested_stride_seconds=requested_stride_seconds,
        )

        error_status = self._collect_finished_call(bookkeeping=bookkeeping)
        if bookkeeping.last_fire_frame_number is None:
            # A single frame has no temporal content to classify; anchor the
            # fire cadence at stream start and wait for a full stride.
            bookkeeping.last_fire_frame_number = frame_number
        # The model declares the fewest frames worth classifying; a fire
        # waits until the buffer reaches it.
        next_fire_frame_number = bookkeeping.last_fire_frame_number + stride_frames
        is_last_file_frame = bool(
            metadata.comes_from_video_file
            and metadata.total_frames is not None
            and frame_number >= metadata.total_frames
        )
        # A file ends between two scheduled calls, so its tail gets one call
        # of its own. The flag holds it to one when the count is too low.
        flushes_tail = (
            is_last_file_frame
            and not bookkeeping.tail_flushed
            and frame_number > bookkeeping.last_fire_frame_number
        )
        has_enough_samples = len(bookkeeping.sampled) >= max(
            1, int(video_sampling.min_frames)
        )
        should_classify = has_enough_samples and (
            frame_number >= next_fire_frame_number or flushes_tail
        )
        if should_classify and bookkeeping.pending is not None:
            # One call per stream at a time. The fire waits for the worker
            # and then takes the newest window, not the one it was due on.
            bookkeeping.deferred_window_end = bookkeeping.pending.last_covered_frame()
        elif should_classify:
            gap_notice = self._describe_deferral_gap(
                bookkeeping,
                supports_observed_duration=model.supports_observed_duration,
            )
            bookkeeping.deferred_window_end = None
            bookkeeping.tail_flushed = is_last_file_frame
            bookkeeping.last_fire_frame_number = frame_number
            call_error = self._classify_buffer(
                model=model,
                bookkeeping=bookkeeping,
                block_filter=block_filter,
                id_vocabulary=id_vocabulary,
                effective_sample_fps=effective_sample_fps,
                sampling_stride=sampling_stride,
                confidence=confidence,
                frame_limit=frame_number + 1,
                background=bookkeeping.background,
            )
            error_status = " ".join(
                part for part in (error_status, gap_notice, call_error) if part
            )

        bookkeeping.last_frame_number = frame_number
        return self._build_output(bookkeeping=bookkeeping, error_status=error_status)

    def _classify_buffer(
        self,
        model,
        bookkeeping: _ActionRecognitionBookkeeping,
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        effective_sample_fps: float,
        sampling_stride: float,
        confidence: Optional[Confidence] = None,
        frame_limit: Optional[int] = None,
        background: bool = False,
    ) -> str:
        if not bookkeeping.sampled:
            return ""
        missing = [
            math.ceil(sample.intended_frame_number - 1e-9)
            for sample in bookkeeping.sampled
            if sample.frame is None
        ]
        if missing:
            first_frame = math.floor(bookkeeping.sampled[0].intended_frame_number)
            last_frame = (
                frame_limit - 1
                if frame_limit is not None
                else math.ceil(bookkeeping.sampled[-1].intended_frame_number)
            )
            indices = ", ".join(str(number) for number in missing[:16])
            remainder = f" and {len(missing) - 16} more" if len(missing) > 16 else ""
            return (
                f"Missing required samples at source-frame indices {indices}{remainder} "
                f"for window {first_frame}-{last_frame}. Inference skipped this window."
            )

        # The buffer keeps sliding while a worker runs, so the call owns a
        # copy of the list. The block never writes the frames inside again.
        call = _WindowCall(
            sampled=list(bookkeeping.sampled),
            frame_limit=frame_limit,
            block_filter=block_filter,
            id_vocabulary=id_vocabulary,
            sampling_stride=sampling_stride,
            supports_observed_duration=model.supports_observed_duration,
        )
        classify = partial(
            self._infer_window,
            model,
            call=call,
            model_lock=self._model_lock,
            effective_sample_fps=effective_sample_fps,
            confidence=confidence,
        )
        if background:
            call.future = self._get_worker().submit(classify)
            bookkeeping.pending = call
            return ""

        try:
            segments = classify()
        except Exception as error:
            logger.warning(
                "Action Recognition Model call failed: %s",
                error,
                exc_info=True,
            )
            return str(error)

        self._apply_call_result(bookkeeping=bookkeeping, call=call, segments=segments)
        return ""

    @classmethod
    def _infer_window(
        cls,
        model,
        *,
        call: _WindowCall,
        model_lock,
        effective_sample_fps: float,
        confidence: Optional[Confidence],
    ) -> _WindowResult:
        """Run one window without keeping the owning block alive in the worker."""
        frames = cls._prepare_frames_for_model(
            [sample.frame for sample in call.sampled]
        )
        infer_kwargs = (
            {"confidence": confidence}
            if model.supports_confidence and confidence not in (None, "default")
            else {}
        )
        if model.supports_observed_duration and call.frame_limit is not None:
            duration_seconds = (
                call.frame_limit - call.sampled[0].intended_frame_number
            ) / (call.sampling_stride * effective_sample_fps)
            infer_kwargs["duration_seconds"] = min(
                duration_seconds, model.video_sampling.window_seconds
            )

        with model_lock:
            segments = model.infer(
                frames=frames,
                class_names=call.block_filter,
                fps=effective_sample_fps,
                **infer_kwargs,
            )
        # Separates "the model output one range" from "the block merged
        # several".
        logger.debug(
            "Action Recognition model call over sampled frames "
            "[%s, %s] returned %d pre-merge segment(s): %s",
            call.sampled[0].source_frame_number,
            call.sampled[-1].source_frame_number,
            len(segments),
            [
                (segment.start_frame_idx, segment.end_frame_idx, segment.class_name)
                for segment in segments
            ],
        )
        return segments

    def _apply_call_result(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        *,
        call: _WindowCall,
        segments: List[ModelActionRecognitionPrediction],
    ) -> None:
        # Same-class ranges merge only when no sampled frame lies in the
        # gap. ceil(stride) is that rule exactly: consecutive samples sit
        # floor/ceil(stride) apart; the next sampled gap is ~2x the stride.
        self._merge_segments(
            sampled=call.sampled,
            bookkeeping=bookkeeping,
            segments=segments,
            id_vocabulary=call.id_vocabulary,
            stride=max(1, math.ceil(call.sampling_stride)),
            block_filter=call.block_filter,
            sample_stride=call.sampling_stride,
            frame_limit=call.frame_limit,
        )
        bookkeeping.record_successful_window(
            start=call.first_covered_frame(), end=call.last_covered_frame()
        )

    def _collect_finished_call(self, bookkeeping: _ActionRecognitionBookkeeping) -> str:
        """Merge a finished background call on the step thread.

        The worker only runs the model. Every write to per-video state stays
        on the thread that reads it, so the state needs no lock.
        """
        call = bookkeeping.pending
        if call is None or not call.future.done():
            return ""

        bookkeeping.pending = None
        if call.future.cancelled():
            return "Classification was cancelled before it ran."

        error = call.future.exception()
        if error is not None:
            logger.warning(
                "Action Recognition Model call failed: %s",
                error,
                exc_info=error,
            )
            return str(error)

        segments = call.future.result()
        self._apply_call_result(bookkeeping=bookkeeping, call=call, segments=segments)
        return ""

    @staticmethod
    def _describe_deferral_gap(
        bookkeeping: _ActionRecognitionBookkeeping,
        *,
        supports_observed_duration: bool,
    ) -> str:
        """Name the frames between a late call and the one before it, if any."""
        if bookkeeping.deferred_window_end is None:
            return ""

        first_uncovered = bookkeeping.deferred_window_end + 1
        # Match the origin rule in merge_window_segments: continuous spans
        # anchor at floor(intended_frame_number), while sample-index spans
        # start at the retained source frame.
        window_start = _first_covered_frame(
            bookkeeping.sampled[0],
            supports_observed_duration=supports_observed_duration,
        )

        last_uncovered = window_start - 1
        if last_uncovered < first_uncovered:
            return ""

        # The whole gap, not only the part lost to the wait: with a stride
        # longer than the window, some of these frames have no window anyway.
        notice = (
            f"The previous classification had not finished, so this one started "
            f"late. No window covers source frames "
            f"{first_uncovered}-{last_uncovered}."
        )

        return notice

    def _get_worker(self) -> _WindowWorker:
        if self._worker is None:
            self._worker = _WindowWorker()
            # A block dropped without close() must not leave its thread behind.
            self._worker_finalizer = finalize(self, self._worker.close)
        return self._worker

    def close(self) -> None:
        """Stop the background worker, drop queued calls and clear stream state.

        A call that already runs finishes on its own; nothing reads its result.
        """
        if self._worker is not None:
            # Calling the finalizer runs the shutdown once and disarms it.
            self._worker_finalizer()
            self._worker = None
            self._worker_finalizer = None

        super().close()

    def _build_output(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        error_status: str,
    ) -> dict:
        predictions = ActionRecognitionResult.model_construct(
            timeline=_SharedTimeline(bookkeeping.timeline_snapshot),
            covered_frame_ranges=bookkeeping.covered_frame_ranges,
            stream_generation=bookkeeping.stream_generation,
            source_fps=bookkeeping.source_fps,
            analyzed_from_frame=bookkeeping.analyzed_from_frame,
            analyzed_until_frame=bookkeeping.analyzed_until_frame,
            error_status=error_status,
        )
        return {"predictions": predictions}
