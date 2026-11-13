"""Stateful action recognition workflow block."""

import logging
import math
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Set, Tuple, Type, Union
from weakref import finalize

import cv2
import numpy as np
from pydantic import ConfigDict, Field, model_validator
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_ACTUAL_RESTRICTION,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    normalise_class_names,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
    REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
)
from roboflow_workflows.errors import WorkflowEnvironmentConfigurationError
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    Batch,
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_PREDICTION_KIND,
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    ModelExecutionLocation,
    ModelRequiredAction,
    Runtime,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
    roboflow_platform_model,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider
from roboflow_workflows.utils.action_recognition import merge_window_segments

from inference_models.entities import Confidence
from inference_models.models.base.action_recognition import WHOLE_VIDEO_MODE
from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models.models.base.action_recognition import (
    VideoSampling,
    effective_max_frame_side,
)

logger = get_logger(__name__)

DEFAULT_SOURCE_FPS = 30.0
# A crop step mints a video identifier per detection per frame, so the
# per-video state needs a ceiling of its own.
MAX_TRACKED_VIDEOS = 256
# A stream that runs for days keeps adding ranges whenever its class changes.
# The timeline is the block's output, so it is copied out per frame, and both
# the copy and the memory grow with it. This holds each stream to a ceiling,
# and 256 streams at the ceiling stay under a gigabyte.
MAX_TIMELINE_ACTIONS = 5000
SHORT_DESCRIPTION = "Classify actions and events over ranges of video frames."
LONG_DESCRIPTION = """
Classify actions and events in a video stream. The block continuously samples
each stream into a sliding window and adds each model result to a cumulative
timeline. The first classification runs one stride after the stream starts.
Later classifications run at each stride. By default, the stride equals the
window, so consecutive windows tile the stream without overlap. Set a smaller
stride to slide overlapping windows for finer range boundaries. Ranges can
overlap. Timeline ranges only grow when the block merges model evidence. The
block does not extend ranges provisionally. The timeline holds at most 5000
ranges per stream. A stream that passes that ceiling loses the ranges that
ended earliest, so its timeline covers recent history rather than all of it. When a stream provides no source
FPS, the block assumes 30 FPS and logs a warning.

The window length and the sample rate are part of the model, not block
settings: a fine-tuned model declares the values its training used, and other
models declare their defaults (a 16 second window at 4 frames per second).

analyzed_until_frame is the source index of the last frame that a successful
classification covered, or null before the first one. An empty timeline after
it means "not analyzed", not "no action". Failed and skipped windows do not
advance it. Overlapping windows can still revise ranges before it.

On a video file that reports its frame count, the block runs one extra
classification on the last frame, so the timeline covers the whole clip, even
one shorter than one stride. That call also needs the model's minimum number of
sampled frames. Live streams and files without a usable frame count get no
extra call, so frames after the final scheduled call receive no result.

By default a classification runs inside the step, so the video waits for
the model. With background_inference, a live stream hands each window to a
worker thread and the step returns at once. The result reaches the timeline
on a later frame, and analyzed_until_frame shows how far results go. One
call runs per stream at a time: a call that comes due while the previous one
runs waits, then takes the newest window, and error_status names any frames
no window covered. Video files always classify inside the step, so every
window is analyzed and results are repeatable.

Use this block with InferencePipeline for full temporal behavior. Still-image
and HTTP execution do not provide a continuous stream. A single frame has no
temporal content, so these paths return an empty timeline.

This block serves fine-tuned models only. A model trained on whole videos
spans each clip in one call, and a stream has no end to span, so the block
refuses it and names the action recognition endpoint instead.

The class vocabulary is optional. Leave it empty to report every class the
model carries, or list classes to report a subset of them. When a model call
fails, error_status carries the error text for that frame and the stream
continues.

If required samples are missing, the block skips that window and reports
their source-frame indices in error_status. Later complete windows resume
inference. Source frames intentionally skipped by sampling are not missing.
"""


def _extract_rgb_frame(image: WorkflowImageData) -> np.ndarray:
    return np.ascontiguousarray(image.numpy_image[:, :, ::-1])


@dataclass
class _SampledFrame:
    intended_frame_number: float
    source_frame_number: Optional[int]
    frame: Any


@dataclass
class _WindowCall:
    """One window handed to the model, with what its merge needs later."""

    sampled: List[_SampledFrame]
    frame_limit: Optional[int]
    block_filter: Optional[List[str]]
    id_vocabulary: Optional[List[str]]
    sampling_stride: float
    future: Optional[Future] = None


@dataclass
class _ActionRecognitionBookkeeping:
    sampled: List[_SampledFrame] = field(default_factory=list)
    timeline: List[ActionRecognitionPrediction] = field(default_factory=list)
    timeline_snapshot: List[ActionRecognitionPrediction] = field(default_factory=list)
    dropped_history: bool = False
    last_frame_number: int = -1
    last_fire_frame_number: Optional[int] = None
    analyzed_until_frame: Optional[int] = None
    tail_flushed: bool = False
    pending: Optional[_WindowCall] = None
    fire_deferred: bool = False
    next_sample_frame_number: Optional[float] = None
    source_fps: Optional[float] = None
    signature: Tuple[Tuple[str, ...], float, float, Optional[Confidence], bool] = field(
        default_factory=lambda: ((), 0.0, 0.0, None, False)
    )


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition Model",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "action recognition",
                "video classification",
                "temporal localization",
                "cosmos",
            ],
            "ui_manifest": {
                "section": "video",
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/roboflow_action_recognition_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_filter: Union[Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])] = (
        Field(
            default=None,
            description=(
                "List of accepted classes. Classes must exist in the model's "
                "training set, and the output is restricted to this subset. "
                "Leave empty to report every class the model carries."
            ),
            examples=[["a", "b", "c"], "$inputs.class_filter"],
        )
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence: Union[
        Optional[Confidence], Selector(kind=[FLOAT_KIND, STRING_KIND])
    ] = Field(
        default=None,
        description=(
            'Candidate threshold before merging. "best" uses model-eval thresholds, '
            '"default" or empty uses the model built-in, or pass a float.'
        ),
    )
    stride_seconds: Union[Optional[float], Selector(kind=[FLOAT_KIND])] = Field(
        default=None,
        description=(
            "Time between classification calls. Leave empty to classify "
            "windows with the model's recorded overlap. A smaller stride slides "
            "overlapping windows for finer range boundaries at the cost of "
            "more model calls."
        ),
        examples=[None, 2.0],
    )
    background_inference: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description=(
            "Classify live-stream windows on a worker thread, so the step "
            "returns at once and the video does not wait for the model. "
            "Results reach the timeline on a later frame. Video files always "
            "classify inside the step."
        ),
        examples=[False, "$inputs.background_inference"],
    )

    @model_validator(mode="after")
    def validate_window_inputs(self) -> "BlockManifest":
        if isinstance(self.confidence, (int, float)) and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between zero and one")
        if isinstance(self.stride_seconds, (int, float)) and (
            self.stride_seconds <= 0 or not math.isfinite(self.stride_seconds)
        ):
            raise ValueError("Stride must be positive and finite.")
        return self

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="timeline",
                kind=[ACTION_RECOGNITION_PREDICTION_KIND],
            ),
            OutputDefinition(name="error_status", kind=[STRING_KIND]),
            OutputDefinition(name="analyzed_until_frame", kind=[INTEGER_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Return the block's coarse execution restrictions.

        Returns:
            Restrictions that apply on this host, each with a stable ``code``.
        """
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            RuntimeRestriction(
                code="requires_gpu_for_local_execution",
                severity=Severity.HARD,
                note="Requires a GPU; action recognition needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
            REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        # Fine-tuned packages carry their own base weights, so the block
        # depends on no separately cached foundation model.
        return None

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        """Declare the action recognition model of the block's local run path.

        The block supports LOCAL step execution only (its run path rejects any
        other mode) and owns its model loading: it asks the model provider for
        the model through `load_action_recognition_model()`, not through the
        generic `add_model()` registration. The declared dependency describes
        that supported execution path, so it is LOCAL and kept away from the
        generic preloader. The configured id is returned verbatim, selector
        included. The tensor sibling re-exports this manifest.

        Returns:
            The configured action recognition model.
        """
        return [
            roboflow_platform_model(
                self.model_id,
                required_action=ModelRequiredAction.EXECUTION,
                execution_location=ModelExecutionLocation.LOCAL,
                preloadable=False,
            )
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.TEMPORAL_BUFFERING,
        ]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare the REMOTE, GPU, cross-frame state-loss and still-image caveats.

        Args:
            ignore_environment_restrictions: If True, return every declaration
                with its condition intact (the portable view). If False,
                evaluate configuration predicates against this host and drop
                entries that definitively do not apply here.

        Returns:
            The step's restrictions. In the host view the discovery is
            incomplete when a configuration predicate cannot be evaluated.
        """
        return actual_restrictions_of(
            declared=[
                STATEFUL_VIDEO_ACTUAL_RESTRICTION,
                REQUIRES_GPU_FOR_LOCAL_EXECUTION,
                STILL_IMAGE_INPUT_SOFT_RESTRICTION,
                REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
            ],
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class ActionRecognitionModelBlockV1(WorkflowBlock):
    """Classify temporal segments in independent video streams."""

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "Action Recognition Model only supports LOCAL workflow step "
        "execution. Remote execution sends frames to separate processes and "
        "breaks the per-video state. Set "
        "WORKFLOWS_STEP_EXECUTION_MODE=local to use this block."
    )

    def __init__(
        self,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
        model_manager: Optional[ModelsProvider] = None,
    ):
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._model_manager = model_manager
        self._model = None
        self._current_model_id: Optional[str] = None
        self._video_bookkeeping: "OrderedDict[str, _ActionRecognitionBookkeeping]" = (
            OrderedDict()
        )
        self._warned_fps_video_ids: Set[str] = set()
        # A file stream classifies on the step thread while a live stream's
        # worker may be inside the same model.
        self._model_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_finalizer = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "step_execution_mode", "model_manager"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_model(self, model_id: str):
        if self._model is None or self._current_model_id != model_id:
            if self._model_manager is None:
                raise WorkflowEnvironmentConfigurationError(
                    public_message=(
                        "This block loads its model through the Workflows models "
                        "provider. Supply `workflows_core.model_manager` in the "
                        "workflow init parameters."
                    ),
                    context="workflow_execution | step_execution | model_loading",
                )
            self._model = self._model_manager.load_action_recognition_model(
                model_id=model_id, api_key=self._api_key
            )
            self._current_model_id = model_id
            self._video_bookkeeping.clear()
        return self._model

    def _extract_frame(self, image: WorkflowImageData):
        return _extract_rgb_frame(image=image)

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
            A timeline, error status and analyzed-frame mark for each
            input frame.
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
            background_inference,
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
            bookkeeping = _ActionRecognitionBookkeeping(signature=signature)
            self._video_bookkeeping[video_id] = bookkeeping
            while len(self._video_bookkeeping) > MAX_TRACKED_VIDEOS:
                evicted_video_id, _ = self._video_bookkeeping.popitem(last=False)
                self._warned_fps_video_ids.discard(evicted_video_id)
                logger.warning(
                    "Action Recognition Model tracks at most %s videos and "
                    "dropped the state of %s. A step that gives every frame a "
                    "new video identifier, such as a crop, causes this.",
                    MAX_TRACKED_VIDEOS,
                    evicted_video_id,
                )

        # The fps pin holds for the video's life; per-frame re-resolution
        # would let estimator jitter reshuffle the frame math mid-stream.
        if bookkeeping.source_fps is None:
            source_fps = self._resolve_source_fps(
                metadata=metadata,
                bookkeeping=bookkeeping,
            )
        else:
            source_fps = bookkeeping.source_fps

        if video_sampling.max_frames is not None:
            # A trained model reads its recorded rate whatever the source
            # supplies. Training drew that many timestamps and repeated the
            # frame under each one, so capping here would hand the model a
            # shorter input stamped at a rate it never saw.
            effective_sample_fps = float(video_sampling.sample_fps)
        else:
            effective_sample_fps = min(float(video_sampling.sample_fps), source_fps)
        sampling_stride = source_fps / effective_sample_fps
        allowed_lateness_seconds = video_sampling.max_sample_lateness_seconds
        if not math.isfinite(allowed_lateness_seconds) or allowed_lateness_seconds < 0:
            raise ValueError("Sample lateness must be nonnegative and finite.")
        allowed_lateness_frames = source_fps * min(
            allowed_lateness_seconds, 0.5 / effective_sample_fps
        )
        window_frames = max(1, round(requested_window_seconds * source_fps))
        stride_frames = max(1, round(requested_stride_seconds * source_fps))
        cutoff_frame_number = frame_number - window_frames
        if bookkeeping.next_sample_frame_number is None:
            bookkeeping.next_sample_frame_number = float(frame_number)
        if (
            video_sampling.requires_regular_sampling
            and bookkeeping.next_sample_frame_number <= cutoff_frame_number
        ):
            expired_positions = (
                math.floor(
                    (cutoff_frame_number - bookkeeping.next_sample_frame_number)
                    / sampling_stride
                )
                + 1
            )
            bookkeeping.next_sample_frame_number += expired_positions * sampling_stride
        if frame_number >= math.ceil(bookkeeping.next_sample_frame_number - 1e-9):
            frame = self._extract_frame(image=image)
            frame_transform = getattr(model, "frame_storage_transform", None)
            if frame_transform is None:
                frame = self._cap_frame_side(
                    frame=frame,
                    max_side=effective_max_frame_side(video_sampling),
                )
            else:
                frame = frame_transform(frame)
            if video_sampling.requires_regular_sampling:
                # Empty slots preserve the model clock without inventing missing pixels.
                frame_used = False
                while (
                    math.ceil(bookkeeping.next_sample_frame_number - 1e-9)
                    <= frame_number
                ):
                    intended_frame = bookkeeping.next_sample_frame_number
                    expected_frame = math.ceil(intended_frame - 1e-9)
                    within_tolerance = (
                        frame_number - expected_frame <= allowed_lateness_frames + 1e-9
                    )
                    # A slow source can map several intended timestamps to this
                    # exact frame. Repeat it for those slots, but do not reuse a
                    # late frame to fill slots that requested a different frame.
                    use_frame = within_tolerance and (
                        not frame_used or expected_frame == frame_number
                    )
                    bookkeeping.sampled.append(
                        _SampledFrame(
                            intended_frame_number=intended_frame,
                            source_frame_number=frame_number if use_frame else None,
                            frame=frame if use_frame else None,
                        )
                    )
                    frame_used = frame_used or use_frame
                    bookkeeping.next_sample_frame_number += sampling_stride
            else:
                if bookkeeping.next_sample_frame_number < frame_number - 1:
                    bookkeeping.next_sample_frame_number = float(frame_number)
                while bookkeeping.next_sample_frame_number <= frame_number:
                    bookkeeping.sampled.append(
                        _SampledFrame(
                            intended_frame_number=float(frame_number),
                            source_frame_number=frame_number,
                            frame=frame,
                        )
                    )
                    bookkeeping.next_sample_frame_number += sampling_stride

        while (
            bookkeeping.sampled
            and bookkeeping.sampled[0].intended_frame_number <= cutoff_frame_number
        ):
            bookkeeping.sampled.pop(0)
        if video_sampling.max_frames is not None:
            # Rounded source-frame windows can contain one extra sampling timestamp.
            while len(bookkeeping.sampled) > video_sampling.max_frames:
                bookkeeping.sampled.pop(0)

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
            bookkeeping.fire_deferred = True
        elif should_classify:
            gap_notice = self._describe_deferral_gap(
                bookkeeping=bookkeeping,
                cutoff_frame_number=cutoff_frame_number,
            )
            bookkeeping.fire_deferred = False
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
                # A file is not paced by the clock, so a worker would fall
                # behind it and leave most windows unclassified.
                background=background_inference and not metadata.comes_from_video_file,
            )
            error_status = " ".join(
                part for part in (error_status, gap_notice, call_error) if part
            )

        bookkeeping.last_frame_number = frame_number
        return self._build_output(bookkeeping=bookkeeping, error_status=error_status)

    def _resolve_source_fps(
        self,
        metadata: VideoMetadata,
        bookkeeping: _ActionRecognitionBookkeeping,
    ) -> float:
        # measured_fps is never consumed: under processing-paced delivery
        # (WebRTC ACK windows) it tracks model latency, not the source
        # clock, and pinning it once fired the model on every frame.
        declared_fps = metadata.fps
        if declared_fps is not None:
            declared_fps = float(declared_fps)
            if declared_fps > 0 and math.isfinite(declared_fps):
                bookkeeping.source_fps = declared_fps
                return declared_fps

        if metadata.video_identifier not in self._warned_fps_video_ids:
            logger.warning(
                "Action Recognition Model did not receive a valid source FPS. "
                "It uses 30 FPS for windowing and sampling."
            )
            self._warned_fps_video_ids.add(metadata.video_identifier)
        bookkeeping.source_fps = DEFAULT_SOURCE_FPS
        return DEFAULT_SOURCE_FPS

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
        # copy of the list. The frames inside are never written again.
        call = _WindowCall(
            sampled=list(bookkeeping.sampled),
            frame_limit=frame_limit,
            block_filter=block_filter,
            id_vocabulary=id_vocabulary,
            sampling_stride=sampling_stride,
        )
        if background:
            call.future = self._get_executor().submit(
                self._infer_window,
                model,
                call=call,
                effective_sample_fps=effective_sample_fps,
                confidence=confidence,
            )
            bookkeeping.pending = call
            return ""

        try:
            segments = self._infer_window(
                model,
                call=call,
                effective_sample_fps=effective_sample_fps,
                confidence=confidence,
            )
        except Exception as error:
            logger.warning(
                "Action Recognition Model call failed: %s",
                error,
                exc_info=True,
            )
            return str(error)

        self._apply_call_result(bookkeeping=bookkeeping, call=call, segments=segments)
        return ""

    def _infer_window(
        self,
        model,
        *,
        call: _WindowCall,
        effective_sample_fps: float,
        confidence: Optional[Confidence],
    ) -> List[ModelActionRecognitionPrediction]:
        """Run the model on one window. Touches no per-video state."""
        frames = self._prepare_frames_for_model(
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
        with self._model_lock:
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
        merge_window_segments(
            timeline=bookkeeping.timeline,
            frame_numbers=[sample.source_frame_number for sample in call.sampled],
            segments=segments,
            id_vocabulary=call.id_vocabulary,
            stride=max(1, math.ceil(call.sampling_stride)),
            class_filter=call.block_filter,
            sample_stride=call.sampling_stride,
            frame_limit=call.frame_limit,
            sample_start_frame=call.sampled[0].intended_frame_number,
        )
        self._evict_oldest_actions(bookkeeping=bookkeeping)
        bookkeeping.timeline.sort(
            key=lambda entry: (
                entry.start_frame_idx,
                entry.class_id,
                entry.end_frame_idx,
            )
        )
        # A fire is the only thing that changes the timeline, so the copy the
        # output needs belongs here rather than under every frame.
        bookkeeping.timeline_snapshot = [
            entry.model_copy() for entry in bookkeeping.timeline
        ]
        if call.frame_limit is not None:
            # Only a call that returned moves the mark, so a consumer can tell
            # "no action" from "not analyzed".
            bookkeeping.analyzed_until_frame = call.frame_limit - 1

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

        self._apply_call_result(
            bookkeeping=bookkeeping,
            call=call,
            segments=call.future.result(),
        )
        return ""

    @staticmethod
    def _describe_deferral_gap(
        bookkeeping: _ActionRecognitionBookkeeping,
        *,
        cutoff_frame_number: int,
    ) -> str:
        """Name the frames a late call left without a result, if any."""
        if not bookkeeping.fire_deferred:
            return ""

        first_uncovered = bookkeeping.last_fire_frame_number + 1
        if cutoff_frame_number < first_uncovered:
            return ""

        return (
            f"The previous classification was still running, so source frames "
            f"{first_uncovered}-{cutoff_frame_number} have no result."
        )

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            # One worker: calls on one model must not overlap on the device.
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="action-recognition",
            )
            # A block dropped without close() must not leave its thread behind.
            self._executor_finalizer = finalize(
                self,
                self._executor.shutdown,
                wait=False,
                cancel_futures=True,
            )
        return self._executor

    def close(self) -> None:
        """Stop the background worker and drop calls that have not started."""
        if self._executor is None:
            return

        # Calling the finalizer runs the shutdown once and disarms it.
        self._executor_finalizer()
        self._executor = None
        self._executor_finalizer = None

    @staticmethod
    def _prepare_frames_for_model(frames: List[Any]) -> List[Any]:
        numpy_frames = [isinstance(frame, np.ndarray) for frame in frames]
        if all(numpy_frames) or not any(numpy_frames):
            return frames
        normalized = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                normalized.append(frame)
                continue
            normalized.append(
                np.ascontiguousarray(frame.detach().permute(1, 2, 0).to("cpu").numpy())
            )
        return normalized

    @staticmethod
    def _cap_frame_side(frame: Any, max_side: Optional[int]) -> Any:
        """Shrink a frame to the side the model trained on, before buffering.

        The buffer holds one window of frames, so its size is set by the frame
        the stream supplies rather than by anything the model asked for. At
        1080p that is 398 MB per stream against 25 MB at a declared 360.

        The model caps the side again before it infers, with this same
        operation, so shrinking here leaves the pixels it reads unchanged and
        only moves the work earlier. A model that declares no side keeps its
        frames whole, because there is no size it is known to accept.
        """
        if not max_side or max_side <= 0:
            return frame
        height, width = frame.shape[:2]
        scale = max_side / max(height, width)
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _evict_oldest_actions(bookkeeping: _ActionRecognitionBookkeeping) -> None:
        """Hold the timeline at ``MAX_TIMELINE_ACTIONS``, dropping stale ranges.

        Eviction goes by the frame an action last covered, not by where the
        entry sorts. The timeline sorts by start, so an action that has run
        since the stream opened sorts first while it is still the current one.
        """
        excess = len(bookkeeping.timeline) - MAX_TIMELINE_ACTIONS
        if excess <= 0:
            return
        bookkeeping.timeline.sort(key=lambda entry: entry.end_frame_idx)
        del bookkeeping.timeline[:excess]
        if not bookkeeping.dropped_history:
            bookkeeping.dropped_history = True
            logger.warning(
                "Action Recognition Model holds at most %s timeline ranges per "
                "video and dropped the ranges that ended earliest. The timeline "
                "covers recent history rather than the whole stream.",
                MAX_TIMELINE_ACTIONS,
            )

    def _build_output(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        error_status: str,
    ) -> dict:
        # Copying every entry here costs one full timeline copy per frame,
        # which grows without bound and caps the frame rate the block keeps
        # up with. The entries come from the snapshot the last fire built.
        # The list is still fresh per frame, so a consumer that appends to
        # one frame's output leaves the next frame alone.
        return {
            "timeline": list(bookkeeping.timeline_snapshot),
            "error_status": error_status,
            "analyzed_until_frame": bookkeeping.analyzed_until_frame,
        }
