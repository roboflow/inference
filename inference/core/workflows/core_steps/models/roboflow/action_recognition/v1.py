"""Stateful action recognition workflow block."""

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from inference.core import logger
from inference.core.managers.base import ModelManager
from inference.core.models.action_recognition import merge_window_segments
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    normalise_class_names,
)
from inference.core.workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    Batch,
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_PREDICTION_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models.models.base.action_recognition import (
    WHOLE_VIDEO_MODE,
    VideoSampling,
)

DEFAULT_SOURCE_FPS = 30.0
# A crop step mints a video identifier per detection per frame, so the
# per-video state needs a ceiling of its own.
MAX_TRACKED_VIDEOS = 256
SHORT_DESCRIPTION = "Classify actions and events over ranges of video frames."
LONG_DESCRIPTION = """
Classify actions and events in a video stream. The block continuously samples
each stream into a sliding window and adds each model result to a cumulative
timeline. The first classification runs one stride after the stream starts.
Later classifications run at each stride. By default, the stride equals the
window, so consecutive windows tile the stream without overlap. Set a smaller
stride to slide overlapping windows for finer range boundaries. Ranges can
overlap. Timeline ranges only grow when the block merges model evidence. The
block does not extend ranges provisionally. When a stream provides no source
FPS, the block assumes 30 FPS and logs a warning.

The window length and the sample rate are part of the model, not block
settings: a fine-tuned model declares the values its training used, and other
models declare their defaults (a 16 second window at 4 frames per second).

The block does not run an extra classification when a stream ends, so frames
after the final scheduled call do not receive a result. On a finite clip, keep
the stride shorter than the clip: the first classification waits one full
stride, so a longer stride never runs. Tail classification requires an
end-of-stream signal and is planned separately.

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
"""


def _extract_rgb_frame(image: WorkflowImageData) -> np.ndarray:
    return np.ascontiguousarray(image.numpy_image[:, :, ::-1])


@dataclass
class _ActionRecognitionBookkeeping:
    sampled: List[Tuple[int, Any]] = field(default_factory=list)
    timeline: List[ActionRecognitionPrediction] = field(default_factory=list)
    last_frame_number: int = -1
    last_fire_frame_number: Optional[int] = None
    next_sample_frame_number: Optional[float] = None
    source_fps: Optional[float] = None
    signature: Tuple[Tuple[str, ...], float, float] = field(
        default_factory=lambda: ((), 0.0, 0.0)
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
    stride_seconds: Union[Optional[float], Selector(kind=[FLOAT_KIND])] = Field(
        default=None,
        description=(
            "Time between classification calls. Leave empty to classify "
            "consecutive windows without overlap. A smaller stride slides "
            "overlapping windows for finer range boundaries at the cost of "
            "more model calls."
        ),
        examples=[None, 2.0],
    )

    @model_validator(mode="after")
    def validate_window_inputs(self) -> "BlockManifest":
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
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; action recognition needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        # Fine-tuned packages carry their own base weights, so the block
        # depends on no separately cached foundation model.
        return None


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
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._model = None
        self._current_model_id: Optional[str] = None
        self._video_bookkeeping: "OrderedDict[str, _ActionRecognitionBookkeeping]" = (
            OrderedDict()
        )
        self._warned_fps_video_ids: Set[str] = set()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_model(self, model_id: str):
        if self._model is None or self._current_model_id != model_id:
            # Imported here so loading the block does not pull the adapters
            # module, and so both surfaces load a model id identically.
            from inference.core.models.inference_models_adapters import (
                load_action_recognition_model,
            )

            self._model = load_action_recognition_model(
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
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        model = self._get_model(model_id=model_id)
        block_filter = normalise_class_names(class_filter) or None
        id_vocabulary = getattr(model, "class_names", None) or block_filter or None
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
    ) -> dict:
        metadata = image.video_metadata
        requested_window_seconds = float(video_sampling.window_seconds)
        requested_stride_seconds = (
            requested_window_seconds
            if stride_seconds is None
            else float(stride_seconds)
        )
        if requested_stride_seconds <= 0 or not math.isfinite(requested_stride_seconds):
            raise ValueError("Stride must be positive and finite.")

        signature = (
            tuple(block_filter or ()),
            requested_window_seconds,
            requested_stride_seconds,
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
        window_frames = max(1, round(requested_window_seconds * source_fps))
        stride_frames = max(1, round(requested_stride_seconds * source_fps))
        if bookkeeping.next_sample_frame_number is None:
            bookkeeping.next_sample_frame_number = float(frame_number)
        if frame_number >= bookkeeping.next_sample_frame_number:
            frame = self._extract_frame(image=image)
            # Frames can arrive with gaps. A timestamp stranded in a gap has
            # no frame of its own, and copying this one under each would
            # flood the buffer, so the cursor snaps past them first.
            if bookkeeping.next_sample_frame_number < frame_number - 1:
                bookkeeping.next_sample_frame_number = float(frame_number)
            # Advance on the float grid; integer anchoring rounds every step
            # up and drags the real sample rate below sample_fps. Several
            # timestamps landing on one frame each take it, which is the
            # repeat training fed a source slower than the recorded rate.
            while bookkeeping.next_sample_frame_number <= frame_number:
                bookkeeping.sampled.append((frame_number, frame))
                bookkeeping.next_sample_frame_number += sampling_stride

        cutoff_frame_number = frame_number - window_frames
        while bookkeeping.sampled and bookkeeping.sampled[0][0] <= cutoff_frame_number:
            bookkeeping.sampled.pop(0)

        error_status = ""
        if bookkeeping.last_fire_frame_number is None:
            # A single frame has no temporal content to classify; anchor the
            # fire cadence at stream start and wait for a full stride.
            bookkeeping.last_fire_frame_number = frame_number
        # The model declares the fewest frames worth classifying; a fire
        # waits until the buffer reaches it.
        next_fire_frame_number = bookkeeping.last_fire_frame_number + stride_frames
        should_classify = (
            len(bookkeeping.sampled) >= max(1, int(video_sampling.min_frames))
            and frame_number >= next_fire_frame_number
        )
        if should_classify:
            bookkeeping.last_fire_frame_number = frame_number
            error_status = self._classify_buffer(
                model=model,
                bookkeeping=bookkeeping,
                block_filter=block_filter,
                id_vocabulary=id_vocabulary,
                effective_sample_fps=effective_sample_fps,
                sampling_stride=sampling_stride,
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
    ) -> str:
        if not bookkeeping.sampled:
            return ""
        frames = self._prepare_frames_for_model(
            [frame for _, frame in bookkeeping.sampled]
        )
        try:
            segments = model.infer(
                frames=frames,
                class_names=block_filter,
                fps=effective_sample_fps,
            )
        except Exception as error:
            logger.warning(
                "Action Recognition Model call failed: %s",
                error,
                exc_info=True,
            )
            return str(error)
        # Separates "the model output one range" from "the block merged
        # several".
        logger.debug(
            "Action Recognition model call over sampled frames "
            "[%s, %s] returned %d pre-merge segment(s): %s",
            bookkeeping.sampled[0][0],
            bookkeeping.sampled[-1][0],
            len(segments),
            [
                (segment.start_frame_idx, segment.end_frame_idx, segment.class_name)
                for segment in segments
            ],
        )
        # Same-class ranges merge only when no sampled frame lies in the
        # gap. ceil(stride) is that rule exactly: consecutive samples sit
        # floor/ceil(stride) apart; the next sampled gap is ~2x the stride.
        self._merge_segments(
            bookkeeping=bookkeeping,
            segments=segments,
            block_filter=block_filter,
            id_vocabulary=id_vocabulary,
            stride=max(1, math.ceil(sampling_stride)),
        )
        return ""

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

    def _merge_segments(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        segments: List[ModelActionRecognitionPrediction],
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        stride: float,
    ) -> None:
        merge_window_segments(
            timeline=bookkeeping.timeline,
            frame_numbers=[frame_number for frame_number, _ in bookkeeping.sampled],
            segments=segments,
            id_vocabulary=id_vocabulary,
            stride=stride,
            class_filter=block_filter,
        )
        bookkeeping.timeline.sort(
            key=lambda entry: (
                entry.start_frame_idx,
                entry.class_id,
                entry.end_frame_idx,
            )
        )

    def _build_output(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        error_status: str,
    ) -> dict:
        return {
            "timeline": [entry.model_copy() for entry in bookkeeping.timeline],
            "error_status": error_status,
        }
