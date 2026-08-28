"""Stateful action recognition workflow block."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from inference_models.models.base.action_recognition import (
    VideoSampling,
    ActionRecognitionModel,
)
from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)

from inference.core import logger
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    normalise_class_names,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    ActionRecognitionPrediction,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ACTION_RECOGNITION_PREDICTION_KIND,
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

DEFAULT_MODEL_ID = "cosmos-3-edge"
DEFAULT_SOURCE_FPS = 30.0
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

When a stream ends, the block classifies sampled frames that arrived after the
final scheduled call, provided the buffer meets the model's minimum frame
count. This gives shorter-than-window clips and longer clip tails a final
result.

A model trained to read a clip as one unit reports whole video sampling. For
these models the block holds a thinned set of frames that spans the stream and
classifies once, when the stream ends.

Use this block with InferencePipeline for full temporal behavior. Still-image
and HTTP execution do not provide a continuous stream. A single frame has no
temporal content, so these paths return an empty timeline.

The class vocabulary is optional. Provide classes for zero-shot models, leave
them empty for fine-tuned models that carry their own class list, or leave them
empty on open-vocabulary models to let the model label events. The
window_classes output lists classes whose range can still merge with a future
classification. A class stays listed for one window plus the sample tolerance
after its range ends. It clears when the range closes. This output works with
Classification Label Visualization. When a model call fails, error_status
carries the error text for that frame and the stream continues.
"""


def _extract_rgb_frame(image: WorkflowImageData) -> np.ndarray:
    return np.ascontiguousarray(image.numpy_image[:, :, ::-1])


@dataclass(frozen=True)
class _ResolvedVideoSampling:
    video_sampling: VideoSampling
    stride_frames: int
    effective_sample_fps: float
    sampling_stride: float
    keep_alive_frames: int


@dataclass
class _ActionRecognitionBookkeeping:
    sampled: List[Tuple[int, Any]] = field(default_factory=list)
    timeline: List[ActionRecognitionPrediction] = field(default_factory=list)
    window_class_names: List[str] = field(default_factory=list)
    last_frame_number: int = -1
    last_fire_frame_number: Optional[int] = None
    next_sample_frame_number: Optional[float] = None
    source_fps: Optional[float] = None
    signature: Tuple[Tuple[str, ...], float, float] = field(
        default_factory=lambda: ((), 0.0, 0.0)
    )
    resolved_sampling: Optional[_ResolvedVideoSampling] = None
    block_filter: Optional[List[str]] = None
    id_vocabulary: Optional[List[str]] = None
    last_image: Optional[WorkflowImageData] = None
    last_batch_index: Optional[Tuple[int, ...]] = None


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
    class_filter: Union[
        Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description=(
            "List of accepted classes. For fine-tuned models, classes must exist "
            "in the model's training set and the output is restricted to this "
            "subset. For zero-shot models, detected events are classified "
            "into this list. Leave empty to accept all classes (open "
            "vocabulary on zero-shot models)."
        ),
        examples=[["a", "b", "c"], "$inputs.class_filter"],
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
            OutputDefinition(
                name="window_classes", kind=[CLASSIFICATION_PREDICTION_KIND]
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
        return [DEFAULT_MODEL_ID]


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
        self._video_bookkeeping: Dict[str, _ActionRecognitionBookkeeping] = {}
        self._warned_fps_video_ids: Set[str] = set()

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_model(self, model_id: str):
        if self._model is not None and self._current_model_id is None:
            self._current_model_id = model_id
            return self._model
        if self._model is None or self._current_model_id != model_id:
            from inference_models import AutoModel

            loaded_model = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self._api_key,
                weights_provider_extra_headers=get_extra_weights_provider_headers(),
            )
            if not isinstance(loaded_model, ActionRecognitionModel):
                from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
                    Cosmos3EdgeReasoner,
                )
                from inference_models.models.cosmos3.cosmos3_action_recognition import (
                    Cosmos3EdgeActionRecognition,
                )

                if isinstance(loaded_model, Cosmos3EdgeReasoner):
                    loaded_model = Cosmos3EdgeActionRecognition(
                        reasoner=loaded_model
                    )
                else:
                    raise ValueError(
                        f"Model {model_id} does not support action recognition."
                    )
            self._model = loaded_model
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
        results = []
        for batch_index, image in zip(_batch_indices(images=images), images):
            results.append(
                self._process_frame(
                    model=model,
                    image=image,
                    batch_index=batch_index,
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
        batch_index: Tuple[int, ...],
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
        if requested_stride_seconds <= 0 or not math.isfinite(
            requested_stride_seconds
        ):
            raise ValueError("Stride must be positive and finite.")

        signature = (
            tuple(block_filter or ()),
            requested_window_seconds,
            requested_stride_seconds,
        )
        video_id = metadata.video_identifier
        frame_number = metadata.frame_number
        bookkeeping = self._video_bookkeeping.get(video_id)
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

        # The fps pin holds for the video's life; per-frame re-resolution
        # would let estimator jitter reshuffle the frame math mid-stream.
        if bookkeeping.source_fps is None:
            source_fps = self._resolve_source_fps(
                metadata=metadata,
                bookkeeping=bookkeeping,
            )
        else:
            source_fps = bookkeeping.source_fps

        effective_sample_fps = min(float(video_sampling.sample_fps), source_fps)
        sampling_stride = max(1.0, source_fps / effective_sample_fps)
        window_frames = max(1, round(requested_window_seconds * source_fps))
        stride_frames = max(1, round(requested_stride_seconds * source_fps))
        keep_alive_frames = window_frames + math.ceil(sampling_stride)
        bookkeeping.resolved_sampling = _ResolvedVideoSampling(
            video_sampling=VideoSampling(
                window_seconds=requested_window_seconds,
                sample_fps=float(video_sampling.sample_fps),
                min_frames=max(1, int(video_sampling.min_frames)),
                mode=video_sampling.mode,
                frame_budget=video_sampling.frame_budget,
            ),
            stride_frames=stride_frames,
            effective_sample_fps=effective_sample_fps,
            sampling_stride=sampling_stride,
            keep_alive_frames=keep_alive_frames,
        )
        bookkeeping.block_filter = block_filter
        bookkeeping.id_vocabulary = id_vocabulary
        bookkeeping.last_image = image
        bookkeeping.last_batch_index = batch_index

        if bookkeeping.next_sample_frame_number is None:
            bookkeeping.next_sample_frame_number = float(frame_number)
        if frame_number >= bookkeeping.next_sample_frame_number:
            bookkeeping.sampled.append((frame_number, self._extract_frame(image=image)))
            # Advance on the float grid; integer anchoring rounds every
            # step up and drags the real sample rate below sample_fps.
            while bookkeeping.next_sample_frame_number <= frame_number:
                bookkeeping.next_sample_frame_number += sampling_stride

        if video_sampling.classifies_whole_video:
            # The model reads a clip as one unit, so the buffer spans
            # everything seen so far, thinned to the trained frame budget.
            _thin_to_frame_budget(
                bookkeeping=bookkeeping,
                frame_budget=video_sampling.window_frames,
            )
        else:
            cutoff_frame_number = frame_number - window_frames
            while (
                bookkeeping.sampled
                and bookkeeping.sampled[0][0] <= cutoff_frame_number
            ):
                bookkeeping.sampled.pop(0)

        error_status = ""
        if bookkeeping.last_fire_frame_number is None:
            # A single frame has no temporal content to classify; anchor the
            # fire cadence at stream start and wait for a full stride.
            bookkeeping.last_fire_frame_number = frame_number
        # The model declares the fewest frames worth classifying; a fire
        # waits until the buffer reaches it.
        next_fire_frame_number = (
            bookkeeping.last_fire_frame_number
            + bookkeeping.resolved_sampling.stride_frames
        )
        should_classify = (
            not video_sampling.classifies_whole_video
            and len(bookkeeping.sampled)
            >= bookkeeping.resolved_sampling.video_sampling.min_frames
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
        return self._build_output(
            bookkeeping=bookkeeping,
            image=image,
            id_vocabulary=id_vocabulary,
            frame_number=frame_number,
            keep_alive_frames=keep_alive_frames,
            error_status=error_status,
        )

    def is_stream_pipelined(self) -> bool:
        return True

    def stream_pipeline_depth(self) -> int:
        return 0

    def flush_stream_pipeline_outputs(
        self,
    ) -> List[Tuple[List[Tuple[int, ...]], BlockResult]]:
        if self._model is None:
            return []
        results = []
        for bookkeeping in self._video_bookkeeping.values():
            resolved_sampling = bookkeeping.resolved_sampling
            if (
                not bookkeeping.sampled
                or resolved_sampling is None
                or bookkeeping.last_fire_frame_number is None
                or bookkeeping.last_image is None
                or bookkeeping.last_batch_index is None
            ):
                continue
            newest_sample_frame_number = bookkeeping.sampled[-1][0]
            if newest_sample_frame_number <= bookkeeping.last_fire_frame_number:
                continue
            if (
                len(bookkeeping.sampled)
                < resolved_sampling.video_sampling.min_frames
            ):
                continue
            bookkeeping.last_fire_frame_number = newest_sample_frame_number
            error_status = self._classify_buffer(
                model=self._model,
                bookkeeping=bookkeeping,
                block_filter=bookkeeping.block_filter,
                id_vocabulary=bookkeeping.id_vocabulary,
                effective_sample_fps=_buffer_sample_fps(
                    bookkeeping=bookkeeping,
                    resolved_sampling=resolved_sampling,
                ),
                sampling_stride=resolved_sampling.sampling_stride,
            )
            output = self._build_output(
                bookkeeping=bookkeeping,
                image=bookkeeping.last_image,
                id_vocabulary=bookkeeping.id_vocabulary,
                frame_number=bookkeeping.last_frame_number,
                keep_alive_frames=resolved_sampling.keep_alive_frames,
                error_status=error_status,
            )
            results.append(([bookkeeping.last_batch_index], [output]))
        return results

    def close_stream_pipeline(self) -> None:
        self._video_bookkeeping.clear()

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
            stride=math.ceil(sampling_stride),
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
        sampled_count = len(bookkeeping.sampled)
        for segment in segments:
            class_name = segment.class_name
            if block_filter is not None and class_name not in block_filter:
                continue
            start_idx = min(
                sampled_count - 1,
                max(0, int(segment.start_frame_idx)),
            )
            end_idx = min(
                sampled_count - 1,
                max(0, int(segment.end_frame_idx)),
            )
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            segment = ActionRecognitionPrediction(
                start_frame_idx=bookkeeping.sampled[start_idx][0],
                end_frame_idx=bookkeeping.sampled[end_idx][0],
                class_name=class_name,
                class_id=(
                    id_vocabulary.index(class_name)
                    if id_vocabulary is not None and class_name in id_vocabulary
                    else -1
                ),
            )
            self._merge_segment(
                timeline=bookkeeping.timeline,
                segment=segment,
                stride=stride,
            )
        bookkeeping.timeline.sort(
            key=lambda entry: (
                entry.start_frame_idx,
                entry.class_id,
                entry.end_frame_idx,
            )
        )

    @staticmethod
    def _merge_segment(
        timeline: List[ActionRecognitionPrediction],
        segment: ActionRecognitionPrediction,
        stride: float,
    ) -> None:
        matching = [
            existing
            for existing in timeline
            if existing.class_name == segment.class_name
            and existing.start_frame_idx <= segment.end_frame_idx + stride
            and segment.start_frame_idx <= existing.end_frame_idx + stride
        ]
        if not matching:
            timeline.append(segment)
            return
        segment.start_frame_idx = min(
            segment.start_frame_idx,
            *(entry.start_frame_idx for entry in matching),
        )
        segment.end_frame_idx = max(
            segment.end_frame_idx,
            *(entry.end_frame_idx for entry in matching),
        )
        timeline[:] = [entry for entry in timeline if entry not in matching]
        timeline.append(segment)

    def _build_output(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        image: WorkflowImageData,
        id_vocabulary: Optional[List[str]],
        frame_number: int,
        keep_alive_frames: int,
        error_status: str,
    ) -> dict:
        timeline = [entry.model_copy(deep=True) for entry in bookkeeping.timeline]
        # A class stays visible while a future positive fire could still merge
        # with its range, then clears exactly when that range is closed.
        bookkeeping.window_class_names = list(
            dict.fromkeys(
                entry.class_name
                for entry in bookkeeping.timeline
                if frame_number - entry.end_frame_idx <= keep_alive_frames
            )
        )
        window_classes = self._build_window_classes(
            bookkeeping=bookkeeping,
            image=image,
            id_vocabulary=id_vocabulary,
        )
        return {
            "timeline": timeline,
            "window_classes": window_classes,
            "error_status": error_status,
        }

    def _build_window_classes(
        self,
        bookkeeping: _ActionRecognitionBookkeeping,
        image: WorkflowImageData,
        id_vocabulary: Optional[List[str]],
    ) -> Any:
        window_class_names = list(bookkeeping.window_class_names)
        height, width = image._read_shape_without_materialization()
        parent_id = image.parent_metadata.parent_id
        return {
            "image": {"width": width, "height": height},
            "predictions": {
                class_name: {
                    "confidence": 1.0,
                    "class_id": (
                        id_vocabulary.index(class_name)
                        if id_vocabulary is not None
                        and class_name in id_vocabulary
                        else -1
                    ),
                }
                for class_name in window_class_names
            },
            "predicted_classes": window_class_names,
            "prediction_type": "classification",
            "parent_id": parent_id,
            "root_parent_id": parent_id,
            "inference_id": str(uuid4()),
        }


def _buffer_sample_fps(
    bookkeeping: "_ActionRecognitionBookkeeping",
    resolved_sampling: "_ResolvedVideoSampling",
) -> float:
    """The rate the buffered frames actually represent.

    Thinning a whole-video buffer lowers its real rate below the sampled
    one, and the model reads its frame timestamps from this value.
    """
    if not resolved_sampling.video_sampling.classifies_whole_video:
        return resolved_sampling.effective_sample_fps
    source_fps = bookkeeping.source_fps
    if len(bookkeeping.sampled) < 2 or not source_fps:
        return resolved_sampling.effective_sample_fps
    span_frames = bookkeeping.sampled[-1][0] - bookkeeping.sampled[0][0]
    if span_frames <= 0:
        return resolved_sampling.effective_sample_fps
    return (len(bookkeeping.sampled) - 1) * source_fps / span_frames

def _thin_to_frame_budget(bookkeeping, frame_budget: int) -> None:
    """Keep a uniform spread of at most ``frame_budget`` samples.

    Whole-video models see the frame budget spread over the full clip, and
    a stream has no known end, so the buffer thins as it grows. Keeping the
    first and last samples holds the span the model reasons over.
    """
    sample_count = len(bookkeeping.sampled)
    if sample_count <= frame_budget:
        return
    step = (sample_count - 1) / (frame_budget - 1) if frame_budget > 1 else 0.0
    kept_positions = sorted(
        {round(index * step) for index in range(max(1, frame_budget))}
    )
    bookkeeping.sampled = [bookkeeping.sampled[index] for index in kept_positions]

def _batch_indices(images: Batch[WorkflowImageData]) -> List[Tuple[int, ...]]:
    indices = getattr(images, "indices", None)
    if indices is not None:
        return indices
    return [(i,) for i in range(len(images))]
