"""Stateful video segment classification workflow block."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from inference_models.models.base.video_segment_classification import (
    VideoSegmentClassificationModel,
)
from inference_models.models.base.video_segment_classification import (
    VideoSegmentClassificationPrediction as ModelVideoSegmentClassificationPrediction,
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
    VideoSegmentClassificationPrediction,
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
    VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND,
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
window length, so each classification sees one new window. Set a smaller
stride for overlapping windows. Ranges can overlap. An active range
advances with the stream until a later classification closes it. When a stream
provides no source FPS, the block assumes 30 FPS and logs a warning.

Frames per call equal window_seconds x sample_fps. The model spreads a fixed
pixel budget across those frames. Use fewer frames to keep each frame sharper
when small objects matter. Use more frames for denser temporal coverage.
Very short windows can fall below a model's temporal-localization floor:
cosmos-3-edge stops emitting events for windows under roughly 5 seconds, so
keep window_seconds at 6 or more for that model.

The block does not run an extra classification when a stream ends, so frames
after the final scheduled call do not receive a new result. On a finite clip,
keep the window shorter than the clip: the first classification waits one
full stride, so a longer window never runs. Tail classification requires an
end-of-stream signal and is planned separately.

Use this block with InferencePipeline for full temporal behavior. Still-image
and HTTP execution do not provide a continuous stream. A single frame has no
temporal content, so these paths return an empty timeline.

The class vocabulary is optional. Provide classes for zero-shot models, leave
them empty for fine-tuned models that carry their own class list, or leave them
empty on open-vocabulary models to let the model label events. The
window_classes output lists every class the most recent window classification
detected. It updates on each classification and holds between classifications.
It works with Classification Label Visualization. When a model call fails,
error_status carries the error text for that frame and the stream continues.
"""


def _extract_rgb_frame(image: WorkflowImageData) -> np.ndarray:
    return np.ascontiguousarray(image.numpy_image[:, :, ::-1])


@dataclass
class _VideoSegmentClassificationBookkeeping:
    sampled: List[Tuple[int, Any]] = field(default_factory=list)
    timeline: List[VideoSegmentClassificationPrediction] = field(default_factory=list)
    open_classes: Set[str] = field(default_factory=set)
    window_class_names: List[str] = field(default_factory=list)
    last_frame_number: int = -1
    last_fire_frame_number: Optional[int] = None
    next_sample_frame_number: Optional[float] = None
    source_fps: Optional[float] = None
    signature: Tuple[Tuple[str, ...], float, Optional[float], float] = field(
        default_factory=lambda: ((), 0.0, None, 0.0)
    )


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Video Segment Classification Model",
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

    type: Literal["roboflow_core/video_segment_classification_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_filter: Union[
        Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description=(
            "List of accepted classes. For fine-tuned models, classes must exist "
            "in the model's training set and the output is restricted to this "
            "subset. For zero-shot models such as cosmos-3-edge, detected "
            "events are classified into this list. Leave empty to accept all "
            "classes (open vocabulary on zero-shot models)."
        ),
        examples=[["a", "b", "c"], "$inputs.class_filter"],
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    window_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=2.0,
        description=(
            "Duration of the sliding classification window in seconds. Frames "
            "per call equal window_seconds x sample_fps. The model spreads a "
            "fixed pixel budget across those frames. A shorter window uses fewer, "
            "sharper frames. A longer window increases temporal coverage. "
            "cosmos-3-edge needs 6 seconds or more."
        ),
        examples=[2.0],
    )
    stride_seconds: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description=(
            "Time between classification calls. When unset, it defaults to "
            "window_seconds, so each call sees one new window. Set it below "
            "window_seconds for overlapping windows, at the cost of more "
            "model calls."
        ),
        examples=[None, 1.0, 2.0],
    )
    sample_fps: float = Field(
        default=4.0,
        description=(
            "Frames sampled per second for model input. Frames per call equal "
            "window_seconds x sample_fps. The model spreads a fixed pixel budget "
            "across those frames. A lower value keeps frames sharper for small "
            "objects. A higher value gives denser temporal coverage."
        ),
        examples=[4.0],
    )

    @model_validator(mode="after")
    def validate_window_inputs(self) -> "BlockManifest":
        numeric_values = [self.sample_fps]
        if isinstance(self.window_seconds, (int, float)):
            numeric_values.append(self.window_seconds)
        if isinstance(self.stride_seconds, (int, float)):
            numeric_values.append(self.stride_seconds)
        if any(value <= 0 or not math.isfinite(value) for value in numeric_values):
            raise ValueError(
                "Window, stride, and sample FPS must be positive and finite."
            )
        return self

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="timeline",
                kind=[VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND],
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
                note="Requires a GPU; video segment classification needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [DEFAULT_MODEL_ID]


class VideoSegmentClassificationModelBlockV1(WorkflowBlock):
    """Classify temporal segments in independent video streams."""

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "Video Segment Classification Model only supports LOCAL workflow step "
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
        self._video_bookkeeping: Dict[str, _VideoSegmentClassificationBookkeeping] = {}
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
            if not isinstance(loaded_model, VideoSegmentClassificationModel):
                from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
                    Cosmos3EdgeReasoner,
                )
                from inference_models.models.cosmos3.cosmos3_video_segment_classification import (
                    Cosmos3EdgeVideoSegmentClassification,
                )

                if isinstance(loaded_model, Cosmos3EdgeReasoner):
                    loaded_model = Cosmos3EdgeVideoSegmentClassification(
                        reasoner=loaded_model
                    )
                else:
                    raise ValueError(
                        f"Model {model_id} does not support video segment classification."
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
        window_seconds: float = 2.0,
        stride_seconds: Optional[float] = None,
        sample_fps: float = 4.0,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        model = self._get_model(model_id=model_id)
        block_filter = normalise_class_names(class_filter) or None
        id_vocabulary = getattr(model, "class_names", None) or block_filter or None
        results = []
        for image in images:
            results.append(
                self._process_frame(
                    model=model,
                    image=image,
                    block_filter=block_filter,
                    id_vocabulary=id_vocabulary,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    sample_fps=sample_fps,
                )
            )
        return results

    def _process_frame(
        self,
        model,
        image: WorkflowImageData,
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        window_seconds: float,
        stride_seconds: Optional[float],
        sample_fps: float,
    ) -> dict:
        metadata = image.video_metadata
        requested_sample_fps = float(sample_fps)
        requested_window_seconds = float(window_seconds)
        requested_stride_seconds = (
            None if stride_seconds is None else float(stride_seconds)
        )
        if (
            requested_sample_fps <= 0
            or not math.isfinite(requested_sample_fps)
            or requested_window_seconds <= 0
            or not math.isfinite(requested_window_seconds)
            or (
                requested_stride_seconds is not None
                and (
                    requested_stride_seconds <= 0
                    or not math.isfinite(requested_stride_seconds)
                )
            )
        ):
            raise ValueError(
                "Window, stride, and sample FPS must be positive and finite."
            )

        signature = (
            tuple(block_filter or ()),
            requested_window_seconds,
            requested_stride_seconds,
            requested_sample_fps,
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
            bookkeeping = _VideoSegmentClassificationBookkeeping(signature=signature)
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

        effective_sample_fps = min(requested_sample_fps, source_fps)
        sampling_stride = max(1.0, source_fps / effective_sample_fps)
        window_frames = max(1, round(requested_window_seconds * source_fps))
        effective_stride_seconds = (
            requested_window_seconds
            if requested_stride_seconds is None
            else requested_stride_seconds
        )
        stride_frames = max(1, round(effective_stride_seconds * source_fps))

        if bookkeeping.next_sample_frame_number is None:
            bookkeeping.next_sample_frame_number = float(frame_number)
        if frame_number >= bookkeeping.next_sample_frame_number:
            bookkeeping.sampled.append((frame_number, self._extract_frame(image=image)))
            # Advance on the float grid; integer anchoring rounds every
            # step up and drags the real sample rate below sample_fps.
            while bookkeeping.next_sample_frame_number <= frame_number:
                bookkeeping.next_sample_frame_number += sampling_stride

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
        should_classify = bookkeeping.sampled and (
            frame_number >= bookkeeping.last_fire_frame_number + stride_frames
        )
        if should_classify:
            bookkeeping.last_fire_frame_number = frame_number
            # The model rounds boundaries by ~10-20% of its window;
            # exact-touch closes ongoing events too early.
            open_end_slack_frames = max(
                0.15 * requested_window_seconds * source_fps,
                sampling_stride,
            )
            error_status = self._classify_buffer(
                model=model,
                bookkeeping=bookkeeping,
                block_filter=block_filter,
                id_vocabulary=id_vocabulary,
                effective_sample_fps=effective_sample_fps,
                sampling_stride=sampling_stride,
                open_end_slack_frames=open_end_slack_frames,
            )

        bookkeeping.last_frame_number = frame_number
        return self._build_output(
            bookkeeping=bookkeeping,
            image=image,
            id_vocabulary=id_vocabulary,
            frame_number=frame_number,
            error_status=error_status,
        )

    def _resolve_source_fps(
        self,
        metadata: VideoMetadata,
        bookkeeping: _VideoSegmentClassificationBookkeeping,
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
                "Video Segment Classification Model did not receive a valid source FPS. "
                "It uses 30 FPS for windowing and sampling."
            )
            self._warned_fps_video_ids.add(metadata.video_identifier)
        bookkeeping.source_fps = DEFAULT_SOURCE_FPS
        return DEFAULT_SOURCE_FPS

    def _classify_buffer(
        self,
        model,
        bookkeeping: _VideoSegmentClassificationBookkeeping,
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        effective_sample_fps: float,
        sampling_stride: float,
        open_end_slack_frames: float,
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
                "Video Segment Classification Model call failed: %s",
                error,
                exc_info=True,
            )
            return str(error)
        bookkeeping.window_class_names = list(
            dict.fromkeys(segment.class_name for segment in segments)
        )
        # Separates "the model output one range" from "the block merged
        # several".
        logger.debug(
            "Video Segment Classification model call over sampled frames "
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
            open_end_slack_frames=open_end_slack_frames,
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
        bookkeeping: _VideoSegmentClassificationBookkeeping,
        segments: List[ModelVideoSegmentClassificationPrediction],
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        stride: float,
        open_end_slack_frames: float,
    ) -> None:
        sampled_count = len(bookkeeping.sampled)
        new_open_classes = set()
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
            segment = VideoSegmentClassificationPrediction(
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
            if (
                bookkeeping.sampled[-1][0] - segment.end_frame_idx
                <= open_end_slack_frames
            ):
                new_open_classes.add(class_name)
        bookkeeping.timeline.sort(
            key=lambda entry: (
                entry.start_frame_idx,
                entry.class_id,
                entry.end_frame_idx,
            )
        )
        bookkeeping.open_classes = new_open_classes

    @staticmethod
    def _merge_segment(
        timeline: List[VideoSegmentClassificationPrediction],
        segment: VideoSegmentClassificationPrediction,
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

    @staticmethod
    def _build_output(
        bookkeeping: _VideoSegmentClassificationBookkeeping,
        image: WorkflowImageData,
        id_vocabulary: Optional[List[str]],
        frame_number: int,
        error_status: str,
    ) -> dict:
        timeline = [entry.model_copy(deep=True) for entry in bookkeeping.timeline]
        for class_name in bookkeeping.open_classes:
            for entry in reversed(timeline):
                if entry.class_name == class_name:
                    entry.end_frame_idx = frame_number
                    break
        window_class_names = list(bookkeeping.window_class_names)
        height, width = image._read_shape_without_materialization()
        parent_id = image.parent_metadata.parent_id
        window_classes = {
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
        return {
            "timeline": timeline,
            "window_classes": window_classes,
            "error_status": error_status,
        }
