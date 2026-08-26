"""Stateful video segment classification workflow block."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union

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
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND,
    ImageInputField,
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
timeline. Classification starts on the first frame with a growing buffer and
repeats at a configurable stride. By default, the stride is half the window
length for 50 percent overlap. Ranges can overlap. An active range advances
with the stream until a later classification closes it.

The block does not run an extra classification when a stream ends, so frames
after the final scheduled call do not receive a new result. Tail
classification requires an end-of-stream signal and is planned separately.

Use this block with InferencePipeline for full temporal behavior. Still-image
and HTTP execution do not provide a continuous stream, so they classify one
frame without temporal context.
"""


def _extract_rgb_frame(image: WorkflowImageData) -> np.ndarray:
    return np.ascontiguousarray(image.numpy_image[:, :, ::-1])


@dataclass
class _VideoClassificationBookkeeping:
    sampled: List[Tuple[int, Any]] = field(default_factory=list)
    timeline: List[VideoSegmentClassificationPrediction] = field(default_factory=list)
    open_classes: Set[str] = field(default_factory=set)
    last_frame_number: int = -1
    last_fire_frame_number: Optional[int] = None
    signature: Tuple[Tuple[str, ...], float, Optional[float], float, float] = field(
        default_factory=lambda: ((), 0.0, None, 0.0, 0.0)
    )


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Video Classification Model",
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

    type: Literal["roboflow_core/video_classification_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_names: Union[
        List[str],
        str,
        Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND]),
    ] = Field(
        description=(
            "Classes to locate, as a list or a comma-separated string. "
            "The class position becomes the output class ID."
        ),
        examples=[["walking", "running"]],
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default=DEFAULT_MODEL_ID,
        title="Model Id",
        description="Model used for temporal localization.",
        examples=[DEFAULT_MODEL_ID],
    )
    window_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=2.0,
        description="Duration of the sliding classification window in seconds.",
        examples=[2.0],
    )
    stride_seconds: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description=(
            "Time between classification calls. When unset, it defaults to "
            "window_seconds / 2 for 50 percent overlap. Set it equal to "
            "window_seconds for non-overlapping windows."
        ),
        examples=[None, 1.0, 2.0],
    )
    sample_fps: float = Field(
        default=4.0,
        description="Frames sampled per second for model input.",
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
            OutputDefinition(name="active_classes", kind=[LIST_OF_VALUES_KIND]),
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
                note="Requires a GPU; video classification needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [DEFAULT_MODEL_ID]


class VideoClassificationModelBlockV1(WorkflowBlock):
    """Classify temporal segments in independent video streams."""

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "Video Classification Model only supports LOCAL workflow step "
        "execution. Remote execution sends frames to separate processes and "
        "breaks the per-video classification state. Set "
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
        self._video_bookkeeping: Dict[str, _VideoClassificationBookkeeping] = {}
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
                from inference_models.models.cosmos3.cosmos3_video_classification import (
                    Cosmos3VideoSegmentClassification,
                )

                if isinstance(loaded_model, Cosmos3EdgeReasoner):
                    loaded_model = Cosmos3VideoSegmentClassification(
                        reasoner=loaded_model
                    )
                else:
                    raise ValueError(
                        f"Model {model_id} does not support video classification."
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
        class_names: Union[List[str], str],
        model_id: str = DEFAULT_MODEL_ID,
        window_seconds: float = 2.0,
        stride_seconds: Optional[float] = None,
        sample_fps: float = 4.0,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        model = self._get_model(model_id=model_id)
        classes = normalise_class_names(class_names)
        results = []
        for image in images:
            results.append(
                self._process_frame(
                    model=model,
                    image=image,
                    class_names=classes,
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
        class_names: List[str],
        window_seconds: float,
        stride_seconds: Optional[float],
        sample_fps: float,
    ) -> dict:
        metadata = image.video_metadata
        source_fps = self._resolve_source_fps(metadata=metadata)
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

        effective_sample_fps = min(requested_sample_fps, source_fps)
        sampling_stride = max(1.0, source_fps / effective_sample_fps)
        window_frames = max(1, round(requested_window_seconds * source_fps))
        effective_stride_seconds = (
            requested_window_seconds / 2
            if requested_stride_seconds is None
            else requested_stride_seconds
        )
        stride_frames = max(1, round(effective_stride_seconds * source_fps))
        signature = (
            tuple(class_names),
            requested_window_seconds,
            requested_stride_seconds,
            requested_sample_fps,
            source_fps,
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
            bookkeeping = _VideoClassificationBookkeeping(signature=signature)
            self._video_bookkeeping[video_id] = bookkeeping

        if (
            not bookkeeping.sampled
            or frame_number >= bookkeeping.sampled[-1][0] + sampling_stride
        ):
            bookkeeping.sampled.append((frame_number, self._extract_frame(image=image)))

        cutoff_frame_number = frame_number - window_frames
        while (
            bookkeeping.sampled
            and bookkeeping.sampled[0][0] <= cutoff_frame_number
        ):
            bookkeeping.sampled.pop(0)

        error_status = ""
        should_classify = bookkeeping.sampled and (
            bookkeeping.last_fire_frame_number is None
            or frame_number
            >= bookkeeping.last_fire_frame_number + stride_frames
        )
        if should_classify:
            bookkeeping.last_fire_frame_number = frame_number
            error_status = self._classify_buffer(
                model=model,
                bookkeeping=bookkeeping,
                class_names=class_names,
                effective_sample_fps=effective_sample_fps,
                sampling_stride=sampling_stride,
            )

        bookkeeping.last_frame_number = frame_number
        return self._build_output(
            bookkeeping=bookkeeping,
            class_names=class_names,
            frame_number=frame_number,
            error_status=error_status,
        )

    def _resolve_source_fps(self, metadata: VideoMetadata) -> float:
        for candidate in (metadata.fps, metadata.measured_fps):
            if candidate is None:
                continue
            candidate = float(candidate)
            if candidate > 0 and math.isfinite(candidate):
                return candidate
        if metadata.video_identifier not in self._warned_fps_video_ids:
            logger.warning(
                "Video Classification Model did not receive a valid source FPS. "
                "It uses 30 FPS for windowing and sampling."
            )
            self._warned_fps_video_ids.add(metadata.video_identifier)
        return DEFAULT_SOURCE_FPS

    def _classify_buffer(
        self,
        model,
        bookkeeping: _VideoClassificationBookkeeping,
        class_names: List[str],
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
                class_names=class_names,
                fps=effective_sample_fps,
            )
        except Exception as error:
            logger.warning(
                "Video Classification Model call failed: %s",
                error,
                exc_info=True,
            )
            return str(error)
        self._merge_segments(
            bookkeeping=bookkeeping,
            segments=segments,
            class_names=class_names,
            stride=sampling_stride,
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
        bookkeeping: _VideoClassificationBookkeeping,
        segments: List[ModelVideoSegmentClassificationPrediction],
        class_names: List[str],
        stride: float,
    ) -> None:
        sampled_count = len(bookkeeping.sampled)
        new_open_classes = set()
        for segment in segments:
            class_name = segment.class_name
            if class_name not in class_names:
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
                class_id=class_names.index(class_name),
            )
            self._merge_segment(
                timeline=bookkeeping.timeline,
                segment=segment,
                stride=stride,
            )
            if end_idx == sampled_count - 1:
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
        bookkeeping: _VideoClassificationBookkeeping,
        class_names: List[str],
        frame_number: int,
        error_status: str,
    ) -> dict:
        timeline = [entry.model_copy(deep=True) for entry in bookkeeping.timeline]
        for class_name in bookkeeping.open_classes:
            for entry in reversed(timeline):
                if entry.class_name == class_name:
                    entry.end_frame_idx = frame_number
                    break
        active_classes = [
            class_name
            for class_name in class_names
            if class_name in bookkeeping.open_classes
        ]
        return {
            "timeline": timeline,
            "active_classes": active_classes,
            "error_status": error_status,
        }
