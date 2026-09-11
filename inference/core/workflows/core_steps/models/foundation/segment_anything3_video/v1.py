"""SAM3 Video Tracker workflow block.

Wraps ``inference_models``'s ``SAM3Video`` concept tracker and
``SAM3TrackerVideo`` visual tracker so they can run in a workflow powered by
``InferencePipeline``. The pipeline delivers one frame at a time with
``WorkflowImageData.video_metadata``; this block keeps one state dictionary
per ``video_identifier``.

In concept mode, text prompts are registered once and the model runs fused
detection and tracking on every frame. Objects that enter the scene and match
a concept receive new tracker IDs automatically. The block re-seeds the
session only when the stream restarts or ``class_names`` changes.

In visual mode, labeled points define one object and each box defines a
separate object. The ``prompt_mode`` policy controls when the block re-seeds
the visual tracker. Between re-prompts, the tracker keeps temporal memory.
Each re-prompt starts a new session and restarts tracker IDs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, model_validator

from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    SAM3_CONCEPT_VIDEO_MODEL_ID,
    SAM3_VISUAL_VIDEO_MODEL_ID,
    VideoSessionBookkeeping,
    build_obj_id_metadata_from_visual_prompts,
    concept_frame_to_sv_detections,
    decide_prompt_vs_track,
    extract_box_prompts,
    masks_to_sv_detections,
    normalise_class_names,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.visual_prompt import (
    SYNTHETIC_POINT_PROMPT_CLASS_ID,
    SYNTHETIC_POINT_PROMPT_CLASS_NAME,
    normalise_labeled_points,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LABELED_POINTS_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
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
from inference.usage_tracking.collector import usage_collector

PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]
TrackingMode = Literal["concept", "visual"]

SHORT_DESCRIPTION = "Track concepts or visually prompted objects across video frames."

_EMPTY_MASKS = np.zeros((0, 0, 0), dtype=bool)
_EMPTY_IDS = np.zeros((0,), dtype=np.int64)
_EMPTY_SCORES = np.zeros((0,), dtype=np.float32)
_EMPTY_BOXES = np.zeros((0, 4), dtype=np.float32)


def _frame_for_model(single_image: WorkflowImageData) -> np.ndarray:
    """Convert a workflow BGR image into an HWC RGB model frame."""
    return np.ascontiguousarray(single_image.numpy_image[:, :, ::-1])


LONG_DESCRIPTION = """
Run Segment Anything 3 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.

Use `concept` mode to track text concepts. SAM3 detects matching objects
on every frame and assigns tracker IDs to objects that enter the scene.

Use `visual` mode to track objects from points or boxes. The block reads
the visual prompts according to `prompt_mode`, then keeps temporal state
between prompts. Each re-prompt starts a new session and restarts tracker IDs.

The block multiplexes a single SAM3 streaming model across many video
streams by keying state on `video_metadata.video_identifier`. The block
uses `sam3video` for concept mode and `sam3trackervideo` for visual mode.

Intended for use with `InferencePipeline`, which delivers one frame at
a time and tags each frame with video metadata.
"""


@dataclass
class _ConceptSessionBookkeeping:
    """Per-video session state for the concept tracker.

    ``prompt_signature`` records the exact concept set the live session
    was seeded with, so a change in ``class_names`` (e.g. driven by a
    workflow parameter) re-seeds instead of silently tracking stale
    concepts.
    """

    state_dict: Optional[dict] = None
    last_frame_number: int = -1
    prompt_signature: Tuple[str, ...] = field(default_factory=tuple)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM3 Video Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "SAM3",
                "segment anything 3",
                "video",
                "tracking",
                "open vocabulary",
                "visual tracking",
                "point prompt",
                "box prompt",
                "PVS",
                "META",
            ],
            "ui_manifest": {
                "section": "video",
                "icon": "fa-brands fa-meta",
                "blockPriority": 9.39,
                "needsGPU": True,
                "inference": True,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/sam3_video@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    tracking_mode: TrackingMode = Field(
        default="concept",
        title="Tracking Mode",
        description="Use `concept` for text prompts. Use `visual` for point or box prompts.",
        json_schema_extra={"always_visible": True},
    )
    class_names: Optional[
        Union[List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])]
    ] = Field(
        default=None,
        description=(
            "Concepts to segment and track, as a list of phrases (or a "
            "single comma-separated string).  Each emitted mask carries "
            "the concept it matched as its class name."
        ),
        examples=[["person", "forklift"]],
        json_schema_extra={
            "always_visible": True,
            "relevant_for": {
                "tracking_mode": {"values": ["concept"], "required": True}
            },
        },
    )
    points: Optional[Union[List[Any], Selector(kind=[LABELED_POINTS_KIND])]] = Field(
        default=None,
        title="Point Prompts",
        description=(
            "Labeled points for one object. Positive points include regions. "
            "Negative points exclude regions."
        ),
        examples=[
            [{"x": 320, "y": 240, "positive": True}],
            "$inputs.points",
        ],
        json_schema_extra={
            "always_visible": True,
            "relevant_for": {"tracking_mode": {"values": ["visual"], "required": True}},
        },
    )
    boxes: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description=(
            "Bounding boxes for visual tracking. The block tracks one object "
            "for each box."
        ),
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={
            "always_visible": True,
            "relevant_for": {"tracking_mode": {"values": ["visual"], "required": True}},
        },
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default=SAM3_CONCEPT_VIDEO_MODEL_ID,
        title="Model Id",
        description="Streaming SAM3 model ID for concept tracking.",
        examples=[SAM3_CONCEPT_VIDEO_MODEL_ID],
        json_schema_extra={"relevant_for": {"tracking_mode": {"values": ["concept"]}}},
    )
    visual_model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default=SAM3_VISUAL_VIDEO_MODEL_ID,
        title="Model Id",
        description="Streaming SAM3 model ID for visual tracking.",
        examples=[SAM3_VISUAL_VIDEO_MODEL_ID],
        json_schema_extra={"relevant_for": {"tracking_mode": {"values": ["visual"]}}},
    )
    prompt_mode: PromptMode = Field(
        default="first_frame",
        description=(
            "When visual mode reads its prompts. `first_frame` reads prompts "
            "once. `every_n_frames` reads them at the selected interval. "
            "`every_frame` reads them on each frame. Each re-prompt starts a "
            "new session and restarts tracker IDs."
        ),
        json_schema_extra={"relevant_for": {"tracking_mode": {"values": ["visual"]}}},
    )
    prompt_interval: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description=(
            "The visual prompt interval for `every_n_frames` mode. A scheduled "
            "re-prompt starts a new session."
        ),
        examples=[30],
        json_schema_extra={"relevant_for": {"tracking_mode": {"values": ["visual"]}}},
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.5,
        description=(
            "Minimum confidence for emitted masks. Concept mode uses the "
            "SAM3 detection score. Visual mode uses prompt confidence."
        ),
        examples=[0.5],
    )

    @model_validator(mode="after")
    def validate_tracking_mode(self) -> "BlockManifest":
        if self.tracking_mode == "concept" and self.class_names is None:
            raise ValueError("Concept mode requires `class_names`.")
        if (
            self.tracking_mode == "visual"
            and self.points is None
            and self.boxes is None
        ):
            raise ValueError("Visual mode requires `points`, `boxes`, or both.")
        if isinstance(self.points, list):
            normalise_labeled_points(self.points)
        return self

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "boxes"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
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
                note="Requires a GPU; the streaming SAM3 video model needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [SAM3_CONCEPT_VIDEO_MODEL_ID, SAM3_VISUAL_VIDEO_MODEL_ID]

    # `discover_dependent_resources()` deliberately not implemented: this
    # block loads its weights via AutoModel.from_pretrained, not the model
    # manager — dependencies stay undeclared (None) for now.


class SegmentAnything3VideoBlockV1(WorkflowBlock):
    """Stateful SAM3 concept and visual tracking block."""

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "SAM3 Video Tracker only supports LOCAL workflow step "
        "execution.  Remote execution would ship each frame to a "
        "separate process and break the per-video SAM3 session "
        "that holds the temporal memory.  Set "
        "WORKFLOWS_STEP_EXECUTION_MODE=local (or run on a "
        "dedicated deployment) to use this block."
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
        self._model = None  # lazily loaded
        self._current_model_id: Optional[str] = None
        self._concept_sessions: Dict[str, _ConceptSessionBookkeeping] = {}
        self._visual_sessions: Dict[str, VideoSessionBookkeeping] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_model(self, model_id: str):
        if self._model is None or self._current_model_id != model_id:
            from inference_models import AutoModel

            extra_weights_provider_headers = get_extra_weights_provider_headers()
            self._model = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self._api_key,
                weights_provider_extra_headers=extra_weights_provider_headers,
                content_addressed_artifact_cache=(
                    self._model_manager.content_addressed_artifact_cache
                ),
            )
            self._current_model_id = model_id
            # Switching model invalidates every session we held.
            self._concept_sessions.clear()
            self._visual_sessions.clear()
        return self._model

    def run(
        self,
        images: Batch[WorkflowImageData],
        class_names: Optional[Union[List[str], str]],
        model_id: str,
        threshold: float,
        tracking_mode: TrackingMode = "concept",
        visual_model_id: str = SAM3_VISUAL_VIDEO_MODEL_ID,
        points: Optional[List[Any]] = None,
        boxes: Optional[Batch[sv.Detections]] = None,
        prompt_mode: PromptMode = "first_frame",
        prompt_interval: int = 30,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        # The usage decorator reads `model_id` off the tracked call, so the
        # dispatch happens out here where the mode-dependent model is picked.
        selected_model_id = model_id if tracking_mode == "concept" else visual_model_id
        return self._tracked_run(
            images=images,
            class_names=class_names,
            model_id=selected_model_id,
            threshold=threshold,
            tracking_mode=tracking_mode,
            points=points,
            boxes=boxes,
            prompt_mode=prompt_mode,
            prompt_interval=prompt_interval,
        )

    @usage_collector("model")
    def _tracked_run(
        self,
        images: Batch[WorkflowImageData],
        class_names: Optional[Union[List[str], str]],
        model_id: str,
        threshold: float,
        tracking_mode: TrackingMode,
        points: Optional[List[Any]],
        boxes: Optional[Batch[sv.Detections]],
        prompt_mode: PromptMode,
        prompt_interval: int,
    ) -> BlockResult:
        model = self._get_model(model_id=model_id)
        if tracking_mode == "visual":
            return self._run_visual(
                model=model,
                images=images,
                points=points,
                boxes=boxes,
                prompt_mode=prompt_mode,
                prompt_interval=prompt_interval,
                threshold=threshold,
            )
        return self._run_concept(
            model=model,
            images=images,
            class_names=class_names,
            threshold=threshold,
        )

    def _run_concept(
        self,
        model,
        images: Batch[WorkflowImageData],
        class_names: Optional[Union[List[str], str]],
        threshold: float,
    ) -> BlockResult:
        class_list = normalise_class_names(class_names)
        prompt_signature = tuple(class_list)

        batch_detections: List[sv.Detections] = []
        for single_image in images:
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0

            session = self._concept_sessions.setdefault(
                video_id, _ConceptSessionBookkeeping()
            )
            stream_restarted = (
                session.last_frame_number >= 0
                and frame_number < session.last_frame_number
            )
            if stream_restarted or session.prompt_signature != prompt_signature:
                session.state_dict = None

            if not class_list:
                detections = concept_frame_to_sv_detections(
                    masks=_EMPTY_MASKS,
                    object_ids=_EMPTY_IDS,
                    scores=_EMPTY_SCORES,
                    boxes=_EMPTY_BOXES,
                    prompt_to_object_ids={},
                    class_names=class_list,
                    image=single_image,
                    threshold=threshold,
                )
            else:
                frame_np = _frame_for_model(single_image)
                if session.state_dict is None:
                    result = model.prompt(
                        image=frame_np,
                        text=class_list,
                        clear_old_prompts=True,
                    )
                    session.prompt_signature = prompt_signature
                else:
                    result = model.track(image=frame_np, state_dict=session.state_dict)
                session.state_dict = result.state_dict
                detections = concept_frame_to_sv_detections(
                    masks=result.masks,
                    object_ids=result.object_ids,
                    scores=result.scores,
                    boxes=result.boxes,
                    prompt_to_object_ids=result.prompt_to_object_ids,
                    class_names=class_list,
                    image=single_image,
                    threshold=threshold,
                )

            session.last_frame_number = frame_number
            batch_detections.append(detections)

        batch_detections = attach_prediction_type_info_to_sv_detections_batch(
            predictions=batch_detections,
            prediction_type="instance-segmentation",
        )
        batch_detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=batch_detections,
        )
        return [{"predictions": pred} for pred in batch_detections]

    def _run_visual(
        self,
        model,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch[sv.Detections]],
        prompt_mode: PromptMode,
        prompt_interval: int,
        threshold: float,
    ) -> BlockResult:
        point_prompts = normalise_labeled_points(points)
        boxes_iter = boxes if boxes is not None else [None] * len(images)
        batch_detections: List[sv.Detections] = []

        for single_image, boxes_for_image in zip(images, boxes_iter):
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0
            session = self._visual_sessions.setdefault(
                video_id, VideoSessionBookkeeping()
            )
            has_box_prompts = boxes_for_image is not None and len(boxes_for_image) > 0
            has_prompts = has_box_prompts or bool(point_prompts)
            should_reset, should_prompt = decide_prompt_vs_track(
                session=session,
                frame_number=frame_number,
                prompt_mode=prompt_mode,
                prompt_interval=prompt_interval,
                has_prompts=has_prompts,
            )
            if should_reset:
                session.state_dict = None
                session.obj_id_metadata = {}
                session.frames_since_prompt = 0

            if should_prompt:
                frame_np = _frame_for_model(single_image)
                boxes_xyxy, per_box_meta = extract_box_prompts(boxes_for_image)
                masks, obj_ids, new_state = model.prompt(
                    image=frame_np,
                    bboxes=boxes_xyxy,
                    points=point_prompts,
                    clear_old_prompts=True,
                    frame_idx=frame_number,
                )
                session.obj_id_metadata = build_obj_id_metadata_from_visual_prompts(
                    obj_ids=obj_ids,
                    box_metas=per_box_meta,
                    has_point_prompt=bool(point_prompts),
                )
                session.state_dict = new_state
                session.frames_since_prompt = 0
            elif session.state_dict is not None:
                frame_np = _frame_for_model(single_image)
                masks, obj_ids, new_state = model.track(
                    image=frame_np, state_dict=session.state_dict
                )
                session.state_dict = new_state
                session.frames_since_prompt += 1
            else:
                height, width = single_image._read_shape_without_materialization()
                masks = np.zeros((0, height, width), dtype=bool)
                obj_ids = np.zeros((0,), dtype=np.int64)

            session.last_frame_number = frame_number
            batch_detections.append(
                masks_to_sv_detections(
                    masks=masks,
                    obj_ids=obj_ids,
                    image=single_image,
                    obj_id_metadata=session.obj_id_metadata,
                    threshold=threshold,
                    fallback_class_id=SYNTHETIC_POINT_PROMPT_CLASS_ID,
                    fallback_class_name=SYNTHETIC_POINT_PROMPT_CLASS_NAME,
                )
            )

        batch_detections = attach_prediction_type_info_to_sv_detections_batch(
            predictions=batch_detections,
            prediction_type="instance-segmentation",
        )
        batch_detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=batch_detections,
        )
        return [{"predictions": pred} for pred in batch_detections]
