"""SAM2 Video Tracker workflow block.

Wraps ``inference_models``'s ``SAM2Video`` (the HuggingFace streaming
tracker) so it can be driven from a workflow powered by
``InferencePipeline``.  The pipeline delivers one frame at a time with
``WorkflowImageData.video_metadata``; this block keeps one state_dict
per ``video_identifier`` and either re-prompts or propagates existing
tracks on each frame.

Prompt modes (see ``prompt_mode``):

``first_frame``
    Consume ``boxes`` once per session, then track silently.
``every_n_frames``
    Re-seed every ``prompt_interval`` frames (counted from the last
    prompt).
``every_frame``
    Re-seed on every frame — effectively turns the block into a
    per-frame detector→mask adapter with SAM-stable tracker ids.

``boxes`` is ignored on frames where we only propagate.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
)
from inference.core.workflows.core_steps.models.foundation._streaming_video_common import (
    VideoSessionBookkeeping,
    build_obj_id_metadata_from_boxes,
    decide_prompt_vs_track,
    extract_box_prompts,
    masks_to_sv_detections,
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
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]

SHORT_DESCRIPTION = (
    "Segment and track objects across video frames with SAM2's streaming "
    "camera predictor."
)

LONG_DESCRIPTION = """
Run Segment Anything 2 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.

Feed box detections from an upstream detector (e.g. a YOLO block) as
prompts.  The block multiplexes a single SAM2 camera predictor across
many video streams by keying state on `video_metadata.video_identifier`;
depending on `prompt_mode`, it either re-seeds the prompts periodically
or simply propagates existing tracks.

Intended for use with `InferencePipeline`, which delivers one frame at
a time and tags each frame with video metadata.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM2 Video Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "SAM2",
                "segment anything",
                "video",
                "tracking",
                "META",
            ],
            "ui_manifest": {
                "section": "video",
                "icon": "fa-brands fa-meta",
                "blockPriority": 9.4,
                "needsGPU": True,
                "inference": True,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/segment_anything_2_video@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    boxes: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        description=(
            "Bounding boxes to use as SAM2 prompts.  Only read on frames "
            "where the block re-prompts (see `prompt_mode`)."
        ),
        examples=["$steps.object_detection_model.predictions"],
        default=None,
        json_schema_extra={"always_visible": True},
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="sam2video/small",
        description=(
            "Streaming SAM2 model id resolved by `inference_models`.  "
            "The `sam2video` family ships four Hiera backbone sizes; "
            "`small` is the default trade-off between speed and quality."
        ),
        examples=[
            "sam2video/tiny",
            "sam2video/small",
            "sam2video/base-plus",
            "sam2video/large",
        ],
    )
    prompt_mode: Literal["first_frame", "every_n_frames", "every_frame"] = Field(
        default="first_frame",
        description=(
            "When to consume `boxes` as SAM2 prompts.  `first_frame` "
            "prompts once per session and then tracks; `every_n_frames` "
            "re-seeds every `prompt_interval` frames; `every_frame` "
            "re-seeds every frame.  On frames where re-seeding does not "
            "happen, `boxes` is ignored and the block simply propagates."
        ),
    )
    prompt_interval: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="For `prompt_mode=every_n_frames`: re-prompt every N frames.",
        examples=[30],
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.0,
        description="Minimum confidence for emitted masks.",
        examples=[0.0],
    )

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
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [
            "sam2video/small",
            "sam2video/tiny",
            "sam2video/base-plus",
            "sam2video/large",
        ]


class SegmentAnything2VideoBlockV1(WorkflowBlock):
    """Stateful SAM2 streaming video tracking block."""

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        if step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "SAM2 Video Tracker only supports LOCAL workflow step "
                "execution.  Remote execution would ship each frame to a "
                "separate process and break the per-video SAM2 session "
                "that holds the temporal memory.  Set "
                "WORKFLOWS_STEP_EXECUTION_MODE=local (or run on a "
                "dedicated deployment) to use this block."
            )
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._model = None  # lazily loaded
        self._current_model_id: Optional[str] = None
        self._sessions: Dict[str, VideoSessionBookkeeping] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_model(self, model_id: str):
        if self._model is None or self._current_model_id != model_id:
            from inference_models import AutoModel

            self._model = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self._api_key,
            )
            self._current_model_id = model_id
            # Switching model invalidates every session we held.
            self._sessions.clear()
        return self._model

    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch[sv.Detections]],
        model_id: str,
        prompt_mode: PromptMode,
        prompt_interval: int,
        threshold: float,
    ) -> BlockResult:
        model = self._get_model(model_id=model_id)
        if boxes is None:
            boxes = [None] * len(images)

        batch_detections: List[sv.Detections] = []
        for single_image, boxes_for_image in zip(images, boxes):
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0

            session = self._sessions.setdefault(video_id, VideoSessionBookkeeping())
            has_box_prompts = boxes_for_image is not None and len(boxes_for_image) > 0
            should_reset, should_prompt = decide_prompt_vs_track(
                session=session,
                frame_number=frame_number,
                prompt_mode=prompt_mode,
                prompt_interval=prompt_interval,
                has_prompts=has_box_prompts,
            )
            if should_reset:
                session.state_dict = None
                session.obj_id_metadata = {}
                session.frames_since_prompt = 0

            frame_np = single_image.numpy_image

            if should_prompt:
                boxes_xyxy, per_box_meta = extract_box_prompts(boxes_for_image)
                masks, obj_ids, new_state = model.prompt(
                    image=frame_np,
                    bboxes=boxes_xyxy,
                    state_dict=session.state_dict,
                    # clear old points is implied by our own reset gating
                    clear_old_prompts=True,
                    frame_idx=frame_number,
                )
                session.obj_id_metadata = build_obj_id_metadata_from_boxes(
                    obj_ids=obj_ids, box_metas=per_box_meta
                )
                session.state_dict = new_state
                session.frames_since_prompt = 0
            elif session.state_dict is not None:
                masks, obj_ids, new_state = model.track(
                    image=frame_np, state_dict=session.state_dict
                )
                session.state_dict = new_state
                session.frames_since_prompt += 1
            else:
                import numpy as np

                masks = np.zeros((0, frame_np.shape[0], frame_np.shape[1]), dtype=bool)
                obj_ids = np.zeros((0,), dtype=np.int64)

            session.last_frame_number = frame_number

            batch_detections.append(
                masks_to_sv_detections(
                    masks=masks,
                    obj_ids=obj_ids,
                    image=single_image,
                    obj_id_metadata=session.obj_id_metadata,
                    threshold=threshold,
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
