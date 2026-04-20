"""SAM3 Video Tracker workflow block.

Same shape as the SAM2 video block but backed by
``inference_models``'s ``SAM3ForStream``, which uses the HuggingFace
transformers port of SAM3 video for true frame-by-frame streaming.

In addition to the SAM2 block's bbox-prompt path, SAM3 accepts open
vocabulary text prompts via ``class_names``.  When both are supplied,
box prompts win on frames where the block re-prompts.
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
    build_obj_id_metadata_from_text,
    decide_prompt_vs_track,
    extract_box_prompts,
    masks_to_sv_detections,
    normalise_class_names,
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
    LIST_OF_VALUES_KIND,
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
    "Segment and track objects across video frames with SAM3's streaming "
    "video predictor.  Supports text and box prompts."
)

LONG_DESCRIPTION = """
Run Segment Anything 3 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.

Two kinds of prompts are accepted:

- **Text prompts** (`class_names`): SAM3's open-vocabulary text
  interface.  Matching objects are segmented and tracked across
  subsequent frames.
- **Box prompts** (`boxes`): bounding boxes from an upstream detector.
  Each box is converted to a mask and tracked.  If both `class_names`
  and `boxes` are supplied, `boxes` wins on frames where the block
  re-prompts.

Intended for use with `InferencePipeline`, which delivers one frame at
a time and tags each frame with video metadata.
"""


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
                "META",
            ],
            "ui_manifest": {
                "section": "video",
                "icon": "fa-solid fa-eye",
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
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="segment-anything-3-rt",
        description=(
            "Streaming SAM3 model id resolved by `inference_models`."
        ),
        examples=["segment-anything-3-rt"],
    )
    class_names: Optional[
        Union[List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])]
    ] = Field(
        default=None,
        description=(
            "Text prompts — list of class names to segment and track.  "
            "Only used when no `boxes` are supplied for the re-prompt "
            "frame."
        ),
        examples=[["person", "car"]],
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
            "Bounding boxes from an upstream detector to use as SAM3 "
            "prompts.  Only read on frames where the block re-prompts."
        ),
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={"always_visible": True},
    )
    prompt_mode: Literal["first_frame", "every_n_frames", "every_frame"] = Field(
        default="first_frame",
        description=(
            "When to (re-)seed prompts.  `first_frame` prompts once per "
            "session then tracks; `every_n_frames` re-seeds every "
            "`prompt_interval` frames; `every_frame` re-seeds every frame."
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
        return ["segment-anything-3-rt"]


class SegmentAnything3VideoBlockV1(WorkflowBlock):
    """Stateful SAM3 streaming video tracking block."""

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        if step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "SAM3 Video Tracker only supports LOCAL workflow step "
                "execution.  Remote execution would ship each frame to a "
                "separate process and break the per-video SAM3 session "
                "that holds the temporal memory.  Set "
                "WORKFLOWS_STEP_EXECUTION_MODE=local (or run on a "
                "dedicated deployment) to use this block."
            )
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._model = None
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
            self._sessions.clear()
        return self._model

    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch[sv.Detections]],
        model_id: str,
        class_names: Optional[Union[List[str], str]],
        prompt_mode: PromptMode,
        prompt_interval: int,
        threshold: float,
    ) -> BlockResult:
        model = self._get_model(model_id=model_id)
        class_name_list = normalise_class_names(class_names)
        if boxes is None:
            boxes = [None] * len(images)

        batch_detections: List[sv.Detections] = []
        for single_image, boxes_for_image in zip(images, boxes):
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0

            session = self._sessions.setdefault(
                video_id, VideoSessionBookkeeping()
            )
            has_box_prompts = (
                boxes_for_image is not None and len(boxes_for_image) > 0
            )
            has_text_prompts = bool(class_name_list)
            has_any_prompt = has_box_prompts or has_text_prompts

            should_reset, should_prompt = decide_prompt_vs_track(
                session=session,
                frame_number=frame_number,
                prompt_mode=prompt_mode,
                prompt_interval=prompt_interval,
                has_prompts=has_any_prompt,
            )
            if should_reset:
                session.state_dict = None
                session.obj_id_metadata = {}
                session.frames_since_prompt = 0

            frame_np = single_image.numpy_image

            if should_prompt and has_box_prompts:
                boxes_xyxy, per_box_meta = extract_box_prompts(boxes_for_image)
                masks, obj_ids, new_state = model.prompt(
                    image=frame_np,
                    bboxes=boxes_xyxy,
                    state_dict=session.state_dict,
                    clear_old_prompts=True,
                    frame_idx=frame_number,
                )
                session.obj_id_metadata = build_obj_id_metadata_from_boxes(
                    obj_ids=obj_ids, box_metas=per_box_meta
                )
                session.state_dict = new_state
                session.frames_since_prompt = 0
            elif should_prompt and has_text_prompts:
                text = ", ".join(class_name_list)
                masks, obj_ids, new_state = model.prompt(
                    image=frame_np,
                    text=text,
                    state_dict=session.state_dict,
                    clear_old_prompts=True,
                    frame_idx=frame_number,
                )
                session.obj_id_metadata = build_obj_id_metadata_from_text(
                    obj_ids=obj_ids, class_names=class_name_list
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

                masks = np.zeros(
                    (0, frame_np.shape[0], frame_np.shape[1]), dtype=bool
                )
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
