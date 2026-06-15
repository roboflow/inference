"""Tensor-native sibling of `roboflow_core/segment_anything_2_video@v1`.

SCRATCH — first pass for review. SAM2 Video Tracker is a STATEFUL, LOCAL-only
streaming instance-segmentation producer. It differs from the SAM2 image block:

- No remote path (per-video session state cannot be shipped per-frame).
- The model is loaded directly via AutoModel (inference_models). Its HF
  Sam2VideoProcessor expects HWC RGB; the model's `_ensure_numpy_image`
  (hf_streaming_video.py) now permutes a CHW tensor to HWC before the host
  transfer, so the block passes `WorkflowImageData.tensor_image` (CHW RGB)
  tensor-natively — which also feeds the processor correct RGB (v1's numpy_image
  was HWC BGR).

The tensor-native surface is the IMAGE (CHW RGB tensor), the INPUT boxes
(tensor-native prediction prompts), and the OUTPUT
(inference_models.InstanceDetections with tracker ids), plus the RLE-by-default
mask carriage. The layout-agnostic session/state helpers
(VideoSessionBookkeeping, decide_prompt_vs_track, build_obj_id_metadata_from_boxes,
BoxPromptMetadata) are reused verbatim; only extract_box_prompts and
masks_to_sv_detections get tensor-native equivalents here.
"""

import uuid
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pydantic import ConfigDict, Field

from inference.core.env import GCP_SERVERLESS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
    split_key_point_prediction,
)
from inference.core.workflows.core_steps.models.foundation._streaming_video_common import (
    BoxPromptMetadata,
    VideoSessionBookkeeping,
    build_obj_id_metadata_from_boxes,
    decide_prompt_vs_track,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    DETECTION_ID_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    ROBOFLOW_MODEL_ID_KIND,
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

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]
PREDICTION_TYPE = "instance-segmentation"

SHORT_DESCRIPTION = (
    "Segment and track objects across video frames with SAM2's streaming "
    "camera predictor."
)
LONG_DESCRIPTION = """
Run Segment Anything 2 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.
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
            "search_keywords": ["SAM2", "segment anything", "video", "tracking", "META"],
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
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
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
        description="Streaming SAM2 model id resolved by `inference_models`.",
        examples=[
            "sam2video/tiny",
            "sam2video/small",
            "sam2video/base-plus",
            "sam2video/large",
        ],
    )
    prompt_mode: PromptMode = Field(
        default="first_frame",
        description=(
            "When to consume `boxes` as SAM2 prompts.  `first_frame` prompts "
            "once per session and then tracks; `every_n_frames` re-seeds every "
            "`prompt_interval` frames; `every_frame` re-seeds every frame."
        ),
    )
    prompt_interval: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="For `prompt_mode=every_n_frames`: re-prompt every N frames.",
        examples=[30],
    )
    threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.0,
        description="Minimum confidence for emitted masks.",
        examples=[0.0],
    )
    mask_representation: Literal["rle", "dense"] = Field(
        default="rle",
        description="Carrier for instance masks. RLE (compact) by default; forced to "
        "'rle' on GCP_SERVERLESS regardless of this value.",
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "boxes"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND],
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
                note="Requires a GPU; the streaming SAM2 video model needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [
            "sam2video/small",
            "sam2video/tiny",
            "sam2video/base-plus",
            "sam2video/large",
        ]


class SegmentAnything2VideoBlockV1(WorkflowBlock):
    """Stateful SAM2 streaming video tracking block (tensor-native output)."""

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        if step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "SAM2 Video Tracker only supports LOCAL workflow step execution. "
                "Remote execution would ship each frame to a separate process and "
                "break the per-video SAM2 session that holds the temporal memory."
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
            self._sessions.clear()
        return self._model

    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch],
        model_id: str,
        prompt_mode: PromptMode,
        prompt_interval: int,
        threshold: float,
        mask_representation: Literal["rle", "dense"],
    ) -> BlockResult:
        if GCP_SERVERLESS:
            mask_representation = "rle"
        model = self._get_model(model_id=model_id)
        boxes_iter = boxes if boxes is not None else [None] * len(images)

        results: List[dict] = []
        for single_image, boxes_for_image in zip(images, boxes_iter):
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

            # Tensor-native CHW RGB frame; the model permutes CHW->HWC internally
            # (_ensure_numpy_image) so the HF processor receives correct RGB.
            frame = single_image.tensor_image

            if should_prompt:
                boxes_xyxy, per_box_meta = _extract_box_prompts_tensor(boxes_for_image)
                masks, obj_ids, new_state = model.prompt(
                    image=frame,
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
            elif session.state_dict is not None:
                masks, obj_ids, new_state = model.track(
                    image=frame, state_dict=session.state_dict
                )
                session.state_dict = new_state
                session.frames_since_prompt += 1
            else:
                height, width = single_image._read_shape_without_materialization()
                masks = np.zeros((0, height, width), dtype=bool)
                obj_ids = np.zeros((0,), dtype=np.int64)

            session.last_frame_number = frame_number

            results.append(
                {
                    "predictions": _masks_to_instance_detections(
                        masks=masks,
                        obj_ids=obj_ids,
                        image=single_image,
                        obj_id_metadata=session.obj_id_metadata,
                        threshold=threshold,
                        mask_representation=mask_representation,
                    )
                }
            )
        return results


def _extract_box_prompts_tensor(
    boxes_for_image,
) -> Tuple[List[Tuple[float, float, float, float]], List[BoxPromptMetadata]]:
    """Tensor-native equivalent of extract_box_prompts: flatten a tensor-native
    prediction prompt into xyxy tuples + per-box metadata.

    `boxes` follows the prediction kinds (OD Detections, IS InstanceDetections, or
    the keypoint tuple Tuple[KeyPoints, Optional[Detections]]). A keypoint tuple with
    a missing Detections component is a hard runtime error (split_key_point_prediction
    raises). Empty / absent input returns two empty lists.
    """
    if boxes_for_image is None:
        return [], []
    _key_points, detections = split_key_point_prediction(boxes_for_image)
    n = len(detections)
    if n == 0:
        return [], []
    bboxes_metadata = detections.bboxes_metadata
    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    metas: List[BoxPromptMetadata] = []
    for i in range(n):
        x1, y1, x2, y2 = detections.xyxy[i].tolist()
        boxes_xyxy.append((float(x1), float(y1), float(x2), float(y2)))
        meta = (
            bboxes_metadata[i]
            if bboxes_metadata is not None and i < len(bboxes_metadata)
            else {}
        )
        parent_id = meta.get(DETECTION_ID_KEY)
        metas.append(
            BoxPromptMetadata(
                class_id=int(detections.class_id[i]),
                class_name=str(meta.get(CLASS_NAME_KEY, "foreground")),
                confidence=(
                    float(detections.confidence[i])
                    if detections.confidence is not None
                    else 1.0
                ),
                parent_id=str(parent_id) if parent_id is not None else None,
            )
        )
    return boxes_xyxy, metas


def _masks_to_instance_detections(
    masks: np.ndarray,
    obj_ids: np.ndarray,
    image: WorkflowImageData,
    obj_id_metadata: Dict[int, BoxPromptMetadata],
    threshold: float,
    mask_representation: str,
) -> InstanceDetections:
    """Tensor-native equivalent of masks_to_sv_detections: one InstanceDetections per
    SAM-assigned object, tracker id carried in bboxes_metadata. Masks with no positive
    pixels, or whose forwarded confidence < threshold, are dropped."""
    height, width = image._read_shape_without_materialization()
    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    class_names_map: Dict[int, str] = {}
    bboxes_metadata: List[dict] = []
    kept_masks: List[np.ndarray] = []

    for mask, obj_id in zip(masks, obj_ids.tolist()):
        meta = obj_id_metadata.get(int(obj_id))
        confidence = meta.confidence if meta is not None else 1.0
        if confidence < threshold:
            continue
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        class_id = int(meta.class_id) if meta is not None else 0
        class_name = meta.class_name if meta is not None else "foreground"
        xyxy.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        confidences.append(float(confidence))
        class_ids.append(class_id)
        class_names_map[class_id] = class_name
        bboxes_metadata.append(
            {
                DETECTION_ID_KEY: str(uuid.uuid4()),
                CLASS_NAME_KEY: class_name,
                TRACKER_ID_KEY: int(obj_id),
            }
        )
        kept_masks.append(mask.astype(bool))

    n = len(kept_masks)
    if n == 0:
        xyxy_t = torch.zeros((0, 4), dtype=torch.float32)
        class_id_t = torch.zeros((0,), dtype=torch.int64)
        confidence_t = torch.zeros((0,), dtype=torch.float32)
        mask = (
            InstancesRLEMasks(image_size=(height, width), masks=[])
            if mask_representation == "rle"
            else torch.zeros((0, height, width), dtype=torch.bool)
        )
    else:
        xyxy_t = torch.tensor(xyxy, dtype=torch.float32)
        class_id_t = torch.tensor(class_ids, dtype=torch.int64)
        confidence_t = torch.tensor(confidences, dtype=torch.float32)
        if mask_representation == "rle":
            rle_dicts = [torch_mask_to_coco_rle(torch.from_numpy(m)) for m in kept_masks]
            mask = InstancesRLEMasks.from_coco_rle_masks(
                image_size=(height, width), masks=rle_dicts
            )
        else:
            mask = torch.from_numpy(np.stack(kept_masks, axis=0))

    detections = InstanceDetections(
        xyxy=xyxy_t, class_id=class_id_t, confidence=confidence_t, mask=mask
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names_map,
        prediction_type=PREDICTION_TYPE,
        inference_id=str(uuid.uuid4()),
    )
    detections.bboxes_metadata = bboxes_metadata if n else None
    return detections
