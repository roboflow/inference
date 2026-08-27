"""Tensor-native sibling of `roboflow_core/segment_anything_2_video@v1`.

SAM2 Video Tracker is a STATEFUL, LOCAL-only streaming instance-segmentation
producer. It differs from the SAM2 image block:

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
mask carriage (an execution-level choice driven by the
WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION env variable — NOT a manifest field,
so the manifest stays identical to the numpy sibling; GCP_SERVERLESS forces
"rle"). This sibling does not decorate ``run()`` with
``@usage_collector("model")``, so flag-on deployments emit no model-category
usage row for SAM2 video. The layout-agnostic session/state helpers
(VideoSessionBookkeeping, decide_prompt_vs_track, build_obj_id_metadata_from_boxes,
BoxPromptMetadata) are shared with the NumPy sibling. Tensor-native prompt
extraction and prediction assembly live in
``segment_anything_common.streaming_video_tensor``.
"""

from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.env import GCP_SERVERLESS, WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    VideoSessionBookkeeping,
    build_obj_id_metadata_from_boxes,
    decide_prompt_vs_track,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common.streaming_video_tensor import (
    extract_box_prompts_tensor,
    masks_to_instance_detections,
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

PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]


def _resolve_mask_representation() -> str:
    """Execution-level selection of the instance-mask carrier ("rle"/"dense").

    Deliberately NOT a manifest field: the numpy sibling has no such knob and
    manifests must stay identical across the flag swap. Driven by the
    ``WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION`` env variable ("rle" default);
    ``GCP_SERVERLESS`` forces the compact RLE carrier regardless.
    """
    if GCP_SERVERLESS:
        return "rle"
    return WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION


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

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "SAM2 Video Tracker only supports LOCAL workflow step "
        "execution.  Remote execution would ship each frame to a "
        "separate process and break the per-video SAM2 session "
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

            extra_weights_provider_headers = get_extra_weights_provider_headers()
            self._model = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self._api_key,
                weights_provider_extra_headers=extra_weights_provider_headers,
            )
            self._current_model_id = model_id
            self._sessions.clear()
        return self._model

    # No @usage_collector("model") here: the numpy sibling decorates run(),
    # but this tensor-native path is left without a model-category usage row.
    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch],
        model_id: str,
        prompt_mode: PromptMode,
        prompt_interval: int,
        threshold: float,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        mask_representation = _resolve_mask_representation()
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

            # CROSS-REPO CONTRACT (intentional divergence from the numpy sibling):
            #   - This block forwards the tensor-native CHW *RGB* frame.
            #   - inference_models hf_streaming_video._ensure_numpy_image is
            #     contracted to permute CHW->HWC WITHOUT reinterpreting channel
            #     order, so the HF Sam2VideoProcessor receives HWC RGB.
            #   - The numpy v1 sibling fed HWC *BGR* (numpy_image); the HF
            #     processor expects RGB, so the tensor path is the correct one.
            # This correctness now hinges on _ensure_numpy_image NOT swapping
            # channels: a regression there silently corrupts every mask. The
            # `frame.dim() == 3` guard below catches a layout regression (a
            # non-CHW frame) early instead of letting a malformed tensor reach
            # the model. Channel-order parity itself is pinned by a focused test
            # in inference_models (see hf_streaming_video tests); it cannot be
            # asserted from a workflow block without materialising the frame.
            # Use the tensor when already materialised, otherwise hand the model an RGB
            # host frame: _ensure_numpy_image permutes CHW->HWC WITHOUT swapping channels
            # and the HF processor expects RGB, so the BGR numpy frame is flipped here
            # (avoids a forced CHW transpose + H2D the model would undo anyway).
            if single_image.is_tensor_materialised():
                frame = single_image.tensor_image
                if frame.dim() != 3:
                    raise ValueError(
                        "SAM2 video tracker expects a CHW (3-D) RGB frame tensor; got "
                        f"a tensor with {frame.dim()} dim(s). The model's "
                        "_ensure_numpy_image permutes CHW->HWC and assumes this layout."
                    )
            else:
                frame = np.ascontiguousarray(single_image.numpy_image[:, :, ::-1])

            if should_prompt:
                boxes_xyxy, per_box_meta = extract_box_prompts_tensor(boxes_for_image)
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
                    "predictions": masks_to_instance_detections(
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
