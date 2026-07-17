"""Tensor-native sibling of ``roboflow_core/sam3_video@v1``, loaded when
``ENABLE_TENSOR_DATA_REPRESENTATION`` is on.

SAM3 Video Tracker is a STATEFUL, LOCAL-only streaming open-vocabulary
concept tracker (see the numpy source for the full behavioural notes).
The session bookkeeping, prompt-signature reseeding and prompt/track state
machine are verbatim copies of the numpy source; only the prediction
representation is rewritten:

- INPUT frame: when the workflow image is tensor-materialised the block
  forwards ``WorkflowImageData.tensor_image`` (CHW RGB) directly —
  ``SAM3Video``'s ``_ensure_numpy_image`` permutes CHW->HWC WITHOUT
  reinterpreting channel order, so the HF processor receives HWC RGB (the
  numpy sibling fed HWC BGR ``numpy_image``; the tensor path is the correct
  colour order — the same cross-repo contract as the SAM2 video sibling).
  Non-materialised inputs are flipped BGR->RGB on host.
- OUTPUT: native ``inference_models.InstanceDetections`` instead of
  ``sv.Detections``. Masks are carried as compact ``InstancesRLEMasks`` by
  default; the carrier is an execution-level choice driven by the
  ``WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION`` env variable ("rle"/"dense",
  same convention as the SAM2 video sibling; GCP_SERVERLESS forces "rle") —
  NOT a manifest field, so the manifest stays identical to the numpy
  sibling. Per-box
  ``bboxes_metadata`` carries ``detection_id`` / ``class`` / ``tracker_id``;
  ``image_metadata`` is built via ``build_native_image_metadata`` with the
  full prompt-position ``class_id -> name`` map (``class_id`` is the
  prompt's index in ``class_names``, exactly like the numpy sibling), which
  replaces ``attach_prediction_type_info_to_sv_detections_batch`` +
  ``attach_parents_coordinates_to_batch_of_sv_detections``.
- Objects whose id maps to no registered prompt keep the numpy fallback:
  ``class_id=0`` with per-box class name ``"foreground"`` (the per-box name
  wins at serialization, so output parity holds even on collision with
  ``class_names[0]``).
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pydantic import ConfigDict, Field

from inference.core.env import (
    GCP_SERVERLESS,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
)
from inference.core.workflows.core_steps.models.foundation._streaming_video_common import (
    normalise_class_names,
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
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
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
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

PREDICTION_TYPE = "instance-segmentation"


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
    "Segment and track objects across video frames from text prompts with "
    "SAM3's streaming concept tracker."
)

LONG_DESCRIPTION = """
Run Segment Anything 3 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.

Provide the concepts to track as text in `class_names` (e.g.
`["person", "forklift"]`) — no upstream detector is needed.  SAM3 runs
fused detection and tracking on every frame, so objects matching a
concept that enter the scene mid-stream are picked up automatically and
assigned fresh `tracker_id`s.  Each emitted mask carries the prompt it
matched as its class name and the model's detection score as its
confidence.

The block multiplexes a single SAM3 streaming model across many video
streams by keying state on `video_metadata.video_identifier`; a session
is re-seeded only when the source stream restarts or `class_names`
changes.  For detector-driven (box-prompted) video tracking, use the
SAM2 Video Tracker block instead.

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
    class_names: Union[
        List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])
    ] = Field(
        description=(
            "Concepts to segment and track, as a list of phrases (or a "
            "single comma-separated string).  Each emitted mask carries "
            "the concept it matched as its class name."
        ),
        examples=[["person", "forklift"]],
        json_schema_extra={"always_visible": True},
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="sam3video",
        description="Streaming SAM3 model id resolved by `inference_models`.",
        examples=["sam3video"],
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.5,
        description=(
            "Minimum detection score for emitted masks.  Scores come "
            "from SAM3's per-object concept detection head."
        ),
        examples=[0.5],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

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
                note="Requires a GPU; the streaming SAM3 video model needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return ["sam3video"]


class SegmentAnything3VideoBlockV1(WorkflowBlock):
    """Stateful SAM3 streaming concept tracking block (tensor-native output)."""

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
        self._sessions: Dict[str, _ConceptSessionBookkeeping] = {}

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
            # Switching model invalidates every session we held.
            self._sessions.clear()
        return self._model

    def run(
        self,
        images: Batch[WorkflowImageData],
        class_names: Union[List[str], str],
        model_id: str,
        threshold: float,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        model = self._get_model(model_id=model_id)
        class_list = normalise_class_names(class_names)
        prompt_signature = tuple(class_list)

        batch_predictions: List[InstanceDetections] = []
        for single_image in images:
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0

            session = self._sessions.setdefault(video_id, _ConceptSessionBookkeeping())
            stream_restarted = (
                session.last_frame_number >= 0
                and frame_number < session.last_frame_number
            )
            if stream_restarted or session.prompt_signature != prompt_signature:
                session.state_dict = None

            # Same cross-repo contract as the SAM2 video sibling: SAM3Video's
            # _ensure_numpy_image permutes a CHW tensor to HWC WITHOUT swapping
            # channels and the HF processor expects RGB, so the materialised
            # CHW RGB tensor is forwarded as-is, and a host-only frame is
            # flipped BGR->RGB before the call.
            if single_image.is_tensor_materialised():
                frame = single_image.tensor_image
                if frame.dim() != 3:
                    raise ValueError(
                        "SAM3 video tracker expects a CHW (3-D) RGB frame tensor; got "
                        f"a tensor with {frame.dim()} dim(s). The model's "
                        "_ensure_numpy_image permutes CHW->HWC and assumes this layout."
                    )
            else:
                frame = np.ascontiguousarray(single_image.numpy_image[:, :, ::-1])

            if not class_list:
                predictions = _empty_instance_detections(
                    image=single_image, class_names=class_list
                )
            else:
                if session.state_dict is None:
                    result = model.prompt(
                        image=frame,
                        text=class_list,
                        clear_old_prompts=True,
                    )
                    session.prompt_signature = prompt_signature
                else:
                    result = model.track(image=frame, state_dict=session.state_dict)
                session.state_dict = result.state_dict
                predictions = _concept_frame_to_instance_detections(
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
            batch_predictions.append(predictions)

        return [{"predictions": predictions} for predictions in batch_predictions]


def _concept_frame_to_instance_detections(
    masks: np.ndarray,
    object_ids: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    prompt_to_object_ids: Dict[str, List[int]],
    class_names: List[str],
    image: WorkflowImageData,
    threshold: float,
) -> InstanceDetections:
    """Tensor-native equivalent of ``concept_frame_to_sv_detections``: assemble one
    ``InstanceDetections`` from one SAM3 concept-tracker frame.

    Class labels and confidences are rebuilt every frame from
    ``prompt_to_object_ids`` / ``scores`` (mid-stream objects get labelled
    correctly); ``class_id`` is the prompt's position in ``class_names``. Masks
    with no positive pixels, or whose score < ``threshold``, are dropped. Kept
    masks are carried as compact ``InstancesRLEMasks``.
    """
    if masks.shape[0] == 0:
        return _empty_instance_detections(image=image, class_names=class_names)

    object_id_to_prompt = {
        int(obj_id): prompt
        for prompt, obj_ids in prompt_to_object_ids.items()
        for obj_id in obj_ids
    }

    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    bboxes_metadata: List[dict] = []
    kept_masks: List[np.ndarray] = []

    for mask, obj_id, score, box in zip(
        masks, object_ids.tolist(), scores.tolist(), boxes.tolist()
    ):
        if score < threshold:
            continue
        if not mask.any():
            continue
        prompt = object_id_to_prompt.get(int(obj_id), "foreground")
        try:
            class_id = class_names.index(prompt)
        except ValueError:
            class_id = 0
        xyxy.append([float(v) for v in box[:4]])
        confidences.append(float(score))
        class_ids.append(class_id)
        bboxes_metadata.append(
            {
                DETECTION_ID_KEY: str(uuid.uuid4()),
                CLASS_NAME_KEY: prompt,
                TRACKER_ID_KEY: int(obj_id),
            }
        )
        kept_masks.append(mask.astype(bool))

    if not kept_masks:
        return _empty_instance_detections(image=image, class_names=class_names)

    height, width = image._read_shape_without_materialization()
    if _resolve_mask_representation() == "rle":
        rle_dicts = [torch_mask_to_coco_rle(torch.from_numpy(m)) for m in kept_masks]
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(height, width), masks=rle_dicts
        )
    else:
        mask = torch.from_numpy(np.stack(kept_masks, axis=0)).to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
    detections = InstanceDetections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        class_id=torch.tensor(class_ids, dtype=torch.int64).to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.tensor(confidences, dtype=torch.float32).to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        mask=mask,
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=_prompt_class_names_map(class_names),
        prediction_type=PREDICTION_TYPE,
    )
    detections.bboxes_metadata = bboxes_metadata
    return detections


def _empty_instance_detections(
    image: WorkflowImageData,
    class_names: List[str],
) -> InstanceDetections:
    height, width = image._read_shape_without_materialization()
    if _resolve_mask_representation() == "rle":
        mask = InstancesRLEMasks(image_size=(height, width), masks=[])
    else:
        mask = torch.zeros((0, height, width), dtype=torch.bool).to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
    detections = InstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32).to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        class_id=torch.zeros((0,), dtype=torch.int64).to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        confidence=torch.zeros((0,), dtype=torch.float32).to(
            WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        mask=mask,
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=_prompt_class_names_map(class_names),
        prediction_type=PREDICTION_TYPE,
    )
    detections.bboxes_metadata = None
    return detections


def _prompt_class_names_map(class_names: List[str]) -> Dict[int, str]:
    """``class_id -> name`` over the FULL prompt list (ids are prompt positions,
    stable for a given prompt set). Unmatched-prompt objects fall back to
    ``class_id=0`` with a per-box ``"foreground"`` label, which wins over this map
    at serialization (C1 convention)."""
    return {class_id: class_name for class_id, class_name in enumerate(class_names)}
