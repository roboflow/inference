"""Tensor-native sibling of ``segment_anything3_interactive/v1.py``, loaded when
``ENABLE_TENSOR_DATA_REPRESENTATION`` is on.

The numpy source drives SAM3 PVS through SAM2-style requests against
``sam3/sam3_interactive`` and expands each polygon of the response into a
separate ``sv.Detections`` row. Tensor-native differences:

- ``boxes`` prompts arrive as native ``Detections`` / ``InstanceDetections`` /
  ``(KeyPoints, Detections)`` — box coordinates are read from the ``xyxy``
  tensor, class names resolved via per-box ``CLASS_NAME_KEY`` override or the
  ``image_metadata[CLASS_NAMES_KEY]`` map, ``detection_id`` from
  ``bboxes_metadata`` (mirroring ``segment_anything2/v1_tensor.py``).
- LOCAL goes through ``ModelManager.run_tensor_native_inference`` (the
  ``InferenceModelsSAM3InteractiveAdapter`` ``action="segment"`` bridge to
  ``SAM3Torch.segment_with_visual_prompts``): tensor-resident images are handed
  over directly (RGB; host-only images are flipped BGR→RGB), prompts are
  converted with the exact ``to_sam2_inputs`` + padding chain the legacy
  adapter applied, and raw logits are binarised at the adapter's
  ``MASK_THRESHOLD`` — instances carry the model's RAW binary mask, one
  instance per prompt, no polygon collapse (the family convention; numpy
  splits a multi-region mask into one instance per polygon). REMOTE asks for
  ``mask_input_format="rle"``; the proxy path (``SAM3_EXEC_MODE=remote``)
  requests RLE too but falls back to rasterising polygon-format predictions if
  the proxy strips the ``format`` field.
- Masks are carried as ``InstancesRLEMasks`` (compact, serverless-safe);
  ``xyxy`` is the mask's tight bbox (inclusive min/max, no +1 — family-wide
  convention keeping numpy-faithful boxes).
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import requests
import torch
from pydantic import ConfigDict, Field, model_validator

from inference.core import logger
from inference.core.entities.requests.sam2 import Box, Point, Sam2Prompt, Sam2PromptSet
from inference.core.env import (
    API_BASE_URL,
    CORE_MODEL_SAM3_ENABLED,
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    SAM3_EXEC_MODE,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import build_roboflow_api_headers
from inference.core.utils.url_utils import wrap_url
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
    split_key_point_prediction,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
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
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    LABELED_POINTS_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)
from inference_sdk import InferenceHTTPClient

DETECTIONS_CLASS_NAME_FIELD = "class_name"
DETECTION_ID_FIELD = "detection_id"

SAM3_INTERACTIVE_MODEL_ID = "sam3/sam3_interactive"

PREDICTION_TYPE = "instance-segmentation"

# The legacy adapter's MASK_THRESHOLD: raw logits from the model are binarised
# at this cutoff (see `_build_rle_response` in
# inference/models/sam3/visual_segmentation_inference_models.py).
SAM3_MASK_LOGITS_THRESHOLD = 0.0

SHORT_DESCRIPTION = (
    "Segment a specific object with SAM3 using point and/or bounding box prompts."
)

LONG_DESCRIPTION = """
Run the interactive (promptable visual segmentation) head of Segment Anything 3 (SAM3) on an image.

Unlike the SAM 3 concept segmentation block (which takes text or exemplar prompts and returns
ALL instances of a concept), this block performs SAM2-style interactive segmentation: each prompt
targets ONE object and the model returns a single mask for it.

Two prompt inputs are supported (at least one must be provided):
- **points**: a list of labeled 2D points defining a single object. Positive points mark the
  object to segment, negative points mark regions to exclude (useful to refine the mask).
- **boxes**: detections from another model. Each bounding box becomes a separate prompt and
  the model segments the object inside it. Class names of the boxes are forwarded to the
  predicted masks.
"""


def _as_sam2_points(points: List[Any]) -> List[Point]:
    """Normalise points given as dicts or (x, y[, positive]) sequences into Sam2 Points."""
    result = []
    for raw_point in points:
        if isinstance(raw_point, Point):
            result.append(raw_point)
            continue
        if isinstance(raw_point, dict):
            if "x" not in raw_point or "y" not in raw_point:
                raise ValueError(
                    f"Each point prompt must define `x` and `y` coordinates - got: {raw_point}"
                )
            x, y = raw_point["x"], raw_point["y"]
            positive = raw_point.get("positive", True)
        elif isinstance(raw_point, (list, tuple)) and len(raw_point) in {2, 3}:
            x, y = raw_point[0], raw_point[1]
            positive = raw_point[2] if len(raw_point) == 3 else True
        else:
            raise ValueError(
                f"Invalid point prompt: {raw_point}. Expected dict with `x`, `y` and optional "
                f"`positive` keys, or a sequence of (x, y) or (x, y, positive)."
            )
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError(f"Point coordinates must be numbers - got: {raw_point}")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Point coordinates must be numbers - got: {raw_point}")
        result.append(Point(x=float(x), y=float(y), positive=bool(positive)))
    return result


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3 Interactive",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Sam",
                "SAM3",
                "segment anything",
                "segment anything 3",
                "point prompt",
                "interactive segmentation",
                "PVS",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fa-solid fa-eye",
                "blockPriority": 9.47,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/sam3_interactive@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    points: Optional[Union[List[Any], Selector(kind=[LABELED_POINTS_KIND])]] = Field(
        default=None,
        title="Point Prompts",
        description="Labeled points defining a single object to segment. "
        "Each point is {'x': ..., 'y': ..., 'positive': ...} in absolute pixel coordinates - "
        "positive points mark the object, negative points mark regions to exclude. "
        "Plain (x, y) or (x, y, positive) sequences are also accepted.",
        examples=[
            [{"x": 320, "y": 240, "positive": True}],
            "$inputs.points",
        ],
        json_schema_extra={"always_visible": True},
    )
    boxes: Optional[
        Selector(
            kind=[
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        default=None,
        description="Bounding boxes (from another model) to use as prompts - "
        "the model segments the object inside each box",
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={"always_visible": True},
    )
    threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.0,
        description="Minimum confidence threshold for predicted masks",
        examples=[0.3],
    )
    multimask_output: Union[Optional[bool], Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Flag to determine whether to use SAM3 internal multimask or single mask mode. "
        "For ambiguous prompts (like a single point) setting to True is recommended.",
        examples=[True, "$inputs.multimask_output"],
    )

    @model_validator(mode="after")
    def _validate_points(self) -> "BlockManifest":
        if isinstance(self.points, list):
            _as_sam2_points(self.points)
        return self

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
        restrictions = [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; run_locally() loads a model that needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ]
        if not CORE_MODEL_SAM3_ENABLED:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.HARD,
                    note=(
                        "CORE_MODEL_SAM3_ENABLED=False on Roboflow Hosted "
                        "Serverless: the SAM3 endpoint is not registered, so "
                        "run_remotely() returns 404."
                    ),
                    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
                )
            )
        return restrictions

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return list of model_id variants that can satisfy this block."""
        return [SAM3_INTERACTIVE_MODEL_ID]


class SegmentAnything3InteractiveBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        if SAM3_EXEC_MODE == "remote":
            logger.debug(
                "Running SAM3 Interactive via inference proxy (SAM3_EXEC_MODE=remote)"
            )
            return self.run_via_request(
                images=images,
                points=points,
                boxes=boxes,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                points=points,
                boxes=boxes,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        if self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                points=points,
                boxes=boxes,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        if boxes is None:
            boxes = [None] * len(images)

        self._model_manager.add_model(
            model_id=SAM3_INTERACTIVE_MODEL_ID,
            api_key=self._api_key,
        )

        results: List[dict] = []
        for single_image, boxes_for_image in zip(images, boxes):
            groups = self._build_prompt_groups(
                boxes_for_image=boxes_for_image, points=points
            )
            if single_image.is_tensor_materialised():
                # SAM3Torch normalises CHW/HWC itself and expects RGB channel
                # order - the tensor representation is already RGB.
                model_image = single_image.tensor_image
            else:
                # The numpy representation is BGR; flip to the RGB that the
                # legacy `load_image_rgb` preprocessing produced.
                model_image = np.ascontiguousarray(single_image.numpy_image[:, :, ::-1])
            raw_masks, confidences, class_ids, class_names, detection_ids = (
                [],
                [],
                [],
                [],
                [],
            )
            for group in groups:
                model_inputs = _prompt_group_to_model_inputs(prompts=group.prompts)
                sam3_predictions = self._model_manager.run_tensor_native_inference(
                    SAM3_INTERACTIVE_MODEL_ID,
                    action="segment",
                    images=[model_image],
                    point_coordinates=model_inputs["point_coords"],
                    point_labels=model_inputs["point_labels"],
                    boxes=model_inputs["box"],
                    multi_mask_output=multimask_output,
                    # Raw logits, binarised below at the legacy adapter's
                    # MASK_THRESHOLD - the family's explicit-binarisation
                    # convention (see segment_anything2/v1_tensor.py).
                    return_logits=True,
                )
                prediction = sam3_predictions[0]
                binary_masks = prediction.masks >= SAM3_MASK_LOGITS_THRESHOLD
                scores = prediction.scores.detach().to("cpu").tolist()
                for instance_mask, score in zip(binary_masks, scores):
                    raw_masks.append(instance_mask)
                    confidences.append(float(score))
                class_ids.extend(group.class_ids)
                class_names.extend(group.class_names)
                detection_ids.extend(group.detection_ids)
            instance_detections = _segmentation_results_to_instance_detections(
                raw_masks=raw_masks,
                confidences=confidences,
                class_ids=class_ids,
                class_names=class_names,
                detection_ids=detection_ids,
                image=single_image,
                threshold=threshold,
            )
            results.append({"predictions": instance_detections})
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()

        if boxes is None:
            boxes = [None] * len(images)

        results: List[dict] = []
        for single_image, boxes_for_image in zip(images, boxes):
            groups = self._build_prompt_groups(
                boxes_for_image=boxes_for_image, points=points
            )
            raw_masks, confidences, class_ids, class_names, detection_ids = (
                [],
                [],
                [],
                [],
                [],
            )
            for group in groups:
                result = client.sam3_visual_segment(
                    inference_input=single_image.base64_image,
                    prompts=[
                        prompt.dict(exclude_none=True) for prompt in group.prompts
                    ],
                    multimask_output=multimask_output,
                    # Compact transfer + lossless masks (same as the SAM2 tensor
                    # sibling's remote path).
                    mask_input_format="rle",
                )
                for masks_payload, confidence in _parse_segmentation_predictions(
                    result
                ):
                    raw_masks.append(masks_payload)
                    confidences.append(confidence)
                class_ids.extend(group.class_ids)
                class_names.extend(group.class_names)
                detection_ids.extend(group.detection_ids)
            instance_detections = _segmentation_results_to_instance_detections(
                raw_masks=raw_masks,
                confidences=confidences,
                class_ids=class_ids,
                class_names=class_names,
                detection_ids=detection_ids,
                image=single_image,
                threshold=threshold,
            )
            results.append({"predictions": instance_detections})
        return results

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        endpoint = f"{API_BASE_URL}/inferenceproxy/sam3-pvs"

        if boxes is None:
            boxes = [None] * len(images)

        results: List[dict] = []
        for single_image, boxes_for_image in zip(images, boxes):
            groups = self._build_prompt_groups(
                boxes_for_image=boxes_for_image, points=points
            )
            raw_masks, confidences, class_ids, class_names, detection_ids = (
                [],
                [],
                [],
                [],
                [],
            )
            for group in groups:
                payload = {
                    "image": {"type": "base64", "value": single_image.base64_image},
                    "prompts": Sam2PromptSet(prompts=group.prompts).dict(
                        exclude_none=True
                    ),
                    "multimask_output": multimask_output,
                    # Ask for lossless RLE; if the proxy strips this field the
                    # polygon fallback in the converter rasterises the response.
                    "format": "rle",
                }
                try:
                    headers = {"Content-Type": "application/json"}
                    if ROBOFLOW_INTERNAL_SERVICE_NAME:
                        headers["X-Roboflow-Internal-Service-Name"] = (
                            ROBOFLOW_INTERNAL_SERVICE_NAME
                        )
                    if ROBOFLOW_INTERNAL_SERVICE_SECRET:
                        headers["X-Roboflow-Internal-Service-Secret"] = (
                            ROBOFLOW_INTERNAL_SERVICE_SECRET
                        )
                    headers = build_roboflow_api_headers(explicit_headers=headers)
                    response = requests.post(
                        wrap_url(f"{endpoint}?api_key={self._api_key}"),
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    response.raise_for_status()
                    resp_json = response.json()
                except Exception as e:
                    raise Exception(f"SAM3 interactive request failed: {e}")

                for masks_payload, confidence in _parse_segmentation_predictions(
                    resp_json
                ):
                    raw_masks.append(masks_payload)
                    confidences.append(confidence)
                class_ids.extend(group.class_ids)
                class_names.extend(group.class_names)
                detection_ids.extend(group.detection_ids)
            instance_detections = _segmentation_results_to_instance_detections(
                raw_masks=raw_masks,
                confidences=confidences,
                class_ids=class_ids,
                class_names=class_names,
                detection_ids=detection_ids,
                image=single_image,
                threshold=threshold,
            )
            results.append({"predictions": instance_detections})
        return results

    @staticmethod
    def _build_prompt_groups(
        boxes_for_image,
        points: Optional[List[Any]],
    ) -> List["_PromptGroup"]:
        # The SAM prompt encoder batches a request's prompts into a single
        # tensor, so one request cannot mix box-carrying and box-less prompts -
        # box and point prompts are therefore issued as separate requests.
        groups: List[_PromptGroup] = []
        prompt_detections = _prompt_detections(boxes_for_image)
        if prompt_detections is not None:
            prompts: List[Sam2Prompt] = []
            class_ids: List[Optional[int]] = []
            class_names: List[str] = []
            detection_ids: List[Optional[str]] = []
            bboxes_metadata = prompt_detections.bboxes_metadata or [
                {} for _ in range(len(prompt_detections.xyxy))
            ]
            for index in range(int(prompt_detections.xyxy.shape[0])):
                x1, y1, x2, y2 = prompt_detections.xyxy[index].tolist()
                width = x2 - x1
                height = y2 - y1
                prompts.append(
                    Sam2Prompt(
                        box=Box(
                            x=float(x1 + width / 2),
                            y=float(y1 + height / 2),
                            width=float(width),
                            height=float(height),
                        )
                    )
                )
                class_ids.append(int(prompt_detections.class_id[index]))
                class_names.append(_resolve_prompt_class_name(prompt_detections, index))
                detection_ids.append(bboxes_metadata[index].get(DETECTION_ID_KEY))
            groups.append(
                _PromptGroup(
                    prompts=prompts,
                    class_ids=class_ids,
                    class_names=class_names,
                    detection_ids=detection_ids,
                )
            )
        if points:
            groups.append(
                _PromptGroup(
                    prompts=[Sam2Prompt(points=_as_sam2_points(points))],
                    class_ids=[0],
                    class_names=["foreground"],
                    detection_ids=[None],
                )
            )
        if boxes_for_image is None and not points:
            raise ValueError(
                "SAM 3 Interactive block requires at least one prompt - "
                "provide `points` and/or `boxes` input."
            )
        # boxes input connected but no detections for this image -> no prompts,
        # runners emit empty predictions
        return groups


@dataclass(frozen=True)
class _PromptGroup:
    prompts: List[Sam2Prompt]
    class_ids: List[Optional[int]]
    class_names: List[str]
    detection_ids: List[Optional[str]]


def _prompt_detections(boxes_for_image):
    """Normalize the tensor-native ``boxes`` prompt into a bbox-bearing prediction.

    ``boxes`` arrives as native ``Detections`` / ``InstanceDetections`` or the
    keypoint ``(KeyPoints, Optional[Detections])`` tuple; the tuple's instances
    component is required (``split_key_point_prediction`` raises when missing).
    Returns ``None`` when ``boxes`` is absent or carries zero instances.
    """
    if boxes_for_image is None:
        return None
    _key_points, detections = split_key_point_prediction(boxes_for_image)
    if int(detections.xyxy.shape[0]) == 0:
        return None
    return detections


def _resolve_prompt_class_name(prompt_detections, source_index: int) -> str:
    """Forward the prompt box's class name, mirroring the numpy block. Native OD
    producers carry names in ``image_metadata['class_names']`` keyed by ``class_id``
    (NOT a per-box field); a per-box ``class`` override (vlm/ocr prompts) wins if
    present."""
    if prompt_detections.bboxes_metadata is not None and source_index < len(
        prompt_detections.bboxes_metadata
    ):
        override = prompt_detections.bboxes_metadata[source_index].get(CLASS_NAME_KEY)
        if override is not None:
            return str(override)
    class_names = (prompt_detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    return class_names.get(int(prompt_detections.class_id[source_index]), "foreground")


def _prompt_group_to_model_inputs(
    prompts: List[Sam2Prompt],
) -> Dict[str, Optional[np.ndarray]]:
    """Convert a prompt group into the array inputs that
    ``SAM3Torch.segment_with_visual_prompts`` expects - the exact
    ``to_sam2_inputs`` + point-padding + ``np.array`` chain the legacy adapter's
    ``segment_image`` applies, so prompts hit the model identically on both
    paths (padding is an identity for the groups this block builds today, but
    keeps ragged point prompts safe)."""
    args = Sam2PromptSet(prompts=prompts).to_sam2_inputs()
    if args["point_coords"] is not None:
        max_len = max(max(len(prompt) for prompt in args["point_coords"]), 1)
        for prompt in args["point_coords"]:
            for _ in range(max_len - len(prompt)):
                prompt.append([0, 0])
        for label in args["point_labels"]:
            for _ in range(max_len - len(label)):
                label.append(-1)
    elif args["point_labels"] is not None:
        raise ValueError(
            "Can't have point labels without corresponding point coordinates"
        )
    return {
        "point_coords": (
            np.array(args["point_coords"]) if args["point_coords"] is not None else None
        ),
        "point_labels": (
            np.array(args["point_labels"]) if args["point_labels"] is not None else None
        ),
        "box": np.array(args["box"]) if args["box"] is not None else None,
    }


def _parse_segmentation_predictions(
    response: Union[dict, list],
) -> List[Tuple[Any, float]]:
    """Extract ``(masks_payload, confidence)`` pairs from a raw SAM3 PVS response.

    ``masks_payload`` is either a COCO RLE dict (``{"size": [h, w], "counts": ...}``,
    when the server honored ``format="rle"``) or a list of polygons (default
    polygon-format response) - the converter handles both.
    """
    if isinstance(response, list):
        raw_predictions = response
    else:
        raw_predictions = response.get("predictions", [])
    return [
        (
            raw_prediction.get("masks", []),
            float(raw_prediction.get("confidence", 0.0)),
        )
        for raw_prediction in raw_predictions
    ]


def _segmentation_results_to_instance_detections(
    raw_masks: List[Any],
    confidences: List[float],
    class_ids: List[Optional[int]],
    class_names: List[str],
    detection_ids: List[Optional[str]],
    image: WorkflowImageData,
    threshold: float,
) -> InstanceDetections:
    """Build a native ``InstanceDetections`` from per-prompt SAM3 PVS results.

    Mirrors ``convert_sam2_segmentation_response_to_inference_instances_seg_response``
    semantics (confidence filter, class/detection-id forwarding by prompt order)
    with the family's raw-mask instance model: an RLE mask yields ONE instance per
    prompt; a polygon-format payload is rasterised per polygon (numpy-parity
    expansion for responses where RLE was unavailable).
    """
    height, width = image._read_shape_without_materialization()
    kept_masks: List[torch.Tensor] = []
    kept_confidences: List[float] = []
    kept_class_ids: List[int] = []
    kept_class_names: List[str] = []
    kept_detection_ids: List[Optional[str]] = []

    def _keep(
        mask: torch.Tensor,
        confidence: float,
        class_id: Optional[int],
        class_name: str,
        detection_id: Optional[str],
    ) -> None:
        kept_masks.append(mask)
        kept_confidences.append(confidence)
        kept_class_ids.append(int(class_id) if class_id is not None else 0)
        kept_class_names.append(class_name)
        kept_detection_ids.append(detection_id)

    for masks_payload, confidence, class_id, class_name, detection_id in zip(
        raw_masks, confidences, class_ids, class_names, detection_ids
    ):
        if confidence < threshold:
            # skipping masks below threshold - mirrors the numpy converter
            continue
        if isinstance(masks_payload, torch.Tensor):
            # Native-bridge path: an already-binarised (H, W) mask straight from
            # SAM3Torch - one instance per prompt; single DtoH copy here (the
            # RLE carrier built below is CPU-side anyway).
            _keep(
                masks_payload.detach().to("cpu").to(torch.bool),
                confidence,
                class_id,
                class_name,
                detection_id,
            )
            continue
        if isinstance(masks_payload, dict):
            dense = _coco_rle_to_torch_mask(
                masks_payload, image_height=height, image_width=width
            )
            _keep(dense, confidence, class_id, class_name, detection_id)
            continue
        # Polygon-format payload: one instance PER polygon (>= 3 points), exactly
        # like the numpy converter's expansion.
        for polygon in masks_payload:
            if len(polygon) < 3:
                # skipping empty masks
                continue
            dense = _polygon_to_torch_mask(
                polygon, image_height=height, image_width=width
            )
            _keep(dense, confidence, class_id, class_name, detection_id)

    n = len(kept_masks)
    if n == 0:
        xyxy = torch.zeros((0, 4), dtype=torch.float32)
        class_id_tensor = torch.zeros((0,), dtype=torch.int64)
        confidence_tensor = torch.zeros((0,), dtype=torch.float32)
    else:
        xyxy = torch.stack([_mask_to_xyxy(mask) for mask in kept_masks])
        class_id_tensor = torch.tensor(kept_class_ids, dtype=torch.int64)
        confidence_tensor = torch.tensor(kept_confidences, dtype=torch.float32)

    rle_dicts = [torch_mask_to_coco_rle(mask) for mask in kept_masks]
    mask_carrier = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(height, width), masks=rle_dicts
    )

    detections = InstanceDetections(
        xyxy=xyxy.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        class_id=class_id_tensor.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        confidence=confidence_tensor.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        mask=mask_carrier,
    )
    class_names_map: Dict[int, str] = {}
    for instance_class_id, instance_class_name in zip(kept_class_ids, kept_class_names):
        class_names_map[instance_class_id] = instance_class_name
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names_map,
        prediction_type=PREDICTION_TYPE,
        inference_id=str(uuid.uuid4()),
    )
    bboxes_metadata = []
    for instance_class_name, instance_detection_id in zip(
        kept_class_names, kept_detection_ids
    ):
        entry = {
            DETECTION_ID_KEY: (
                instance_detection_id
                if instance_detection_id is not None
                else str(uuid.uuid4())
            ),
            # Per-box class override keeps numpy's per-row name fidelity even if
            # two prompts share a class_id with different labels.
            CLASS_NAME_KEY: instance_class_name,
        }
        bboxes_metadata.append(entry)
    detections.bboxes_metadata = bboxes_metadata
    return detections


def _coco_rle_to_torch_mask(
    rle_payload: dict, image_height: int, image_width: int
) -> torch.Tensor:
    counts = rle_payload["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    size = rle_payload.get("size") or [image_height, image_width]
    dense = coco_rle_masks_to_numpy_mask(
        InstancesRLEMasks(image_size=(int(size[0]), int(size[1])), masks=[counts])
    )[0]
    return torch.from_numpy(dense).to(torch.bool)


def _polygon_to_torch_mask(
    polygon: List[List[float]], image_height: int, image_width: int
) -> torch.Tensor:
    import cv2

    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    contour = np.round(np.array(polygon, dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(canvas, [contour], 1)
    return torch.from_numpy(canvas.astype(bool))


def _mask_to_xyxy(mask: torch.Tensor) -> torch.Tensor:
    """Tight xyxy bbox of a 2-D boolean mask (inclusive min/max, NO +1) - the
    convention shared by the SAM tensor family, keeping boxes numpy-faithful."""
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        return torch.zeros((4,), dtype=torch.float32, device=mask.device)
    ys, xs = nonzero[:, 0], nonzero[:, 1]
    return torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).to(torch.float32)
