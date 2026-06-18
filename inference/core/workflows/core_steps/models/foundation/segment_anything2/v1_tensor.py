import uuid
from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from inference.core.env import (
    CORE_MODEL_SAM2_ENABLED,
    GCP_SERVERLESS,
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
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
    STRING_KIND,
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

# SAM2 mask-binarisation threshold in logit space, mirroring the numpy adapter's
# MASK_THRESHOLD (inference/models/sam2/segment_anything2_inference_models.py): a pixel
# is foreground when its mask logit is >= 0.0 (equivalently sigmoid(logit) >= 0.5).
MASK_THRESHOLD = 0.0

LONG_DESCRIPTION = """
Run Segment Anything 2, a zero-shot instance segmentation model, on an image.

** Dedicated inference server required (GPU recomended) **

You can use pass in boxes/predictions from other models to Segment Anything 2 to use as prompts for the model.
If you pass in box detections from another model, the class names of the boxes will be forwarded to the predicted masks.  If using the model unprompted, the model will assign integers as class names / ids.
"""


PREDICTION_TYPE = "instance-segmentation"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Segment Anything 2 Model",
            "version": "v1",
            "short_description": "Convert bounding boxes to polygons, or run SAM2 on an entire image to generate a mask.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["SAM2", "META"],
            "ui_manifest": {
                "section": "model",
                "icon": "fa-brands fa-meta",
                "blockPriority": 9.5,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/segment_anything@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    boxes: Optional[
        Selector(
            kind=[
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        description="Bounding boxes (from another model) to convert to polygons",
        examples=["$steps.object_detection_model.predictions"],
        default=None,
        json_schema_extra={"always_visible": True},
    )
    version: Union[
        Selector(kind=[STRING_KIND]),
        Literal["hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus"],
    ] = Field(
        default="hiera_tiny",
        description="Model to be used.  One of hiera_large, hiera_small, hiera_tiny, hiera_b_plus",
        examples=["hiera_large", "$inputs.openai_model"],
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.0, description="Threshold for predicted masks scores", examples=[0.3]
    )
    multimask_output: Union[Optional[bool], Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Flag to determine whether to use sam2 internal multimask or single mask mode. For ambiguous prompts setting to True is recomended.",
        examples=[True, "$inputs.multimask_output"],
    )
    mask_representation: Literal["rle", "dense"] = Field(
        default="rle",
        description="Carrier for instance masks. RLE (compact) by default; forced to "
        "'rle' on GCP_SERVERLESS regardless of this value.",
    )
    collapse_to_most_confident: bool = Field(
        default=True,
        description="Collapse SAM2 multi-mask proposals to the single most-confident "
        "mask per prompt.",
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
        restrictions = [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; run_locally() loads a model that needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ]
        if not CORE_MODEL_SAM2_ENABLED:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.HARD,
                    note=(
                        "CORE_MODEL_SAM2_ENABLED=False on Roboflow Hosted "
                        "Serverless: the SAM2 endpoint is not registered, so "
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
        return [
            "sam2/hiera_large",
            "sam2/hiera_small",
            "sam2/hiera_tiny",
            "sam2/hiera_b_plus",
        ]


class SegmentAnything2BlockV1(WorkflowBlock):

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
    def get_manifest(cls):
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch],
        version: str,
        threshold: float,
        multimask_output: bool,
        mask_representation: Literal["rle", "dense"],
        collapse_to_most_confident: bool,
    ) -> BlockResult:
        # GCP_SERVERLESS forces RLE regardless of the requested knob.
        if GCP_SERVERLESS:
            mask_representation = "rle"
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                boxes=boxes,
                version=version,
                threshold=threshold,
                multimask_output=multimask_output,
                mask_representation=mask_representation,
                collapse_to_most_confident=collapse_to_most_confident,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                boxes=boxes,
                version=version,
                threshold=threshold,
                multimask_output=multimask_output,
                mask_representation=mask_representation,
                collapse_to_most_confident=collapse_to_most_confident,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch],
        version: str,
        threshold: float,
        multimask_output: bool,
        mask_representation: Literal["rle", "dense"],
        collapse_to_most_confident: bool,
    ) -> BlockResult:
        sam_model_id = f"sam2/{version}"
        self._model_manager.add_model(
            sam_model_id, self._api_key, endpoint_type=ModelEndpointType.CORE_MODEL
        )

        boxes_iter = boxes if boxes is not None else [None] * len(images)
        results: List[dict] = []
        for image, boxes_for_image in zip(images, boxes_iter):
            prompt_detections = _prompt_detections(
                boxes_for_image
            )  # raises if KP-no-bbox
            box_tensor = (
                prompt_detections.xyxy if prompt_detections is not None else None
            )
            sam2_predictions = self._model_manager.run_tensor_native_inference(
                sam_model_id,
                action="segment",
                images=[image.tensor_image],
                boxes=[box_tensor] if box_tensor is not None else None,
                multi_mask_output=multimask_output,
                input_color_format="rgb",
                # Return raw mask logits and binarise explicitly below (legacy style),
                # rather than relying on segment_images' internal default threshold.
                return_logits=True,
            )
            instance_detections = _sam2_prediction_to_instance_detections(
                sam2_prediction=sam2_predictions[0],
                image=image,
                prompt_detections=prompt_detections,
                threshold=threshold,
                collapse=collapse_to_most_confident,
                mask_representation=mask_representation,
            )
            results.append({"predictions": instance_detections})
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch],
        version: str,
        threshold: float,
        multimask_output: bool,
        mask_representation: Literal["rle", "dense"],
        collapse_to_most_confident: bool,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(api_url=api_url, api_key=self._api_key)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()

        # The SAM2 server always collapses to the most-confident mask per prompt
        # (segment_image -> choose_most_confident_sam_prediction), so collapse=False
        # cannot be honored remotely.
        if not collapse_to_most_confident:
            raise NotImplementedError(
                "collapse_to_most_confident=False is not supported on the remote SAM2 "
                "path (the server always returns the most-confident mask)."
            )
        boxes_iter = boxes if boxes is not None else [None] * len(images)
        results: List[dict] = []
        for image, boxes_for_image in zip(images, boxes_iter):
            prompt_detections = _prompt_detections(boxes_for_image)
            prompts = _box_prompts_payload(prompt_detections)
            # C8: request RLE for compact transfer (mask_input_format="rle", NOT
            # response_mask_format). threshold (score filter) and mask_representation
            # (rle/dense output) are applied in the converter.
            response = client.sam2_segment_image(
                inference_input=image.base64_image,
                sam2_version_id=version,
                prompts=prompts,
                multimask_output=multimask_output,
                mask_input_format="rle",
            )
            instance_detections = _rle_response_to_instance_detections(
                response=response,
                image=image,
                prompt_detections=prompt_detections,
                threshold=threshold,
                mask_representation=mask_representation,
            )
            results.append({"predictions": instance_detections})
        return results


def _prompt_detections(boxes_for_image):
    """Normalize the tensor-native ``boxes`` prompt into a bbox-bearing Detections.

    Per the prediction kinds, ``boxes`` arrives as one of:
      * ``Detections``                                   (object detection)
      * ``InstanceDetections``                           (instance segmentation)
      * ``Tuple[KeyPoints, Optional[Detections]]``       (keypoint detection — see
        TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND.internal_data_type)

    The keypoint kind is a TUPLE whose instances (``Detections``) component may be
    missing (``None``). SAM2 can only be box-prompted, so missing instances is a
    hard runtime error — raised by ``split_key_point_prediction`` ("Keypoint
    prediction is missing the bounding-box component required by this block.").

    Returns ``None`` only when ``boxes`` is entirely absent (unprompted whole-image
    segmentation) or the prediction carries zero instances.
    """
    if boxes_for_image is None:
        return None
    # Raises on a keypoint tuple whose Detections component is None (instances
    # missing); returns the prediction as-is for OD / IS.
    _key_points, detections = split_key_point_prediction(boxes_for_image)
    if len(detections) == 0:
        return None
    return detections


def _choose_most_confident_torch(
    masks: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-native equivalent of choose_most_confident_sam_prediction: pick the
    highest-scoring proposed mask per prompt, on-device, no numpy bridge.

    masks: (prompt_set, proposed, H, W) or (proposed, H, W) — SAM2 squeezes the
    prompt_set dimension when prompt_set == 1, so we re-expand it.
    Returns (selected_masks (prompt_set, H, W), selected_scores (prompt_set,)).
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)
        scores = scores.unsqueeze(0)
    prompt_set, _proposed, height, width = masks.shape
    best = torch.argmax(scores, dim=1)  # (prompt_set,)
    gather_index = best.view(prompt_set, 1, 1, 1).expand(prompt_set, 1, height, width)
    selected_masks = torch.gather(masks, 1, gather_index).squeeze(1)
    selected_scores = torch.gather(scores, 1, best.unsqueeze(1)).squeeze(1)
    return selected_masks, selected_scores


def _sam2_prediction_to_instance_detections(
    sam2_prediction,
    *,
    image: WorkflowImageData,
    prompt_detections,
    threshold: float,
    collapse: bool,
    mask_representation: str,
) -> InstanceDetections:
    if not collapse:
        # Non-collapsed output (multiple proposals per prompt) has no defined
        # instance semantics yet — needs a product decision before wiring.
        raise NotImplementedError(
            "collapse_to_most_confident=False is not yet defined for tensor SAM2."
        )
    selected_masks, selected_scores = _choose_most_confident_torch(
        sam2_prediction.masks, sam2_prediction.scores
    )  # (m, H, W) mask logits, (m,)
    # Binarise the mask logits at the SAM2 threshold, mirroring the numpy adapter's
    # `masks >= MASK_THRESHOLD` (MASK_THRESHOLD == 0.0).
    binary = (selected_masks >= MASK_THRESHOLD).to(torch.bool)
    source_indices, keep_index = _score_filter(selected_scores, threshold)
    return _assemble_instance_detections(
        binary=binary[keep_index],
        confidence=selected_scores[keep_index],
        image=image,
        prompt_detections=prompt_detections,
        source_indices=source_indices,
        mask_representation=mask_representation,
    )


def _score_filter(
    scores: torch.Tensor, threshold: float
) -> Tuple[List[int], torch.Tensor]:
    """Confidence filter — mirrors the numpy block's `if confidence < threshold:
    continue`. Returns (surviving source indices, index tensor for tensor selection)."""
    source_indices = [
        i for i, keep in enumerate((scores >= threshold).tolist()) if keep
    ]
    return source_indices, torch.tensor(source_indices, dtype=torch.long)


def _assemble_instance_detections(
    *,
    binary: torch.Tensor,  # (n, H, W) bool, already score-filtered
    confidence: torch.Tensor,  # (n,)
    image: WorkflowImageData,
    prompt_detections,
    source_indices: List[int],
    mask_representation: str,
) -> InstanceDetections:
    n = len(source_indices)
    height, width = image._read_shape_without_materialization()
    if n == 0:
        xyxy = torch.zeros((0, 4), dtype=torch.float32)
        class_id = torch.zeros((0,), dtype=torch.int64)
    else:
        xyxy = torch.stack([_mask_to_xyxy(binary[i]) for i in range(n)])
        class_id = _prompt_class_ids(prompt_detections, source_indices)

    if mask_representation == "rle":
        rle_dicts = [torch_mask_to_coco_rle(binary[i]) for i in range(n)]
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(height, width), masks=rle_dicts
        )
    else:
        mask = binary

    detections = InstanceDetections(
        xyxy=xyxy.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        class_id=class_id.to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        confidence=confidence.to(torch.float32)
        .reshape(-1)
        .to(WORKFLOWS_IMAGE_TENSOR_DEVICE),
        mask=(
            mask
            if isinstance(mask, InstancesRLEMasks)
            else mask.to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
        ),
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=_class_names_map(prompt_detections, source_indices),
        prediction_type=PREDICTION_TYPE,
        inference_id=str(uuid.uuid4()),
    )
    detections.bboxes_metadata = [
        _per_instance_metadata(prompt_detections, src) for src in source_indices
    ]
    return detections


def _mask_to_xyxy(mask: torch.Tensor) -> torch.Tensor:
    """Tight xyxy bbox of a 2-D boolean mask, on the mask's device.

    Uses inclusive min/max with NO +1, matching the numpy SAM2 sibling's
    polygon-derived bbox (``np.min``/``np.max`` on the contour points) and the
    other SAM tensor siblings (``segment_anything2_video``, ``segment_anything3``,
    ``seg_preview``), so masks/areas/IoU stay numpy-faithful across the family.
    """
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        return torch.zeros((4,), dtype=torch.float32, device=mask.device)
    ys, xs = nonzero[:, 0], nonzero[:, 1]
    return torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).to(torch.float32)


def _prompted(prompt_detections) -> bool:
    return prompt_detections is not None and len(prompt_detections) > 0


def _prompt_class_ids(prompt_detections, source_indices: List[int]) -> torch.Tensor:
    # Forward each surviving instance's prompt class_id; unprompted -> 0 (the numpy
    # block uses class_id 0 / "foreground" when there is no prompt).
    if _prompted(prompt_detections):
        ids = [int(prompt_detections.class_id[src]) for src in source_indices]
        return torch.tensor(ids, dtype=torch.int64)
    return torch.zeros((len(source_indices),), dtype=torch.int64)


def _class_names_map(prompt_detections, source_indices: List[int]) -> Dict[int, str]:
    # class_id -> name map carried on image_metadata (used by the serializer).
    names: Dict[int, str] = {}
    for src in source_indices:
        class_id = (
            int(prompt_detections.class_id[src]) if _prompted(prompt_detections) else 0
        )
        names[class_id] = _resolve_prompt_class_name(prompt_detections, src)
    return names


def _per_instance_metadata(prompt_detections, source_index: int) -> dict:
    entry = {DETECTION_ID_KEY: str(uuid.uuid4())}
    if (
        prompt_detections is not None
        and prompt_detections.bboxes_metadata is not None
        and source_index < len(prompt_detections.bboxes_metadata)
    ):
        src = prompt_detections.bboxes_metadata[source_index]
        if DETECTION_ID_KEY in src:
            entry[DETECTION_ID_KEY] = src[DETECTION_ID_KEY]
        # Forward a per-box class override only if the prompt explicitly carried one
        # (vlm/ocr prompts); standard OD prompts resolve the name from the
        # class_id -> name map on image_metadata, so no spurious per-box `class`.
        if CLASS_NAME_KEY in src:
            entry[CLASS_NAME_KEY] = src[CLASS_NAME_KEY]
    return entry


def _resolve_prompt_class_name(prompt_detections, source_index: int) -> str:
    """Forward the prompt box's class name, mirroring the numpy block. Native OD
    producers carry names in ``image_metadata['class_names']`` keyed by ``class_id``
    (NOT a per-box field); a per-box ``class`` override (vlm/ocr prompts) wins if
    present. Unprompted SAM (no boxes) -> ``foreground``."""
    if not _prompted(prompt_detections):
        return "foreground"
    if prompt_detections.bboxes_metadata is not None and source_index < len(
        prompt_detections.bboxes_metadata
    ):
        override = prompt_detections.bboxes_metadata[source_index].get(CLASS_NAME_KEY)
        if override is not None:
            return str(override)
    class_names = (prompt_detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    return class_names.get(int(prompt_detections.class_id[source_index]), "foreground")


def _box_prompts_payload(prompt_detections) -> Optional[List[dict]]:
    if prompt_detections is None:
        return None
    prompts = []
    for i in range(len(prompt_detections)):
        x1, y1, x2, y2 = prompt_detections.xyxy[i].tolist()
        w, h = x2 - x1, y2 - y1
        prompts.append(
            {"box": {"x": x1 + w / 2, "y": y1 + h / 2, "width": w, "height": h}}
        )
    return prompts


def _rle_response_to_instance_detections(
    response,
    image: WorkflowImageData,
    prompt_detections,
    *,
    threshold: float,
    mask_representation: str,
) -> InstanceDetections:
    """Build InstanceDetections from the SAM2 server's rle response. Schema (from
    turn_segmentation_results_into_rle_response -> Sam2SegmentationPrediction):
        {"predictions": [{"masks": {"size": [h, w], "counts": "<utf8>"},
                          "confidence": float, "format": "rle"}, ...]}
    The response carries no bbox, so xyxy is derived from the decoded mask; classes
    are forwarded from the prompt by index (response preserves prompt order). The
    response is decoded to dense, score-filtered, then re-assembled per
    mask_representation — uniform with the local path.
    """
    if isinstance(response, list):
        response = response[0]
    predictions = response.get("predictions", []) or []
    # Class identity is forwarded from the prompt by RESPONSE ROW POSITION
    # (source_indices index into prompt_detections after the score filter). That is
    # only sound if the SAM2 server returns exactly one prediction per prompt in
    # prompt order. Guard the positional contract before forwarding classes: if the
    # server ever drops/reorders a prompt's prediction the indices would silently
    # pull the wrong class_id/detection_id for every subsequent instance.
    if _prompted(prompt_detections) and len(predictions) != len(prompt_detections):
        raise ValueError(
            "SAM2 remote response returned "
            f"{len(predictions)} prediction(s) for {len(prompt_detections)} box "
            "prompt(s); positional class forwarding requires one prediction per "
            "prompt in prompt order."
        )
    height, width = image._read_shape_without_materialization()

    counts: List[bytes] = []
    scores: List[float] = []
    for prediction in predictions:
        raw_counts = prediction["masks"]["counts"]  # {"size":[h,w],"counts":"<utf8>"}
        # Normalize to bytes (local frPyObjects path stores counts as bytes; the remote
        # rle response decodes them to a utf-8 string).
        counts.append(
            raw_counts.encode("utf-8") if isinstance(raw_counts, str) else raw_counts
        )
        scores.append(float(prediction["confidence"]))

    if not counts:
        binary = torch.zeros((0, height, width), dtype=torch.bool)
    else:
        dense = coco_rle_masks_to_numpy_mask(
            InstancesRLEMasks(image_size=(height, width), masks=counts)
        )
        binary = torch.from_numpy(dense).to(torch.bool)

    selected_scores = torch.tensor(scores, dtype=torch.float32)
    source_indices, keep_index = _score_filter(selected_scores, threshold)
    return _assemble_instance_detections(
        binary=binary[keep_index],
        confidence=selected_scores[keep_index],
        image=image,
        prompt_detections=prompt_detections,
        source_indices=source_indices,
        mask_representation=mask_representation,
    )
