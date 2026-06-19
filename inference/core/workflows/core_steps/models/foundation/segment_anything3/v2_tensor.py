"""Tensor-native sibling of `roboflow_core/sam3@v2`.

Builds on the v1_tensor SAM3 conversion, adding per-class confidence thresholds +
cross-prompt NMS. Because `run_tensor_native_inference` is a THIN forward to
`segment_with_text_prompts` (it applies only a single global threshold floor),
this block must replicate the adapter's per-class-threshold + cross-prompt-NMS
orchestration itself. Rather than reach across the package boundary into the
private `inference.models.sam3` adapter internals (the previous, fragile design),
this block keeps LOCAL COPIES of those helpers (see the TODO(sam3-public-adapter)
note below); they run the per-class threshold + cross-prompt NMS on the raw
native masks, then build one `inference_models.InstanceDetections` per image
(RLE-default).

When NMS runs against the RLE carrier, the COCO RLEs that NMS already encodes for
IoU are reused to build the output `InstancesRLEMasks` directly, instead of
re-encoding the kept dense masks a second time in `_build_instance_detections`.

Output is tensor-native. REMOTE/proxy paths forward per-class thresholds +
nms_iou_threshold to the server (which applies them, matching the numpy block) and
rasterise the returned polygon response via the v1_tensor polygon path.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import requests
from pycocotools import mask as mask_utils
from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.entities.requests.sam3 import Sam3Prompt
from inference.core.env import (
    API_BASE_URL,
    CORE_MODEL_SAM3_ENABLED,
    GCP_SERVERLESS,
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

# Reuse the v1_tensor SAM3 conversion machinery verbatim.
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1_tensor import (
    Item,
    _assemble_detections,
    _build_instance_detections,
    _build_instance_detections_from_polygons,
    _normalize_class_names,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    DETECTION_ID_KEY,
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
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
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
from inference_sdk import InferenceHTTPClient

# TODO(sam3-public-adapter): these per-class-threshold + cross-prompt-NMS helpers
# are LOCAL COPIES of the private `inference.models.sam3.
# segment_anything3_inference_models` internals
# (`_to_numpy_masks`, `_collect_masks_with_per_prompt_threshold`,
# `_apply_nms_cross_prompt`, `_nms_greedy_pycocotools`). They were previously
# imported across the package boundary from those underscore-prefixed adapter
# internals, which carry no API-stability guarantee and forced this workflow
# block to track the adapter by hand. We copy them here (decoupling the block
# from adapter refactors) until inference_models exposes a PUBLIC,
# tested entrypoint that performs per-class threshold + cross-prompt NMS
# adapter-side (e.g. a `run_tensor_native_inference` mode that returns
# post-NMS, post-per-class-threshold native masks). When that lands, delete
# these copies and consume the public result directly. Keep this logic in
# byte-for-byte parity with the adapter so numpy/tensor behaviour stays defined
# in one place until the public API exists.

LONG_DESCRIPTION = """
Run Segment Anything 3 (zero-shot, text-prompted) with per-class confidence
thresholds and optional cross-prompt Non-Maximum Suppression.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3",
            "version": "v2",
            "short_description": "Sam3",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Sam3"],
            "ui_manifest": {
                "section": "model",
                "icon": "fa-solid fa-eye",
                "blockPriority": 9.49,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/sam3@v2"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), Optional[str]] = Field(
        default="sam3/sam3_final",
        description="model version.  You only need to change this for fine tuned sam3 models.",
        examples=["sam3/sam3_final", "$inputs.model_variant"],
    )
    class_names: Optional[
        Union[List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])]
    ] = Field(
        title="Class Names",
        default=None,
        description="List of classes to recognise",
        examples=[["car", "person"], "$inputs.classes"],
    )
    confidence: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.5,
        title="Confidence Threshold",
        description="Minimum confidence threshold for predicted masks",
        examples=[0.3],
    )
    per_class_confidence: Optional[
        Union[List[float], Selector(kind=[LIST_OF_VALUES_KIND])]
    ] = Field(
        default=None,
        title="Per-Class Confidence",
        description="List of confidence thresholds per class (must match class_names length)",
        examples=[[0.3, 0.5, 0.7]],
    )
    apply_nms: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        title="Apply NMS",
        description="Whether to apply Non-Maximum Suppression across prompts",
    )
    nms_iou_threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.9,
        title="NMS IoU Threshold",
        description="IoU threshold for cross-prompt NMS. Must be in [0.0, 1.0]",
        examples=[0.5, 0.9],
    )
    mask_representation: Literal["rle", "dense"] = Field(
        default="rle",
        description="Carrier for instance masks. RLE (compact) by default; forced to "
        "'rle' on GCP_SERVERLESS regardless of this value.",
    )

    @field_validator("nms_iou_threshold")
    @classmethod
    def _validate_nms_iou_threshold(cls, v):
        if isinstance(v, (int, float)) and (v < 0.0 or v > 1.0):
            raise ValueError("nms_iou_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _validate_per_class_confidence_length(self) -> "BlockManifest":
        if not isinstance(self.per_class_confidence, list):
            return self
        if isinstance(self.class_names, list):
            class_names_length = len(self.class_names)
        elif isinstance(self.class_names, str):
            class_names_length = len(self.class_names.split(","))
        else:
            return self
        if len(self.per_class_confidence) != class_names_length:
            raise ValueError(
                f"per_class_confidence length ({len(self.per_class_confidence)}) "
                f"must match class_names length ({class_names_length})"
            )
        return self

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
        return ["sam3/sam3_final"]


class SegmentAnything3BlockV2(WorkflowBlock):

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
        model_id: str,
        class_names: Optional[Union[List[str], str]],
        confidence: float,
        per_class_confidence: Optional[List[float]],
        apply_nms: bool,
        nms_iou_threshold: float,
        mask_representation: Literal["rle", "dense"],
    ) -> BlockResult:
        if GCP_SERVERLESS:
            mask_representation = "rle"
        class_names = _normalize_class_names(class_names)
        if SAM3_EXEC_MODE == "remote":
            return self.run_via_request(
                images=images,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
                mask_representation=mask_representation,
            )
        elif self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
                mask_representation=mask_representation,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
                mask_representation=mask_representation,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_names: List[Optional[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]],
        apply_nms: bool,
        nms_iou_threshold: float,
        mask_representation: str,
    ) -> BlockResult:
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)

        # Real Sam3Prompt objects carry .output_prob_thresh for the per-class helper.
        sam3_prompts = [
            Sam3Prompt(
                type="text",
                text=class_name,
                output_prob_thresh=_per_class_threshold(per_class_confidence, idx),
            )
            for idx, class_name in enumerate(class_names)
        ]
        # The model forward takes plain {"text": cn} dicts.
        native_prompts = [{"text": cn} for cn in class_names]
        # MIN-floor so the backend floor never pre-clips a mask a LOWER per-class
        # threshold would keep (mirrors adapter segment_image).
        floor = _min_floor(confidence, per_class_confidence)

        results: List[dict] = []
        for single_image in images:
            # SAM3 converts its input to numpy internally and is colour-agnostic
            # (expects RGB, like the materialised tensor). Pass the tensor when it is
            # already on device, otherwise an RGB host frame — avoiding a forced CHW
            # transpose + H2D that SAM3 would immediately undo.
            if single_image.is_tensor_materialised():
                model_image = single_image.tensor_image
            else:
                model_image = np.ascontiguousarray(single_image.numpy_image[:, :, ::-1])
            per_image = self._model_manager.run_tensor_native_inference(
                model_id,
                images=[model_image],
                prompts=native_prompts,
                output_prob_thresh=float(floor),
            )
            # When NMS runs against the RLE carrier, _collect_and_build reuses the
            # COCO RLEs NMS already encoded for IoU (no second encode); otherwise it
            # falls back to the dense packer (_build_instance_detections).
            results.append(
                {
                    "predictions": _collect_and_build(
                        per_prompt_results=per_image[0],
                        class_names=class_names,
                        prompts=sam3_prompts,
                        global_confidence=confidence,
                        apply_nms=apply_nms,
                        nms_iou_threshold=nms_iou_threshold,
                        image=single_image,
                        mask_representation=mask_representation,
                    )
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_names: List[Optional[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]],
        apply_nms: bool,
        nms_iou_threshold: float,
        mask_representation: str,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(api_url=api_url, api_key=self._api_key)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        http_prompts = _build_http_prompts(class_names, per_class_confidence)

        results: List[dict] = []
        for single_image in images:
            resp_json = client.sam3_concept_segment(
                inference_input=single_image.base64_image,
                prompts=http_prompts,
                model_id=model_id,
                output_prob_thresh=confidence,
                nms_iou_threshold=nms_iou_threshold if apply_nms else None,
            )
            results.append(
                self._build_from_polygon_response(
                    resp_json=resp_json,
                    image=single_image,
                    class_names=class_names,
                    confidence=confidence,
                    mask_representation=mask_representation,
                )
            )
        return results

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[Optional[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]],
        apply_nms: bool,
        nms_iou_threshold: float,
        mask_representation: str,
    ) -> BlockResult:
        endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"
        http_prompts = _build_http_prompts(class_names, per_class_confidence)

        results: List[dict] = []
        for single_image in images:
            payload = {
                "image": {"type": "base64", "value": single_image.base64_image},
                "prompts": http_prompts,
                "output_prob_thresh": confidence,
                "nms_iou_threshold": nms_iou_threshold if apply_nms else None,
            }
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
            try:
                response = requests.post(
                    wrap_url(f"{endpoint}?api_key={self._api_key}"),
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                response.raise_for_status()
                resp_json = response.json()
            except Exception as exc:
                raise Exception(f"SAM3 request failed: {exc}")
            results.append(
                self._build_from_polygon_response(
                    resp_json=resp_json,
                    image=single_image,
                    class_names=class_names,
                    confidence=confidence,
                    mask_representation=mask_representation,
                )
            )
        return results

    def _build_from_polygon_response(
        self,
        resp_json: dict,
        image: WorkflowImageData,
        class_names: List[Optional[str]],
        confidence: float,
        mask_representation: str,
    ) -> dict:
        return {
            "predictions": _build_instance_detections_from_polygons(
                prompt_results=resp_json.get("prompt_results", []),
                class_names=class_names,
                image=image,
                threshold=confidence,
                mask_representation=mask_representation,
            )
        }


def _build_http_prompts(
    class_names: List[Optional[str]],
    per_class_confidence: Optional[List[float]],
) -> List[dict]:
    """Text prompt dicts for the REMOTE/proxy HTTP API, carrying per-class
    output_prob_thresh when provided (matches numpy v2/v3). Unguarded index is
    intentional parity — the manifest validator enforces length for concrete lists."""
    prompts: List[dict] = []
    for idx, class_name in enumerate(class_names):
        prompt: dict = {"type": "text", "text": class_name}
        if per_class_confidence is not None:
            prompt["output_prob_thresh"] = per_class_confidence[idx]
        prompts.append(prompt)
    return prompts


def _per_class_threshold(
    per_class_confidence: Optional[List[float]], idx: int
) -> Optional[float]:
    if per_class_confidence and idx < len(per_class_confidence):
        return per_class_confidence[idx]
    return None


def _min_floor(confidence: float, per_class_confidence: Optional[List[float]]) -> float:
    if per_class_confidence:
        return min([confidence, *[t for t in per_class_confidence if t is not None]])
    return confidence


# --------------------------------------------------------------------------- #
# LOCAL COPIES of the SAM3 adapter per-class-threshold + cross-prompt-NMS
# helpers. See TODO(sam3-public-adapter) at the top of this module. Keep these
# in byte-for-byte parity with
# inference.models.sam3.segment_anything3_inference_models until a public
# adapter entrypoint exists.
# --------------------------------------------------------------------------- #
def _to_numpy_masks(masks_any) -> np.ndarray:
    if masks_any is None:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    if hasattr(masks_any, "detach"):
        masks_np = masks_any.detach().cpu().numpy().astype(np.uint8)
    else:
        arrs = []
        for m in masks_any:
            if hasattr(m, "detach"):
                arrs.append(m.detach().cpu().numpy().astype(np.uint8))
            else:
                arrs.append(np.asarray(m, dtype=np.uint8))
        if not arrs:
            return np.zeros((0, 0, 0), dtype=np.uint8)
        masks_np = np.stack(arrs, axis=0)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, ...]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    return masks_np


def _collect_masks_with_per_prompt_threshold(
    processed: Dict[int, Dict[str, Any]],
    prompts: List[Sam3Prompt],
    default_threshold: float,
) -> List[Tuple[int, np.ndarray, float]]:
    all_masks: List[Tuple[int, np.ndarray, float]] = []
    for idx, p in enumerate(prompts):
        prompt_thresh = getattr(p, "output_prob_thresh", None)
        if prompt_thresh is None:
            prompt_thresh = default_threshold
        masks_np = _to_numpy_masks(processed[idx]["masks"])
        scores = processed[idx]["scores"]
        if masks_np.ndim != 3 or 0 in masks_np.shape:
            continue
        for mask, score in zip(masks_np, scores):
            if score >= prompt_thresh:
                all_masks.append((idx, mask, float(score)))
    return all_masks


def _nms_greedy_pycocotools(
    rles: List[dict],
    confidences: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    num_detections = len(rles)
    if num_detections == 0:
        return np.array([], dtype=bool)
    sort_index = np.argsort(confidences)[::-1]
    sorted_rles = [rles[i] for i in sort_index]
    ious = mask_utils.iou(sorted_rles, sorted_rles, [0] * num_detections)
    keep = np.ones(num_detections, dtype=bool)
    for i in range(num_detections):
        if keep[i]:
            condition = ious[i, :] > iou_threshold
            keep[i + 1 :] = np.where(condition[i + 1 :], False, keep[i + 1 :])
    return keep[np.argsort(sort_index)]


def _encode_mask_to_rle(mask_np: np.ndarray) -> dict:
    """COCO-RLE-encode a single (H,W) mask, returning the pycocotools dict
    ({"size": [...], "counts": <bytes>}) — the exact shape
    InstancesRLEMasks.from_coco_rle_masks consumes."""
    mb = (mask_np > 0).astype(np.uint8)
    return mask_utils.encode(np.asfortranarray(mb))


def _apply_nms_cross_prompt_with_rles(
    all_masks: List[Tuple[int, np.ndarray, float]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, np.ndarray, float]], List[dict]]:
    """Cross-prompt greedy NMS (local copy of the adapter's
    `_apply_nms_cross_prompt`) that ALSO returns the per-kept-mask COCO RLEs it
    already computed for IoU. Reusing these downstream avoids encoding the kept
    masks to RLE a second time when building the RLE output carrier.
    """
    if not all_masks:
        return all_masks, []
    rles = [_encode_mask_to_rle(mask_np) for _, mask_np, _ in all_masks]
    confidences = np.array([score for _, _, score in all_masks])
    keep = _nms_greedy_pycocotools(rles, confidences, iou_threshold)
    kept_masks = [all_masks[i] for i in range(len(all_masks)) if keep[i]]
    kept_rles = [rles[i] for i in range(len(all_masks)) if keep[i]]
    return kept_masks, kept_rles


def _threshold_and_nms(
    per_prompt_results: List[dict],
    prompts: List[Sam3Prompt],
    global_confidence: float,
    apply_nms: bool,
    nms_iou_threshold: float,
) -> Tuple[List[Tuple[int, np.ndarray, float]], Optional[List[dict]]]:
    """Run per-class threshold + (optional) cross-prompt NMS on the raw native
    masks. Returns the kept (prompt_idx, mask(H,W), score) rows and, when NMS
    actually ran, the matching COCO RLEs (else None).

    The adapter helpers key `processed` by enumerate(prompts) order, so we key the
    native results by prompt_index (pre-seeding all idx) to keep them aligned, and
    pass raw (N,1,H,W) masks straight through — `_to_numpy_masks` squeezes/casts.
    """
    processed: Dict[int, Dict[str, Any]] = {
        idx: {"masks": None, "scores": []} for idx in range(len(prompts))
    }
    for result in per_prompt_results:
        idx = result.get("prompt_index", 0)
        if idx not in processed:
            continue
        processed[idx] = {
            "masks": result.get("masks"),
            "scores": list(result.get("scores", [])),
        }

    all_masks = _collect_masks_with_per_prompt_threshold(
        processed=processed,
        prompts=prompts,
        default_threshold=global_confidence,
    )
    kept_rles: Optional[List[dict]] = None
    if apply_nms and nms_iou_threshold is not None and len(all_masks) > 0:
        all_masks, kept_rles = _apply_nms_cross_prompt_with_rles(
            all_masks, nms_iou_threshold
        )
    return all_masks, kept_rles


def _collect_from_native_with_nms(
    per_prompt_results: List[dict],
    class_names: List[Optional[str]],
    prompts: List[Sam3Prompt],
    global_confidence: float,
    apply_nms: bool,
    nms_iou_threshold: float,
) -> List[Item]:
    """Per-class-threshold + cross-prompt NMS on the raw native masks, flattened to
    Items for `_build_instance_detections` (dense-mask packer). Used by v3_tensor
    and by the v2 `dense` carrier path.
    """
    all_masks, _ = _threshold_and_nms(
        per_prompt_results=per_prompt_results,
        prompts=prompts,
        global_confidence=global_confidence,
        apply_nms=apply_nms,
        nms_iou_threshold=nms_iou_threshold,
    )
    items: List[Item] = []
    for prompt_idx, mask, score in all_masks:
        class_name = class_names[prompt_idx] if prompt_idx < len(class_names) else None
        items.append(
            (mask.astype(bool), float(score), prompt_idx, class_name or "foreground")
        )
    return items


def _build_instance_detections_reusing_nms_rles(
    all_masks: List[Tuple[int, np.ndarray, float]],
    kept_rles: List[dict],
    class_names: List[Optional[str]],
    image: WorkflowImageData,
) -> InstanceDetections:
    """RLE-carrier assembler that REUSES the COCO RLEs NMS already encoded, so the
    kept masks are not re-encoded a second time. Bbox is derived from each kept
    dense mask via np.where (cheap, no encode). Rows stay in lockstep because
    `all_masks` and `kept_rles` are produced together by
    `_apply_nms_cross_prompt_with_rles`.
    """
    height, width = image._read_shape_without_materialization()
    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    class_names_map: Dict[int, str] = {}
    bboxes_metadata: List[dict] = []
    rle_dicts: List[dict] = []

    for (prompt_idx, mask_np, score), rle in zip(all_masks, kept_rles):
        ys, xs = np.where(mask_np > 0)
        if xs.size == 0:
            continue
        class_name = (
            class_names[prompt_idx] if prompt_idx < len(class_names) else None
        ) or "foreground"
        xyxy.append(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        )
        confidences.append(float(score))
        class_ids.append(prompt_idx)
        class_names_map[prompt_idx] = class_name
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(uuid.uuid4()), CLASS_NAME_KEY: class_name}
        )
        rle_dicts.append(rle)

    mask = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(height, width), masks=rle_dicts
    )
    return _assemble_detections(
        image=image,
        xyxy=xyxy,
        confidences=confidences,
        class_ids=class_ids,
        class_names_map=class_names_map,
        bboxes_metadata=bboxes_metadata,
        mask=mask,
    )


def _collect_and_build(
    per_prompt_results: List[dict],
    class_names: List[Optional[str]],
    prompts: List[Sam3Prompt],
    global_confidence: float,
    apply_nms: bool,
    nms_iou_threshold: float,
    image: WorkflowImageData,
    mask_representation: str,
) -> InstanceDetections:
    """LOCAL-path builder. When NMS runs against the RLE carrier, reuse the COCO
    RLEs NMS already computed (avoids the double encode). Otherwise fall back to
    the v1 dense packer (`_build_instance_detections`)."""
    all_masks, kept_rles = _threshold_and_nms(
        per_prompt_results=per_prompt_results,
        prompts=prompts,
        global_confidence=global_confidence,
        apply_nms=apply_nms,
        nms_iou_threshold=nms_iou_threshold,
    )
    if mask_representation == "rle" and kept_rles is not None:
        return _build_instance_detections_reusing_nms_rles(
            all_masks=all_masks,
            kept_rles=kept_rles,
            class_names=class_names,
            image=image,
        )
    items: List[Item] = []
    for prompt_idx, mask, score in all_masks:
        class_name = class_names[prompt_idx] if prompt_idx < len(class_names) else None
        items.append(
            (mask.astype(bool), float(score), prompt_idx, class_name or "foreground")
        )
    return _build_instance_detections(
        items=items, image=image, mask_representation=mask_representation
    )
