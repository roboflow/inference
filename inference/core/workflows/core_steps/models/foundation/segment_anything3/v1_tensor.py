"""Tensor-native sibling of `roboflow_core/sam3@v1`.

SCRATCH — first pass for review. SAM3 v1 is a TEXT-prompted open-vocabulary
instance-segmentation producer (one text prompt per `class_names` entry; SAM3
returns N masks per prompt with per-mask scores). Output is tensor-native
`inference_models.InstanceDetections` (RLE masks by default), combining all
prompts/classes into one prediction per image, with class_id = prompt index and
the class name forwarded onto each mask (mirrors the numpy block's
convert_sam3_segmentation_response_to_inference_instances_seg_response).

Paths:
- LOCAL: `run_tensor_native_inference(model_id, images=[tensor_image],
  prompts=[{"text": class_name}], output_prob_thresh=threshold)` -> raw per-prompt
  `{"masks": (N,1,H,W) bool ndarray, "scores": [...]}` (PostProcessImage leaves a
  channel dim from mask.unsqueeze(1); squeezed to (N,H,W) on ingestion). Masks are
  already binary (sigmoid>0.5) and threshold-filtered; built into InstanceDetections here.
- REMOTE / proxy: the server returns polygon predictions; rasterised to masks
  (cv2.fillPoly) then built into InstanceDetections — parity with v1.
"""

import uuid
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import requests
import torch
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.env import (
    API_BASE_URL,
    CORE_MODEL_SAM3_ENABLED,
    GCP_SERVERLESS,
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    SAM3_EXEC_MODE,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import build_roboflow_api_headers
from inference.core.utils.url_utils import wrap_url
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
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
from inference_sdk import InferenceHTTPClient

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

PREDICTION_TYPE = "instance-segmentation"

LONG_DESCRIPTION = """
Run Segment Anything 3, a zero-shot instance segmentation model, on an image.

You can use a text prompt (class names) for open-vocabulary segmentation.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3",
            "version": "v1",
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

    type: Literal["roboflow_core/sam3@v1"]
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
    threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.5, description="Threshold for predicted mask scores", examples=[0.3]
    )
    mask_representation: Literal["rle", "dense"] = Field(
        default="rle",
        description="Carrier for instance masks. RLE (compact) by default; forced to "
        "'rle' on GCP_SERVERLESS regardless of this value.",
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


class SegmentAnything3BlockV1(WorkflowBlock):

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
        threshold: float,
        mask_representation: Literal["rle", "dense"],
    ) -> BlockResult:
        if GCP_SERVERLESS:
            mask_representation = "rle"
        class_names = _normalize_class_names(class_names)
        if SAM3_EXEC_MODE == "remote":
            return self.run_via_request(
                images=images,
                class_names=class_names,
                threshold=threshold,
                mask_representation=mask_representation,
            )
        elif self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_names=class_names,
                threshold=threshold,
                mask_representation=mask_representation,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_names=class_names,
                threshold=threshold,
                mask_representation=mask_representation,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_names: List[Optional[str]],
        threshold: float,
        mask_representation: str,
    ) -> BlockResult:
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        prompts = [{"text": class_name} for class_name in class_names]

        results: List[dict] = []
        for single_image in images:
            per_image = self._model_manager.run_tensor_native_inference(
                model_id,
                images=[single_image.tensor_image],
                prompts=prompts,
                output_prob_thresh=threshold,
            )
            items = _collect_from_native(
                per_prompt_results=per_image[0],
                class_names=class_names,
                threshold=threshold,
            )
            results.append(
                {
                    "predictions": _build_instance_detections(
                        items=items,
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
        threshold: float,
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
        http_prompts = [{"type": "text", "text": cn} for cn in class_names]

        results: List[dict] = []
        for single_image in images:
            resp_json = client.sam3_concept_segment(
                inference_input=single_image.base64_image,
                prompts=http_prompts,
                model_id=model_id,
                output_prob_thresh=threshold,
            )
            results.append(
                self._build_from_polygon_response(
                    resp_json=resp_json,
                    image=single_image,
                    class_names=class_names,
                    threshold=threshold,
                    mask_representation=mask_representation,
                )
            )
        return results

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[Optional[str]],
        threshold: float,
        mask_representation: str,
    ) -> BlockResult:
        endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"
        http_prompts = [{"type": "text", "text": cn} for cn in class_names]

        results: List[dict] = []
        for single_image in images:
            payload = {
                "image": {"type": "base64", "value": single_image.base64_image},
                "prompts": http_prompts,
                "output_prob_thresh": threshold,
            }
            headers = {"Content-Type": "application/json"}
            if ROBOFLOW_INTERNAL_SERVICE_NAME:
                headers["X-Roboflow-Internal-Service-Name"] = ROBOFLOW_INTERNAL_SERVICE_NAME
            if ROBOFLOW_INTERNAL_SERVICE_SECRET:
                headers["X-Roboflow-Internal-Service-Secret"] = ROBOFLOW_INTERNAL_SERVICE_SECRET
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
                    threshold=threshold,
                    mask_representation=mask_representation,
                )
            )
        return results

    def _build_from_polygon_response(
        self,
        resp_json: dict,
        image: WorkflowImageData,
        class_names: List[Optional[str]],
        threshold: float,
        mask_representation: str,
    ) -> dict:
        height, width = image._read_shape_without_materialization()
        items = _collect_from_polygons(
            prompt_results=resp_json.get("prompt_results", []),
            class_names=class_names,
            height=height,
            width=width,
            threshold=threshold,
        )
        return {
            "predictions": _build_instance_detections(
                items=items, image=image, mask_representation=mask_representation
            )
        }


def _normalize_class_names(class_names) -> List[Optional[str]]:
    if isinstance(class_names, str):
        class_names = class_names.split(",")
    elif isinstance(class_names, list):
        class_names = list(class_names)
    elif class_names is None:
        class_names = []
    else:
        raise ValueError(f"Invalid class names type: {type(class_names)}")
    if len(class_names) == 0:
        # A single null prompt = unprompted (class_id 0 / "foreground").
        class_names = [None]
    return class_names


# Each item: (mask (H,W) bool, score, class_id, class_name)
Item = Tuple[np.ndarray, float, int, str]


def _collect_from_native(
    per_prompt_results: List[dict],
    class_names: List[Optional[str]],
    threshold: float,
) -> List[Item]:
    items: List[Item] = []
    for prompt_result in per_prompt_results:
        # Key class identity off the model's prompt_index field (not the loop
        # order), matching the numpy block + the polygon path below.
        idx = prompt_result.get("prompt_index", 0)
        class_name = class_names[idx] if idx < len(class_names) else None
        masks = prompt_result.get("masks")
        if masks is None:
            continue
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            # SAM3 PostProcessImage (non-RLE) returns (N, 1, H, W): the channel dim
            # from mask.unsqueeze(1) is never squeezed on this path. Drop it so
            # downstream np.where / RLE / stack see true (N, H, W) masks.
            masks = masks[:, 0]
        scores = list(prompt_result.get("scores", []))
        for i in range(masks.shape[0]):
            score = float(scores[i]) if i < len(scores) else 1.0
            if score < threshold:
                continue
            items.append(
                (masks[i].astype(bool), score, idx, class_name or "foreground")
            )
    return items


def _collect_from_polygons(
    prompt_results: List[dict],
    class_names: List[Optional[str]],
    height: int,
    width: int,
    threshold: float,
) -> List[Item]:
    items: List[Item] = []
    for prompt_result in prompt_results:
        idx = prompt_result.get("prompt_index", 0)
        class_name = class_names[idx] if idx < len(class_names) else None
        for prediction in prompt_result.get("predictions", []):
            confidence = float(prediction.get("confidence", 0.0))
            if confidence < threshold:
                continue
            for polygon in prediction.get("masks", []):
                if len(polygon) < 3:
                    continue
                mask = _polygon_to_mask(polygon, height, width)
                items.append((mask, confidence, idx, class_name or "foreground"))
    return items


def _polygon_to_mask(polygon, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(
        [[int(round(p[0])), int(round(p[1]))] for p in polygon], dtype=np.int32
    )
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _build_instance_detections(
    items: List[Item],
    image: WorkflowImageData,
    mask_representation: str,
) -> InstanceDetections:
    height, width = image._read_shape_without_materialization()
    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    class_names_map: Dict[int, str] = {}
    bboxes_metadata: List[dict] = []
    kept_masks: List[np.ndarray] = []

    for mask, score, class_id, class_name in items:
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        xyxy.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        confidences.append(score)
        class_ids.append(class_id)
        class_names_map[class_id] = class_name
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(uuid.uuid4()), CLASS_NAME_KEY: class_name}
        )
        kept_masks.append(mask)

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
            rle_dicts = [
                torch_mask_to_coco_rle(torch.from_numpy(m)) for m in kept_masks
            ]
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
