"""Tensor-native sibling of `roboflow_core/seg-preview@v1`.

Seg Preview is a text-prompted open-vocabulary instance-segmentation PRODUCER. In
numpy mode (`seg_preview/v1.py`) it POSTs to the Roboflow-internal
`API_BASE_URL/inferenceproxy/seg-preview` endpoint, which returns POLYGON
point-lists per mask, and builds `sv.Detections`.

Under ENABLE_TENSOR_DATA_REPRESENTATION this producer must instead emit a native
`inference_models.InstanceDetections` (torch tensors) under the tensor-native
kind. The remote-request path (base64 image, text prompts, proxy endpoint,
threshold, air-gapped/hosted restriction, the empty-class `None` append, and the
`except Exception -> empty results` fallback) is preserved verbatim from v1. Only
the post-processing tail changes: the polygon point-lists are rasterised
(cv2.fillPoly) and assembled into an InstanceDetections (RLE masks), mirroring
`segment_anything3/v1_tensor.py`.
"""

import uuid
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import requests
import torch
from pydantic import ConfigDict, Field

from inference.core.env import (
    API_BASE_URL,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
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
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
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


LONG_DESCRIPTION = "Seg Preview"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Seg Preview",
            "version": "v1",
            "short_description": "Seg Preview",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Seg Preview"],
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

    type: Literal["roboflow_core/seg-preview@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    class_names: Union[
        List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])
    ] = Field(
        title="Class Names",
        default=None,
        description="List of classes to recognise",
        examples=[["car", "person"], "$inputs.classes"],
    )
    threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.5, description="Threshold for predicted mask scores", examples=[0.3]
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
            RuntimeRestriction(
                severity=Severity.HARD,
                note=(
                    "Seg Preview calls the Roboflow-internal "
                    "API_BASE_URL/inferenceproxy/seg-preview endpoint, which is "
                    "only reachable from Roboflow-hosted runtimes "
                    "(HOSTED_SERVERLESS, DEDICATED_DEPLOYMENT). Self-hosted "
                    "deployments cannot run this block."
                ),
                applies_to_runtimes=[
                    Runtime.SELF_HOSTED_CPU,
                    Runtime.SELF_HOSTED_GPU,
                    Runtime.INFERENCE_PIPELINE,
                ],
            ),
        ]

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        """This block requires internet access to the remote inference proxy."""
        return AirGappedAvailability(available=False, reason="requires_internet")


class SegPreviewBlockV1(WorkflowBlock):

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
        class_names: Optional[Union[List[str], str]],
        threshold: float,
    ) -> BlockResult:

        if isinstance(class_names, str):
            class_names = class_names.split(",")
        elif isinstance(class_names, list):
            class_names = class_names
        else:
            raise ValueError(f"Invalid class names type: {type(class_names)}")

        return self.run_via_request(
            images=images,
            class_names=class_names,
            threshold=threshold,
        )

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: Optional[List[str]],
        threshold: float,
    ) -> BlockResult:
        results: List[dict] = []
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

        endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"
        api_key = self._api_key

        for single_image in images:
            # Build unified prompt list payloads for HTTP
            http_prompts: List[dict] = []
            for class_name in class_names:
                http_prompts.append({"type": "text", "text": class_name})

            # Prepare image for remote API (base64)
            http_image = {"type": "base64", "value": single_image.base64_image}

            payload = {
                "image": http_image,
                "prompts": http_prompts,
                "output_prob_thresh": threshold,
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
                    wrap_url(f"{endpoint}?api_key={api_key}"),
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                response.raise_for_status()
                resp_json = response.json()
            except Exception:
                resp_json = {"prompt_results": []}

            height, width = single_image._read_shape_without_materialization()
            items = _collect_from_polygons(
                prompt_results=resp_json.get("prompt_results", []),
                class_names=class_names,
                height=height,
                width=width,
                threshold=threshold,
            )
            results.append(
                {
                    "predictions": _build_instance_detections(
                        items=items,
                        image=single_image,
                    )
                }
            )
        return results


# Each item: (mask (H,W) bool, score, class_id, class_name)
Item = Tuple[np.ndarray, float, int, str]


def _collect_from_polygons(
    prompt_results: List[dict],
    class_names: List[Optional[str]],
    height: int,
    width: int,
    threshold: float,
) -> List[Item]:
    items: List[Item] = []
    for prompt_result in prompt_results:
        # Key class identity off the response's prompt_index field (not loop order),
        # matching the numpy block.
        idx = prompt_result.get("prompt_index", 0)
        class_name = class_names[idx] if idx < len(class_names) else None
        for prediction in prompt_result.get("predictions", []):
            confidence = float(prediction.get("confidence", 0.0))
            if confidence < threshold:
                continue
            for polygon in prediction.get("masks", []):
                if len(polygon) < 3:
                    # skipping empty masks
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
        xyxy.append(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        )
        confidences.append(score)
        class_ids.append(class_id)
        class_names_map[class_id] = class_name
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(uuid.uuid4()), CLASS_NAME_KEY: class_name}
        )
        kept_masks.append(mask)

    n = len(kept_masks)
    if n == 0:
        xyxy_t = torch.zeros((0, 4), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE)
        class_id_t = torch.zeros((0,), dtype=torch.int64, device=WORKFLOWS_IMAGE_TENSOR_DEVICE)
        confidence_t = torch.zeros((0,), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE)
        mask = InstancesRLEMasks(image_size=(height, width), masks=[])
    else:
        xyxy_t = torch.tensor(
            xyxy, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
        class_id_t = torch.tensor(
            class_ids, dtype=torch.int64, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
        confidence_t = torch.tensor(
            confidences, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
        rle_dicts = [
            torch_mask_to_coco_rle(torch.from_numpy(m)) for m in kept_masks
        ]
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(height, width), masks=rle_dicts
        )

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
