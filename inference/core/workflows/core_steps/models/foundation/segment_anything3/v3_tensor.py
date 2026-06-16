"""Tensor-native sibling of `roboflow_core/sam3@v3`.

v3 = v2 (per-class thresholds + cross-prompt NMS) plus `class_mapping` (rename
predicted class names after inference). The numpy v3's `output_format`
(rle/polygons response-wire knob) has no LOCAL-tensor analog —
`run_tensor_native_inference` always returns raw binary masks — so it is replaced by
the tensor `mask_representation` carrier (rle/dense), whose default "rle" matches v3's
default `output_format="rle"`.

Output kinds: the numpy v3 declares DUAL kinds
(RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND + INSTANCE_SEGMENTATION_PREDICTION_KIND).
The tensor world has the matching pair
(TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND +
TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND), so `describe_outputs` declares
BOTH (RLE first) — matching the roboflow instance_segmentation tensor producers and
the established convention. At runtime there is still ONE `InstanceDetections`
carrier (rle or dense per `mask_representation`); the dual kind is purely a
wiring/advertisement concern. Downstream tensor consumers (mask/polygon/halo/icon
visualizers, trackers, fusion blocks) accept both kinds, so no graph is broken by
the carrier choice.

Private adapter coupling: the per-class-threshold + cross-prompt-NMS orchestration
is reused from `v2_tensor` (`_collect_from_native_with_nms`, `_min_floor`,
`_per_class_threshold`, `_build_http_prompts`). v2_tensor now holds LOCAL COPIES of
those adapter helpers (see its TODO(sam3-public-adapter) note) instead of importing
the private `inference.models.sam3` internals, so v3 no longer transitively depends
on that private cross-package surface.

`class_mapping` is applied on the flat `items` list before building
`InstanceDetections`, so both image_metadata's class-names map and per-instance
bboxes_metadata inherit the mapped names (v3's sv-based `_apply_class_mapping` is a
no-op on InstanceDetections and cannot be reused).
"""

from typing import Dict, List, Literal, Optional, Type, Union

import requests
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
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import build_roboflow_api_headers
from inference.core.utils.url_utils import wrap_url
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
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

# Reuse the v1_tensor conversion machinery + the v2_tensor per-class/NMS collector.
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1_tensor import (
    Item,
    _build_instance_detections,
    _build_instance_detections_from_polygons,
    _normalize_class_names,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2_tensor import (
    _build_http_prompts,
    _collect_from_native_with_nms,
    _min_floor,
    _per_class_threshold,
)

LONG_DESCRIPTION = """
Run Segment Anything 3 (zero-shot, text-prompted) with per-class confidence
thresholds, optional cross-prompt NMS, and post-inference class renaming.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3",
            "version": "v3",
            "short_description": "Sam3",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Sam3"],
            "ui_manifest": {
                "section": "model",
                "icon": "fa-solid fa-eye",
                "blockPriority": 9.48,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/sam3@v3"]
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
    class_mapping: Optional[Union[Dict[str, str], Selector(kind=[DICTIONARY_KIND])]] = Field(
        default=None,
        title="Class Mapping",
        description="Maps class names in predictions to different output names. Applied "
        "after inference, e.g. {'cat': 'gato'} renames 'cat' predictions to 'gato'.",
        examples=[{"cat": "gato", "dog": "perro"}],
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
        "'rle' on GCP_SERVERLESS regardless of this value. (Replaces v3 output_format.)",
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
        # Declare BOTH the RLE and the plain tensor instance-seg kinds (RLE first,
        # mirroring the default `mask_representation="rle"`). This restores the
        # dual-kind shape the numpy v3 sibling declares
        # (RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND +
        # INSTANCE_SEGMENTATION_PREDICTION_KIND) and matches the established tensor
        # convention used by the roboflow instance_segmentation tensor producers
        # (v1-v4) and accepted by every tensor instance-seg consumer (mask/polygon/
        # halo/icon visualizers, trackers, fusion blocks). The runtime carrier is
        # still ONE InstanceDetections (rle or dense per `mask_representation`); the
        # two kinds are advertised purely so downstream wiring that selects either
        # the rle-tensor or the plain-tensor kind resolves against this producer.
        # Audit: no tensor consumer requires the rle-tensor kind EXCLUSIVELY, so
        # the previous single-kind output was wiring-compatible too, but it diverged
        # from the dual-kind convention; this aligns it.
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
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


class SegmentAnything3BlockV3(WorkflowBlock):

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
        class_mapping: Optional[Dict[str, str]],
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
                class_mapping=class_mapping,
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
                class_mapping=class_mapping,
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
                class_mapping=class_mapping,
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
        class_mapping: Optional[Dict[str, str]],
        confidence: float,
        per_class_confidence: Optional[List[float]],
        apply_nms: bool,
        nms_iou_threshold: float,
        mask_representation: str,
    ) -> BlockResult:
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)

        sam3_prompts = [
            Sam3Prompt(
                type="text",
                text=class_name,
                output_prob_thresh=_per_class_threshold(per_class_confidence, idx),
            )
            for idx, class_name in enumerate(class_names)
        ]
        native_prompts = [{"text": cn} for cn in class_names]
        floor = _min_floor(confidence, per_class_confidence)

        results: List[dict] = []
        for single_image in images:
            per_image = self._model_manager.run_tensor_native_inference(
                model_id,
                images=[single_image.tensor_image],
                prompts=native_prompts,
                output_prob_thresh=float(floor),
            )
            items = _collect_from_native_with_nms(
                per_prompt_results=per_image[0],
                class_names=class_names,
                prompts=sam3_prompts,
                global_confidence=confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
            )
            items = _apply_class_mapping_to_items(items, class_mapping)
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
        class_mapping: Optional[Dict[str, str]],
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
                    class_mapping=class_mapping,
                    confidence=confidence,
                    mask_representation=mask_representation,
                )
            )
        return results

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[Optional[str]],
        class_mapping: Optional[Dict[str, str]],
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
                    class_mapping=class_mapping,
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
        class_mapping: Optional[Dict[str, str]],
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
                class_mapping=class_mapping,
            )
        }


def _apply_class_mapping_to_items(
    items: List[Item], class_mapping: Optional[Dict[str, str]]
) -> List[Item]:
    """Rename predicted class names on the flat items list before building
    InstanceDetections, so image_metadata's class-names map and per-instance
    bboxes_metadata both inherit the mapped names. (v3's sv-based
    _apply_class_mapping is a no-op on InstanceDetections and is not reused.)"""
    if not class_mapping:
        return items
    return [
        (mask, score, class_id, class_mapping.get(class_name, class_name))
        for (mask, score, class_id, class_name) in items
    ]
