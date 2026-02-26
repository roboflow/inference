from types import SimpleNamespace
from typing import List, Literal, Optional, Type, Union

import numpy as np
import requests
import supervision as sv
from pydantic import ConfigDict, Field, model_validator, validator

from inference.core import logger
from inference.core.entities.requests.sam3 import Sam3Prompt, Sam3SegmentationRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.entities.responses.sam3 import Sam3SegmentationPrediction
from inference.core.env import (
    API_BASE_URL,
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    SAM3_EXEC_MODE,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference_sdk import InferenceHTTPClient
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import build_roboflow_api_headers
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

DETECTIONS_CLASS_NAME_FIELD = "class_name"
DETECTION_ID_FIELD = "detection_id"


LONG_DESCRIPTION = """
Run Segment Anything 3, a zero-shot instance segmentation model, on an image.

You can pass in boxes/predictions from other models as prompts, or use a text prompt for open-vocabulary segmentation.
If you pass in box detections from another model, the class names of the boxes will be forwarded to the predicted masks.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3",
            "version": "v2",
            # "short_description": "Convert bounding boxes to polygons, or run SAM3 with optional text prompt to generate masks.",
            "short_description": "Sam3",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            # "search_keywords": ["SAM3", "META"],
            "search_keywords": ["Sam3"],
            "ui_manifest": {
                "section": "model",
                # "icon": "fa-brands fa-meta",
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
        # description="Model variant placeholder (SAM3 local image model).",
        description="model version.  You only need to change this for fine tuned sam3 models.",
        examples=[
            "sam3/sam3_final",
            "$inputs.model_variant",
        ],
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

    @validator("nms_iou_threshold")
    def _validate_nms_iou_threshold(cls, v):
        if isinstance(v, (int, float)) and (v < 0.0 or v > 1.0):
            raise ValueError("nms_iou_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _validate_per_class_confidence_length(self) -> "BlockManifest":
        if not isinstance(self.per_class_confidence, list):
            return self

        # Determine class_names length, handling both list and comma-separated string
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
        per_class_confidence: Optional[List[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
    ) -> BlockResult:

        if isinstance(class_names, str):
            class_names = class_names.split(",")
        elif isinstance(class_names, list):
            class_names = class_names
        else:
            raise ValueError(f"Invalid class names type: {type(class_names)}")

        if SAM3_EXEC_MODE == "remote":
            logger.debug("Running SAM3 v2 via inference proxy (SAM3_EXEC_MODE=remote)")
            return self.run_via_request(
                images=images,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
            )
        elif self._step_execution_mode is StepExecutionMode.LOCAL:
            logger.debug("Running SAM3 v2 locally")
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            logger.debug("Running SAM3 v2 remotely via SDK")
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_names: Optional[List[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
    ) -> BlockResult:
        predictions = []
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )

        for single_image in images:
            # Metadata for visual box prompts (if provided)
            prompt_class_ids: List[Optional[int]] = []
            prompt_class_names: List[Optional[str]] = []
            prompt_detection_ids: List[Optional[str]] = []

            # Build unified prompt list: one per class name
            unified_prompts: List[Sam3Prompt] = []
            for idx, class_name in enumerate(class_names):
                prompt_thresh = (
                    per_class_confidence[idx] if per_class_confidence else None
                )

                unified_prompts.append(
                    Sam3Prompt(
                        type="text", text=class_name, output_prob_thresh=prompt_thresh
                    )
                )

            # Single batched request with all prompts
            inference_request = Sam3SegmentationRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                model_id=model_id,
                api_key=self._api_key,
                prompts=unified_prompts,
                output_prob_thresh=confidence,
                nms_iou_threshold=nms_iou_threshold if apply_nms else None,
            )

            sam3_response = self._model_manager.infer_from_request_sync(
                model_id, inference_request
            )

            # Unpack unified batch response
            class_predictions = []
            for prompt_result in sam3_response.prompt_results:
                idx = prompt_result.prompt_index
                class_name = class_names[idx] if idx < len(class_names) else None
                class_pred = convert_sam3_segmentation_response_to_inference_instances_seg_response(
                    sam3_segmentation_predictions=prompt_result.predictions,
                    image=single_image,
                    prompt_class_ids=prompt_class_ids,
                    prompt_class_names=prompt_class_names,
                    prompt_detection_ids=prompt_detection_ids,
                    confidence=confidence,
                    text_prompt=class_name,
                    specific_class_id=idx,
                )
                class_predictions.extend(class_pred.predictions)

            image_width = single_image.numpy_image.shape[1]
            image_height = single_image.numpy_image.shape[0]
            final_inference_prediction = InstanceSegmentationInferenceResponse(
                predictions=class_predictions,
                image=InferenceResponseImage(width=image_width, height=image_height),
            )
            predictions.append(final_inference_prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_names: Optional[List[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
    ) -> BlockResult:
        predictions = []
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

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

        for single_image in images:
            prompt_class_ids: List[Optional[int]] = []
            prompt_class_names: List[Optional[str]] = []
            prompt_detection_ids: List[Optional[str]] = []

            http_prompts: List[dict] = []
            for idx, class_name in enumerate(class_names):
                prompt_data = {"type": "text", "text": class_name}
                if per_class_confidence is not None:
                    prompt_data["output_prob_thresh"] = per_class_confidence[idx]
                http_prompts.append(prompt_data)

            resp_json = client.sam3_concept_segment(
                inference_input=single_image.base64_image,
                prompts=http_prompts,
                model_id=model_id,
                output_prob_thresh=confidence,
                nms_iou_threshold=nms_iou_threshold if apply_nms else None,
            )

            class_predictions: List[InstanceSegmentationPrediction] = []
            for prompt_result in resp_json.get("prompt_results", []):
                idx = prompt_result.get("prompt_index", 0)
                class_name = class_names[idx] if idx < len(class_names) else None
                raw_predictions = prompt_result.get("predictions", [])
                adapted_predictions = [SimpleNamespace(**p) for p in raw_predictions]
                class_pred = convert_sam3_segmentation_response_to_inference_instances_seg_response(
                    sam3_segmentation_predictions=adapted_predictions,  # type: ignore[arg-type]
                    image=single_image,
                    prompt_class_ids=prompt_class_ids,
                    prompt_class_names=prompt_class_names,
                    prompt_detection_ids=prompt_detection_ids,
                    confidence=confidence,
                    text_prompt=class_name,
                    specific_class_id=idx,
                )
                class_predictions.extend(class_pred.predictions)

            image_width = single_image.numpy_image.shape[1]
            image_height = single_image.numpy_image.shape[0]
            final_inference_prediction = InstanceSegmentationInferenceResponse(
                predictions=class_predictions,
                image=InferenceResponseImage(width=image_width, height=image_height),
            )
            predictions.append(final_inference_prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: Optional[List[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
    ) -> BlockResult:
        predictions = []
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

        endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"
        api_key = self._api_key

        for single_image in images:
            prompt_class_ids: List[Optional[int]] = []
            prompt_class_names: List[Optional[str]] = []
            prompt_detection_ids: List[Optional[str]] = []

            # Build unified prompt list payloads for HTTP
            http_prompts: List[dict] = []
            for idx, class_name in enumerate(class_names):
                prompt_data = {"type": "text", "text": class_name}
                if per_class_confidence is not None:
                    prompt_data["output_prob_thresh"] = per_class_confidence[idx]
                http_prompts.append(prompt_data)

            # Prepare image for remote API (base64)
            http_image = {"type": "base64", "value": single_image.base64_image}

            payload = {
                "image": http_image,
                "prompts": http_prompts,
                "output_prob_thresh": confidence,
                "nms_iou_threshold": nms_iou_threshold if apply_nms else None,
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
                    f"{endpoint}?api_key={api_key}",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                response.raise_for_status()
                resp_json = response.json()
            except Exception:
                raise Exception(f"SAM3 request failed: {Exception}")

            class_predictions: List[InstanceSegmentationPrediction] = []
            for prompt_result in resp_json.get("prompt_results", []):
                idx = prompt_result.get("prompt_index", 0)
                class_name = class_names[idx] if idx < len(class_names) else None
                raw_predictions = prompt_result.get("predictions", [])
                # Adapt JSON dicts to objects with attribute-style access
                adapted_predictions = [SimpleNamespace(**p) for p in raw_predictions]
                class_pred = convert_sam3_segmentation_response_to_inference_instances_seg_response(
                    sam3_segmentation_predictions=adapted_predictions,  # type: ignore[arg-type]
                    image=single_image,
                    prompt_class_ids=prompt_class_ids,
                    prompt_class_names=prompt_class_names,
                    prompt_detection_ids=prompt_detection_ids,
                    confidence=confidence,
                    text_prompt=class_name,
                    specific_class_id=idx,
                )
                class_predictions.extend(class_pred.predictions)

            image_width = single_image.numpy_image.shape[1]
            image_height = single_image.numpy_image.shape[0]
            final_inference_prediction = InstanceSegmentationInferenceResponse(
                predictions=class_predictions,
                image=InferenceResponseImage(width=image_width, height=image_height),
            )
            predictions.append(final_inference_prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [{"predictions": prediction} for prediction in predictions]


def convert_sam3_segmentation_response_to_inference_instances_seg_response(
    sam3_segmentation_predictions: List[Sam3SegmentationPrediction],
    image: WorkflowImageData,
    prompt_class_ids: List[Optional[int]],
    prompt_class_names: List[Optional[str]],
    prompt_detection_ids: List[Optional[str]],
    confidence: float,
    text_prompt: Optional[str] = None,
    specific_class_id: Optional[int] = None,
) -> InstanceSegmentationInferenceResponse:
    image_width = image.numpy_image.shape[1]
    image_height = image.numpy_image.shape[0]
    predictions = []
    if len(prompt_class_ids) == 0:
        prompt_class_ids = [
            specific_class_id if specific_class_id else 0
            for _ in range(len(sam3_segmentation_predictions))
        ]
        prompt_class_names = [
            text_prompt if text_prompt else "foreground"
            for _ in range(len(sam3_segmentation_predictions))
        ]
        prompt_detection_ids = [None for _ in range(len(sam3_segmentation_predictions))]
    for prediction, class_id, class_name, detection_id in zip(
        sam3_segmentation_predictions,
        prompt_class_ids,
        prompt_class_names,
        prompt_detection_ids,
    ):
        for mask in prediction.masks:
            if len(mask) < 3:
                # skipping empty masks
                continue
            if prediction.confidence < confidence:
                # skipping masks below threshold
                continue
            x_coords = [coord[0] for coord in mask]
            y_coords = [coord[1] for coord in mask]
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            predictions.append(
                InstanceSegmentationPrediction(
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                        "points": [Point(x=point[0], y=point[1]) for point in mask],
                        "confidence": prediction.confidence,
                        "class": class_name,
                        "class_id": class_id,
                        "parent_id": detection_id,
                    }
                )
            )
    return InstanceSegmentationInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=image_width, height=image_height),
    )
