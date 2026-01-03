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
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    SAM3_EXEC_MODE,
)
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
Run Segment Anything 3 (SAM3), a zero-shot instance segmentation model that uses text prompts to segment objects in images.

## How This Block Works

This block takes one or more images as input and processes them through Meta's Segment Anything 3 (SAM3) model. SAM3 is a zero-shot segmentation model that can segment objects based on text descriptions without being trained on specific object classes. The block:

1. Takes your list of class names (e.g., ["person", "car", "bicycle"]) and one or more images
2. Processes each image through SAM3 to generate segmentation masks for objects matching your specified class names
3. Filters masks based on confidence thresholds (global and optionally per-class)
4. Applies Non-Maximum Suppression (NMS) across prompts to remove overlapping detections (if enabled)
5. Returns instance segmentation predictions with polygon masks, bounding boxes, class names, and confidence scores

SAM3 uses text prompts to perform open-vocabulary segmentation, meaning you can specify any object classes in natural language without training the model on those specific classes. The model generates pixel-level masks (polygons) for each detected instance of the specified classes.

## Common Use Cases

- **Zero-Shot Segmentation**: Segment objects in images using text descriptions without training a custom segmentation model
- **Open-Vocabulary Segmentation**: Segment custom object categories by simply describing them in text (e.g., "red car", "person wearing helmet", "dog")
- **Precise Object Segmentation**: Generate pixel-accurate masks for objects, useful for detailed analysis, measurement, or extraction
- **Multi-Class Segmentation**: Segment multiple object types in a single pass by specifying multiple class names
- **Content Analysis**: Identify and segment specific content or objects in images for detailed analysis
- **Image Editing and Processing**: Extract precise object masks for downstream processing, editing, or composition tasks

## Requirements

**⚠️ Important: GPU Required**

This block requires a **GPU** for best performance. The execution mode (local or remote) is controlled by the `SAM3_EXEC_MODE` environment variable. For local execution, ensure you have a GPU available. For remote execution, the model runs on Roboflow's infrastructure.

## Connecting to Other Blocks

The instance segmentation predictions from this block can be connected to:

- **Visualization blocks** (e.g., Mask Visualization, Bounding Box Visualization) to draw segmentation results on images
- **Filter blocks** (e.g., Detections Filter) to filter segmentation results based on confidence, class, area, or other criteria
- **Transformation blocks** (e.g., Dynamic Crop) to extract regions based on segmented masks
- **Analytics blocks** (e.g., Data Aggregator) to analyze segmentation results over time
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on segmentation results
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log segmentation results

## Version Differences (v2 vs v1)

This version (v2) includes several enhancements over v1:

- **Per-Class Confidence Thresholds**: Added `per_class_confidence` parameter to set different confidence thresholds for each class, allowing fine-tuned control over detection sensitivity per object type
- **Non-Maximum Suppression (NMS)**: Added `apply_nms` and `nms_iou_threshold` parameters to remove overlapping detections across different prompts, improving result quality when segmenting multiple classes
- **Parameter Renaming**: The `threshold` parameter has been renamed to `confidence` for clarity
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
        description="The SAM3 model to use for inference. Default is 'sam3/sam3_final'. You only need to change this for fine-tuned SAM3 models.",
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
        description="List of class names (text prompts) to segment in the images. Provide a list of strings describing the objects you want to segment (e.g., ['person', 'car', 'bicycle']). You can also provide a comma-separated string. SAM3 uses these text descriptions for zero-shot open-vocabulary segmentation. The length must match `per_class_confidence` if provided.",
        examples=[["car", "person"], "car,person", "$inputs.classes"],
    )
    confidence: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.5,
        title="Confidence Threshold",
        description="Global confidence threshold for predicted mask scores (0.0 to 1.0). Only segmentation masks with confidence scores above this threshold will be returned. This serves as the default threshold; you can override it per-class using `per_class_confidence`. Lower values return more masks (including lower confidence ones), while higher values return only high-confidence masks. Default is 0.5.",
        examples=[0.3, 0.5, 0.7],
    )

    per_class_confidence: Optional[
        Union[List[float], Selector(kind=[LIST_OF_VALUES_KIND])]
    ] = Field(
        default=None,
        title="Per-Class Confidence",
        description="Optional list of confidence thresholds (0.0 to 1.0) for each class, allowing fine-tuned control over detection sensitivity per object type. The length must exactly match the number of class names. If provided, these thresholds override the global `confidence` threshold for each corresponding class. Useful when different object types require different sensitivity levels (e.g., lower threshold for rare objects, higher threshold for common objects).",
        examples=[[0.3, 0.5, 0.7]],
    )

    apply_nms: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        title="Apply NMS",
        description="Whether to apply Non-Maximum Suppression (NMS) across prompts. When enabled, NMS removes overlapping detections from different class prompts, improving result quality when segmenting multiple classes that might overlap. Default is True.",
    )

    nms_iou_threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.9,
        title="NMS IoU Threshold",
        description="Intersection over Union (IoU) threshold for cross-prompt Non-Maximum Suppression (0.0 to 1.0). Detections with IoU above this threshold are considered overlapping and the lower-confidence one is removed. Higher values (e.g., 0.9) allow more overlapping detections, while lower values (e.g., 0.5) are more aggressive at removing overlaps. Default is 0.9. Only used when `apply_nms` is True.",
        examples=[0.5, 0.7, 0.9],
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

        exec_mode = self._step_execution_mode
        if SAM3_EXEC_MODE == "local":
            exec_mode = self._step_execution_mode
        elif SAM3_EXEC_MODE == "remote":
            exec_mode = (
                StepExecutionMode.REMOTE
            )  # if SAM3_EXEC_MODE == "remote" then force remote execution mode only
        else:
            raise ValueError(
                f"Invalid SAM3 execution mode in ENVIRONMENT var SAM3_EXEC_MODE (local or remote): {SAM3_EXEC_MODE}"
            )

        if exec_mode is StepExecutionMode.LOCAL:
            logger.debug(f"Running SAM3 locally")
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
            )
        elif exec_mode is StepExecutionMode.REMOTE:
            logger.debug(f"Running SAM3 remotely")
            return self.run_via_request(
                images=images,
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
