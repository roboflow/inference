from typing import List, Literal, Optional, Type, Union

import numpy as np
import requests
import supervision as sv
from pycocotools import mask as mask_utils
from pydantic import ConfigDict, Field, model_validator, validator

from inference.core import logger
from inference.core.entities.requests.sam3 import Sam3Prompt, Sam3SegmentationRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
    Point,
)
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
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
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
    LIST_OF_VALUES_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Run SAM3 with text prompts for zero-shot segmentation."

LONG_DESCRIPTION = """
Run Segment Anything 3 (SAM3), a zero-shot instance segmentation model that uses text prompts to segment objects in images.

## How This Block Works

This block takes one or more images as input and processes them through Meta's Segment Anything 3 (SAM3) model. SAM3 is a zero-shot segmentation model that can segment objects based on text descriptions without being trained on specific object classes. The block:

1. Takes your list of class names (e.g., ["person", "car", "bicycle"]) and one or more images
2. Processes each image through SAM3 to generate segmentation masks for objects matching your specified class names
3. Filters masks based on confidence thresholds (global and optionally per-class)
4. Applies Non-Maximum Suppression (NMS) across prompts to remove overlapping detections (if enabled)
5. Returns instance segmentation predictions in your chosen format (RLE or polygons) with bounding boxes, class names, and confidence scores

SAM3 uses text prompts to perform open-vocabulary segmentation, meaning you can specify any object classes in natural language without training the model on those specific classes. The model generates pixel-level masks for each detected instance of the specified classes.

This block supports two output formats:

- **rle** (default): Returns masks in RLE (Run-Length Encoding) format, which is more memory-efficient and recommended for high-resolution images or workflows with many detections
- **polygons**: Returns polygon coordinates for each mask, which is useful when you need explicit coordinate data

## Common Use Cases

- **Zero-Shot Segmentation**: Segment objects in images using text descriptions without training a custom segmentation model
- **Open-Vocabulary Segmentation**: Segment custom object categories by simply describing them in text (e.g., "red car", "person wearing helmet", "dog")
- **Precise Object Segmentation**: Generate pixel-accurate masks for objects, useful for detailed analysis, measurement, or extraction
- **Multi-Class Segmentation**: Segment multiple object types in a single pass by specifying multiple class names
- **High-Resolution Processing**: Use RLE format for memory-efficient processing of high-resolution images or images with many detections
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

## Version Differences (v3 vs v2)

This version (v3) includes the following enhancement over v2:

- **Output Format Selection**: Added `output_format` parameter to choose between RLE (Run-Length Encoding) and polygon formats. RLE format is more memory-efficient and recommended for high-resolution images or workflows with many detections, while polygon format provides explicit coordinate data.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3",
            "version": "v3",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Sam",
                "SAM3",
                "segment anything",
                "segment anything 3",
            ],
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

    output_format: Literal["rle", "polygons"] = Field(
        default="rle",
        title="Output Format",
        description="Format for segmentation mask output. 'rle' (Run-Length Encoding) returns masks in a memory-efficient compressed format, recommended for high-resolution images or workflows with many detections. 'polygons' returns explicit polygon coordinates for each mask, useful when you need coordinate data for further processing. Default is 'rle'.",
        examples=["rle", "polygons"],
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
                kind=[
                    # RLE first since it's the default and more efficient
                    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


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
        confidence: float,
        per_class_confidence: Optional[List[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
        output_format: Literal["rle", "polygons"] = "rle",
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
            exec_mode = StepExecutionMode.REMOTE
        else:
            raise ValueError(
                f"Invalid SAM3 execution mode in ENVIRONMENT var SAM3_EXEC_MODE (local or remote): {SAM3_EXEC_MODE}"
            )

        if exec_mode is StepExecutionMode.LOCAL:
            logger.debug(f"Running SAM3 v3 locally with output_format={output_format}")
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
                output_format=output_format,
            )
        elif exec_mode is StepExecutionMode.REMOTE:
            logger.debug(f"Running SAM3 v3 remotely with output_format={output_format}")
            return self.run_via_request(
                images=images,
                class_names=class_names,
                confidence=confidence,
                per_class_confidence=per_class_confidence,
                apply_nms=apply_nms,
                nms_iou_threshold=nms_iou_threshold,
                output_format=output_format,
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
        output_format: Literal["rle", "polygons"] = "rle",
    ) -> BlockResult:
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )

        # Determine format to request from model
        model_format = "rle" if output_format == "rle" else "polygon"

        all_detections = []
        for single_image in images:
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
                format=model_format,
            )

            sam3_response = self._model_manager.infer_from_request_sync(
                model_id, inference_request
            )

            image_width = single_image.numpy_image.shape[1]
            image_height = single_image.numpy_image.shape[0]

            if output_format == "rle":
                # RLE output: build sv.Detections with RLE in data
                detections = self._convert_rle_response_to_sv_detections(
                    sam3_response=sam3_response,
                    class_names=class_names,
                    confidence=confidence,
                    image_height=image_height,
                    image_width=image_width,
                )
            else:
                # Polygon output: use existing conversion
                inference_response = self._convert_polygon_response_to_inference_format(
                    sam3_response=sam3_response,
                    class_names=class_names,
                    confidence=confidence,
                    image_height=image_height,
                    image_width=image_width,
                )
                detections = sv.Detections.from_inference(inference_response)
                detections[DETECTION_ID_KEY] = np.array(
                    [p.detection_id for p in inference_response.predictions]
                )
                detections[PARENT_ID_KEY] = np.array([""] * len(detections))
                detections[IMAGE_DIMENSIONS_KEY] = np.array(
                    [[image_height, image_width]] * len(detections)
                )

            all_detections.append(detections)

        return self._post_process_result(
            images=images,
            predictions=all_detections,
            output_format=output_format,
        )

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        class_names: Optional[List[str]],
        confidence: float,
        per_class_confidence: Optional[List[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
        output_format: Literal["rle", "polygons"] = "rle",
    ) -> BlockResult:
        if class_names is None:
            class_names = []
        if len(class_names) == 0:
            class_names.append(None)

        endpoint = f"{API_BASE_URL}/inferenceproxy/seg-preview"
        api_key = self._api_key
        model_format = "rle" if output_format == "rle" else "polygon"

        all_detections = []
        for single_image in images:
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
                "format": model_format,
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
            except Exception as e:
                raise Exception(f"SAM3 request failed: {e}")

            image_width = single_image.numpy_image.shape[1]
            image_height = single_image.numpy_image.shape[0]

            if output_format == "rle":
                # RLE output
                detections = self._convert_rle_json_response_to_sv_detections(
                    resp_json=resp_json,
                    class_names=class_names,
                    confidence=confidence,
                    image_height=image_height,
                    image_width=image_width,
                )
            else:
                # Polygon output
                inference_response = (
                    self._convert_polygon_json_response_to_inference_format(
                        resp_json=resp_json,
                        class_names=class_names,
                        confidence=confidence,
                        image_height=image_height,
                        image_width=image_width,
                    )
                )
                detections = sv.Detections.from_inference(inference_response)
                detections[DETECTION_ID_KEY] = np.array(
                    [p.detection_id for p in inference_response.predictions]
                )
                detections[PARENT_ID_KEY] = np.array([""] * len(detections))
                detections[IMAGE_DIMENSIONS_KEY] = np.array(
                    [[image_height, image_width]] * len(detections)
                )

            all_detections.append(detections)

        return self._post_process_result(
            images=images,
            predictions=all_detections,
            output_format=output_format,
        )

    @staticmethod
    def build_rle_prediction(
        rle: dict,
        bbox: List[float],
        pred_confidence: float,
        class_name: str,
        class_id: int,
    ) -> InstanceSegmentationRLEPrediction:
        x, y, w, h = bbox
        return InstanceSegmentationRLEPrediction(
            **{
                "x": x + w / 2,
                "y": y + h / 2,
                "width": w,
                "height": h,
                "confidence": pred_confidence,
                "class": class_name,
                "class_id": class_id,
                "rle": rle,
            }
        )

    @staticmethod
    def build_rle_detections(
        predictions: List[InstanceSegmentationRLEPrediction],
        image_height: int,
        image_width: int,
    ) -> sv.Detections:
        if len(predictions) == 0:
            return sv.Detections.empty()

        xyxy_list = []
        for p in predictions:
            x1 = p.x - p.width / 2
            y1 = p.y - p.height / 2
            x2 = p.x + p.width / 2
            y2 = p.y + p.height / 2
            xyxy_list.append([x1, y1, x2, y2])

        detections = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array([p.confidence for p in predictions], dtype=np.float32),
            class_id=np.array([p.class_id for p in predictions], dtype=int),
        )
        detections["class_name"] = np.array([p.class_name for p in predictions])
        detections[DETECTION_ID_KEY] = np.array([p.detection_id for p in predictions])
        detections[PARENT_ID_KEY] = np.array([p.parent_id or "" for p in predictions])
        detections[IMAGE_DIMENSIONS_KEY] = np.array(
            [[image_height, image_width]] * len(detections)
        )
        detections[RLE_MASK_KEY_IN_SV_DETECTIONS] = np.array(
            [p.rle for p in predictions], dtype=object
        )

        return detections

    def _convert_rle_response_to_sv_detections(
        self,
        sam3_response,
        class_names: List[Optional[str]],
        confidence: float,
        image_height: int,
        image_width: int,
    ) -> sv.Detections:
        predictions: List[InstanceSegmentationRLEPrediction] = []

        for prompt_result in sam3_response.prompt_results:
            idx = prompt_result.prompt_index
            class_name = class_names[idx] if idx < len(class_names) else "foreground"
            if class_name is None:
                class_name = "foreground"

            for prediction in prompt_result.predictions:
                if prediction.confidence < confidence:
                    continue

                rle = prediction.masks
                if not isinstance(rle, dict) or "counts" not in rle:
                    continue

                bbox = mask_utils.toBbox(rle)
                pred = self.build_rle_prediction(
                    rle=rle,
                    bbox=bbox,
                    pred_confidence=prediction.confidence,
                    class_name=class_name,
                    class_id=idx,
                )
                predictions.append(pred)

        return self.build_rle_detections(predictions, image_height, image_width)

    def _convert_rle_json_response_to_sv_detections(
        self,
        resp_json: dict,
        class_names: List[Optional[str]],
        confidence: float,
        image_height: int,
        image_width: int,
    ) -> sv.Detections:
        predictions: List[InstanceSegmentationRLEPrediction] = []

        for prompt_result in resp_json.get("prompt_results", []):
            idx = prompt_result.get("prompt_index", 0)
            class_name = class_names[idx] if idx < len(class_names) else "foreground"
            if class_name is None:
                class_name = "foreground"

            for prediction in prompt_result.get("predictions", []):
                pred_confidence = prediction.get("confidence", 0.0)
                if pred_confidence < confidence:
                    continue

                rle = prediction.get("masks")
                if not isinstance(rle, dict) or "counts" not in rle:
                    continue

                bbox = mask_utils.toBbox(rle)
                pred = self.build_rle_prediction(
                    rle=rle,
                    bbox=bbox,
                    pred_confidence=pred_confidence,
                    class_name=class_name,
                    class_id=idx,
                )
                predictions.append(pred)

        return self.build_rle_detections(predictions, image_height, image_width)

    @staticmethod
    def build_polygon_prediction(
        mask: List[List[float]],
        pred_confidence: float,
        class_name: str,
        class_id: int,
    ) -> Optional[InstanceSegmentationPrediction]:
        if len(mask) < 3:
            return None

        x_coords = [coord[0] for coord in mask]
        y_coords = [coord[1] for coord in mask]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        return InstanceSegmentationPrediction(
            **{
                "x": (min_x + max_x) / 2,
                "y": (min_y + max_y) / 2,
                "width": max_x - min_x,
                "height": max_y - min_y,
                "points": [Point(x=p[0], y=p[1]) for p in mask],
                "confidence": pred_confidence,
                "class": class_name,
                "class_id": class_id,
            }
        )

    def _convert_polygon_response_to_inference_format(
        self,
        sam3_response,
        class_names: List[Optional[str]],
        confidence: float,
        image_height: int,
        image_width: int,
    ) -> InstanceSegmentationInferenceResponse:
        predictions: List[InstanceSegmentationPrediction] = []

        for prompt_result in sam3_response.prompt_results:
            idx = prompt_result.prompt_index
            class_name = class_names[idx] if idx < len(class_names) else "foreground"
            if class_name is None:
                class_name = "foreground"

            for prediction in prompt_result.predictions:
                if prediction.confidence < confidence:
                    continue

                for mask in prediction.masks:
                    pred = self.build_polygon_prediction(
                        mask, prediction.confidence, class_name, idx
                    )
                    if pred:
                        predictions.append(pred)

        return InstanceSegmentationInferenceResponse(
            predictions=predictions,
            image=InferenceResponseImage(width=image_width, height=image_height),
        )

    def _convert_polygon_json_response_to_inference_format(
        self,
        resp_json: dict,
        class_names: List[Optional[str]],
        confidence: float,
        image_height: int,
        image_width: int,
    ) -> InstanceSegmentationInferenceResponse:
        predictions: List[InstanceSegmentationPrediction] = []

        for prompt_result in resp_json.get("prompt_results", []):
            idx = prompt_result.get("prompt_index", 0)
            class_name = class_names[idx] if idx < len(class_names) else "foreground"
            if class_name is None:
                class_name = "foreground"

            for prediction in prompt_result.get("predictions", []):
                pred_confidence = prediction.get("confidence", 0.0)
                if pred_confidence < confidence:
                    continue

                for mask in prediction.get("masks", []):
                    pred = self.build_polygon_prediction(
                        mask, pred_confidence, class_name, idx
                    )
                    if pred:
                        predictions.append(pred)

        return InstanceSegmentationInferenceResponse(
            predictions=predictions,
            image=InferenceResponseImage(width=image_width, height=image_height),
        )

    @staticmethod
    def decode_rle_masks(rle_list: np.ndarray) -> np.ndarray:
        masks = [mask_utils.decode(rle).astype(bool) for rle in rle_list]
        return np.array(masks)

    def _decode_and_cache_rle_masks(
        self,
        predictions: List[sv.Detections],
    ) -> List[sv.Detections]:
        for detection in predictions:
            if (
                detection.mask is None
                and RLE_MASK_KEY_IN_SV_DETECTIONS in detection.data
            ):
                rle_masks = detection.data[RLE_MASK_KEY_IN_SV_DETECTIONS]
                detection.mask = self.decode_rle_masks(rle_masks)
        return predictions

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[sv.Detections],
        output_format: Literal["rle", "polygons"],
    ) -> BlockResult:
        prediction_type = (
            "rle-instance-segmentation"
            if output_format == "rle"
            else "instance-segmentation"
        )
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type=prediction_type,
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        if output_format == "rle":
            predictions = self._decode_and_cache_rle_masks(predictions)
        return [{"predictions": prediction} for prediction in predictions]
