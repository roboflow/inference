from typing import List, Literal, Optional, Type, TypeVar, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.sam2 import (
    Box,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.entities.responses.sam2 import Sam2SegmentationPrediction
from inference.core.managers.base import ModelManager
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
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

T = TypeVar("T")
K = TypeVar("K")

DETECTIONS_CLASS_NAME_FIELD = "class_name"
DETECTION_ID_FIELD = "detection_id"

LONG_DESCRIPTION = """
Run Segment Anything 2 (SAM2), a zero-shot instance segmentation model that converts bounding boxes to precise segmentation masks.

## How This Block Works

This block takes one or more images as input and processes them through Meta's Segment Anything 2 (SAM2) model. SAM2 is designed to convert bounding box prompts into precise pixel-level segmentation masks. The block:

1. Takes images and optionally bounding boxes from other models (object detection, keypoint detection, or instance segmentation)
2. If boxes are provided, uses them as prompts to generate precise segmentation masks for each box
3. If no boxes are provided, can generate masks for the entire image (unprompted mode)
4. Filters masks based on the confidence threshold you specify
5. Returns instance segmentation predictions with polygon masks, bounding boxes, class names (from input boxes if provided), and confidence scores

When you provide bounding boxes from other models, SAM2 generates precise segmentation masks that follow the contours of the objects. The class names from the input boxes are preserved in the output masks. If you run SAM2 unprompted (without boxes), the model assigns generic class names.

## Common Use Cases

- **Box-to-Polygon Conversion**: Convert bounding boxes from object detection models into precise segmentation masks for detailed object analysis
- **Precise Object Segmentation**: Generate pixel-accurate masks from bounding boxes, useful for detailed measurement, analysis, or extraction
- **Multi-Model Workflows**: Combine object detection models with SAM2 to get both fast detections and precise segmentations
- **Image-Wide Segmentation**: Generate segmentation masks for entire images when run without box prompts
- **Quality Enhancement**: Improve the precision of bounding box detections by converting them to detailed segmentation masks
- **Measurement and Analysis**: Extract precise object boundaries for measurement, area calculation, or detailed shape analysis

## Requirements

**⚠️ Important: Dedicated Inference Server Required**

This block requires **local execution** (cannot run remotely). A **GPU is recommended** for best performance. The model requires appropriate dependencies to be installed.

## Connecting to Other Blocks

The instance segmentation predictions from this block can be connected to:

- **Visualization blocks** (e.g., Mask Visualization, Bounding Box Visualization) to draw segmentation results on images
- **Filter blocks** (e.g., Detections Filter) to filter segmentation results based on confidence, class, area, or other criteria
- **Transformation blocks** (e.g., Dynamic Crop) to extract regions based on segmented masks
- **Analytics blocks** (e.g., Data Aggregator) to analyze segmentation results over time
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on segmentation results
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log segmentation results

This block is commonly used **after** object detection blocks to convert their bounding box outputs into precise segmentation masks.
"""


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
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        description="Optional bounding boxes from other models (object detection, instance segmentation, or keypoint detection) to use as prompts for SAM2. When provided, SAM2 will generate precise segmentation masks for each box, preserving the class names from the input detections. If not provided, SAM2 will run in unprompted mode.",
        examples=["$steps.object_detection_model.predictions"],
        default=None,
        json_schema_extra={"always_visible": True},
    )
    version: Union[
        Selector(kind=[STRING_KIND]),
        Literal["hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus"],
    ] = Field(
        default="hiera_tiny",
        description="The SAM2 model variant to use. Options include 'hiera_tiny' (smallest, fastest), 'hiera_small', 'hiera_b_plus', and 'hiera_large' (largest, most accurate). Larger models are more accurate but slower and require more GPU memory. Default is 'hiera_tiny'.",
        examples=["hiera_tiny", "hiera_large", "$inputs.model_version"],
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.0,
        description="Confidence threshold for predicted mask scores (0.0 to 1.0). Only segmentation masks with confidence scores above this threshold will be returned. Lower values return more masks (including lower confidence ones), while higher values return only high-confidence masks. Default is 0.0 (returns all masks).",
        examples=[0.0, 0.3, 0.5],
    )

    multimask_output: Union[Optional[bool], Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Whether to use SAM2's internal multimask or single mask mode. When True (default), SAM2 generates multiple candidate masks for each prompt and selects the best one - recommended for ambiguous prompts where multiple valid segmentations might exist. When False, SAM2 generates a single mask per prompt, which is faster but may be less optimal for ambiguous cases.",
        examples=[True, False, "$inputs.multimask_output"],
    )

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
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch[sv.Detections]],
        version: str,
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                boxes=boxes,
                version=version,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Segment Anything. Run a local or dedicated inference server to use this block (GPU recommended)."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch[sv.Detections]],
        version: str,
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:

        predictions = []
        if boxes is None:
            boxes = [None] * len(images)

        for single_image, boxes_for_image in zip(images, boxes):
            prompt_class_ids: List[Optional[int]] = []
            prompt_class_names: List[str] = []
            prompt_detection_ids: List[Optional[str]] = []

            prompts = []
            if boxes_for_image is not None:
                for xyxy, _, confidence, class_id, _, bbox_data in boxes_for_image:
                    x1, y1, x2, y2 = xyxy
                    prompt_class_ids.append(class_id)
                    prompt_class_names.append(bbox_data[DETECTIONS_CLASS_NAME_FIELD])
                    prompt_detection_ids.append(bbox_data[DETECTION_ID_FIELD])
                    width = x2 - x1
                    height = y2 - y1
                    cx = x1 + width / 2
                    cy = y1 + height / 2
                    prompt = Sam2Prompt(
                        box=Box(
                            x=cx,
                            y=cy,
                            width=width,
                            height=height,
                        )
                    )
                    prompts.append(prompt)
            inference_request = Sam2SegmentationRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                sam2_version_id=version,
                api_key=self._api_key,
                source="workflow-execution",
                prompts=Sam2PromptSet(prompts=prompts),
                threshold=threshold,
                multimask_output=multimask_output,
            )
            sam_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="sam2",
            )

            sam2_segmentation_response = self._model_manager.infer_from_request_sync(
                sam_model_id, inference_request
            )

            prediction = convert_sam2_segmentation_response_to_inference_instances_seg_response(
                sam2_segmentation_predictions=sam2_segmentation_response.predictions,
                image=single_image,
                prompt_class_ids=prompt_class_ids,
                prompt_class_names=prompt_class_names,
                prompt_detection_ids=prompt_detection_ids,
                threshold=threshold,
            )
            predictions.append(prediction)

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


def convert_sam2_segmentation_response_to_inference_instances_seg_response(
    sam2_segmentation_predictions: List[Sam2SegmentationPrediction],
    image: WorkflowImageData,
    prompt_class_ids: List[Optional[int]],
    prompt_class_names: List[Optional[str]],
    prompt_detection_ids: List[Optional[str]],
    threshold: float,
) -> InstanceSegmentationInferenceResponse:
    image_width = image.numpy_image.shape[1]
    image_height = image.numpy_image.shape[0]
    predictions = []
    if len(prompt_class_ids) == 0:
        prompt_class_ids = [0 for _ in range(len(sam2_segmentation_predictions))]
        prompt_class_names = [
            "foreground" for _ in range(len(sam2_segmentation_predictions))
        ]
        prompt_detection_ids = [None for _ in range(len(sam2_segmentation_predictions))]
    for prediction, class_id, class_name, detection_id in zip(
        sam2_segmentation_predictions,
        prompt_class_ids,
        prompt_class_names,
        prompt_detection_ids,
    ):
        for mask in prediction.masks:
            if len(mask) < 3:
                # skipping empty masks
                continue
            if prediction.confidence < threshold:
                # skipping maks below threshold
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
