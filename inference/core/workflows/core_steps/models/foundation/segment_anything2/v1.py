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
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    FLOAT_KIND,
    STRING_KIND,
    ImageInputField,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

T = TypeVar("T")
K = TypeVar("K")

DETECTIONS_CLASS_NAME_FIELD = "class_name"

LONG_DESCRIPTION = """
Run Segment Anything 2, a zero-shot instance segmentation model, on an image.

** Dedicated inference server required (GPU recomended) **

You can use pass in boxes/predictions from other models to Segment Anything 2 to use as prompts for the model.
If you pass in box detections from another model, the class names of the boxes will be forwarded to the predicted masks.  If using the model unprompted, the model will assign intengers as class names / ids.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Segment Anything 2 Model",
            "version": "v1",
            "short_description": "Segment Anything 2",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/segment_anything@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    boxes: Optional[
        StepOutputSelector(
            kind=[
                BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        description="Boxes (from other model predictions) to ground SAM2",
        examples=["$steps.object_detection_model.predictions"],
        default=None,
    )
    version: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        Literal["hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus"],
    ] = Field(
        default="hiera_tiny",
        description="Model to be used.  One of hiera_large, hiera_small, hiera_tiny, hiera_b_plus",
        examples=["hiera_large", "$inputs.openai_model"],
    )
    threshold: Union[
        WorkflowParameterSelector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.0, description="Threshold for predicted masks scores", examples=[0.3]
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


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
        boxes: Batch[sv.Detections],
        version: str,
        threshold: float,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images, boxes=boxes, version=version, threshold=threshold
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
    ) -> BlockResult:

        predictions = []
        if boxes is None:
            boxes = [None] * len(images)

        for single_image, boxes_for_image in zip(images, boxes):
            prompt_class_ids: List[Optional[int]] = []
            prompt_class_names: List[str] = []
            prompt_boxes_confidence: List[Optional[float]] = []

            prompts = []
            if boxes_for_image is not None:
                for xyxy, _, confidence, class_id, _, bbox_data in boxes_for_image:
                    x1, y1, x2, y2 = xyxy
                    prompt_class_ids.append(class_id)
                    prompt_class_names.append(bbox_data[DETECTIONS_CLASS_NAME_FIELD])
                    prompt_boxes_confidence.append(confidence)
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
                prompt_boxes_confidence=prompt_boxes_confidence,
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
    prompt_boxes_confidence: List[Optional[float]],
):
    image_width = image.numpy_image.shape[1]
    image_height = image.numpy_image.shape[0]
    predictions = []
    prediction_id = 0
    if len(prompt_class_ids) == 0:
        prompt_class_ids = [i for i in range(len(sam2_segmentation_predictions))]
        prompt_class_names = [str(i) for i in range(len(sam2_segmentation_predictions))]
        prompt_boxes_confidence = [
            1.0 for _ in range(len(sam2_segmentation_predictions))
        ]

    for pred in sam2_segmentation_predictions:
        mask = pred.mask
        for polygon in mask:
            # for some reason this list of points contains empty array elements
            x_coords = [coord[0] for coord in polygon]
            y_coords = [coord[1] for coord in polygon]

            # Calculate min and max values
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)

            # Calculate center coordinates
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            class_id = safe_list_get(items=prompt_class_ids, index=prediction_id)
            if class_id is None:
                # id of unknown class
                class_id = -1
            predictions.append(
                InstanceSegmentationPrediction(
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                        "points": [Point(x=point[0], y=point[1]) for point in polygon],
                        "confidence": safe_list_get(
                            items=prompt_boxes_confidence, index=prediction_id
                        ),
                        "class": safe_list_get(
                            items=prompt_class_names,
                            index=prediction_id,
                            default="unknown",
                        ),
                        "class_id": class_id,
                    }
                )
            )
        prediction_id += 1

    return InstanceSegmentationInferenceResponse(
        predictions=predictions,
        image=InferenceResponseImage(width=image_width, height=image_height),
    )


def safe_list_get(items: List[T], index: int, default: K = None) -> Union[T, K]:
    if len(items) <= index or index < 0:
        return default
    return items[index]
