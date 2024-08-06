from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, PositiveInt

import numpy as np

from inference.core.entities.requests.sam2 import Sam2SegmentationRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)

from inference.core.utils.postprocess import masks2poly

from inference.core.workflows.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    STRING_KIND,
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Run Segment Anything 2 Model
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Segment Anything 2 Model",
            "short_description": "Segment Anything 2",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["SegmentAnything2Model"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    sam2_model: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        Literal["hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus"],
    ] = Field(
        default="hiera_large",
        description="Model to be used.  One of hiera_large, hiera_small, hiera_tiny, hiera_b_plus",
        examples=["hiera_large", "$inputs.openai_model"],
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


class SegmentAnything2Block(WorkflowBlock):

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

    async def run(
        self,
        images: Batch[WorkflowImageData],
        sam2_model: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return await self.run_locally(
                images=images,
                sam2_model=sam2_model,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Segment Anything."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    async def run_locally(
        self,
        images: Batch[WorkflowImageData],
        sam2_model: str,
    ) -> BlockResult:

        predictions = []
        for single_image in images:
            inference_request = Sam2SegmentationRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                sam2_version_id=sam2_model,
                api_key=self._api_key,
                source="workflow-execution",
            )
            sam_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="sam2",
            )

            sam2_segmentation_response = await self._model_manager.infer_from_request(
                sam_model_id, inference_request
            )

            print("SAM2 response:", sam2_segmentation_response)
            prediction = self._convert_sam2_segmentation_response_to_inference_instances_seg_response(sam2_segmentation_response, single_image)

            predictions.append(prediction)
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )
    

    def _convert_sam2_segmentation_response_to_inference_instances_seg_response(self, sample2_segmentation_response, image):

        image_width = image.numpy_image.shape[1],
        image_height = image.numpy_image.shape[0],
        predictions = []

        for raw_mask in sample2_segmentation_response.masks:
            mask = raw_mask

            x_coords = mask[::2]
            y_coords = mask[1::2]

            # Calculate min and max values
            min_x = np.min(x_coords)
            min_y = np.min(y_coords)
            max_x = np.max(x_coords)
            max_y = np.max(y_coords)

            # Calculate center coordinates
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            predictions.append(
                InstanceSegmentationPrediction(
                    **{
                        "x": center_x,
                        "y": center_y,
                        "width": max_x - min_x,
                        "height":  max_y - min_y,
                        "points": [Point(x=point[0], y=point[1]) for point in mask],
                        "confidence": 0.5, #TODO: get confidence from model
                        "class": 'mask', #TODO: sam doesnt really have a class, so we are just using mask for now
                        "class_id": 1,
                    }
                )
            )

        return  InstanceSegmentationInferenceResponse(
                predictions=predictions,
                image=InferenceResponseImage(
                    width=image_width, height=image_height
                ),
            )

        


    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        print("POST PROCESSING", images)

        
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
