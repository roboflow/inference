from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.entities.requests.sam2 import Sam2InferenceRequest


from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
    load_core_model
)
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
    type: Literal["SegmentAnything2Model" ]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    sam2_model: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus"]
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
            inference_request = Sam2InferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                sam2_version_id=sam2_model,
                api_key=self._api_key,
                source="workflow-execution"
            )
            sam_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="sam2",
            )
            prediction = await self._model_manager.infer_from_request(
                sam_model_id, inference_request
            )
            predictions.append(prediction.model_dump(by_alias=True, exclude_none=True))
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )


    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        
    ) -> BlockResult:
        print("POST PROCESSING", images, predictions)
        # predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        # predictions = attach_prediction_type_info_to_sv_detections_batch(
        #     predictions=predictions,
        #     prediction_type="instance-segmentation",
        # )
        # predictions = filter_out_unwanted_classes_from_sv_detections_batch(
        #     predictions=predictions,
        #     classes_to_accept=class_filter,
        # )
        # predictions = attach_parents_coordinates_to_batch_of_sv_detections(
        #     images=images,
        #     predictions=predictions,
        # )
        return [{"predictions": prediction} for prediction in predictions]
