import json
from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import load_image
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_DICTIONARY_KIND,
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    WILDCARD_KIND,
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

TASKS_WITH_PROMPT = [
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>" "<REGION_TO_DESCRIPTION>",
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Florence-2 Model",
            "short_description": "Run a multitask transformer model for a wide range of computer vision tasks.",
            "long_description": "Florence-2 is a multitask transformer model that can be used for a wide range of computer vision tasks. It is based on the Vision Transformer architecture and has been trained on a large-scale dataset of images with a wide range of labels. The model is capable of performing tasks such as image classification, object detection, and image segmentation.",
            "license": "MIT",
            "block_type": "model",
        }
    )
    type: Literal["Florence2Model", "Florence2"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            title="Model",
            default="florence-pretrains/1",
            description="Roboflow model identifier",
            examples=["florence-pretrains/1", "$inputs.model"],
        )
    )
    vision_task: Union[
        Literal[
            "<OD>",
            "<CAPTION_TO_PHRASE_GROUNDING>",
            "<DENSE_REGION_CAPTION>",
            "<REGION_PROPOSAL>",
            "<OCR_WITH_REGION>",
            "<REFERRING_EXPRESSION_SEGMENTATION>",
            "<REGION_TO_SEGMENTATION>",
            "<OPEN_VOCABULARY_DETECTION>",
            "<REGION_TO_CATEGORY>",
            "<REGION_TO_DESCRIPTION>",
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        description="The computer vision task to perform.",
        default="<OPEN_VOCABULARY_DETECTION>",
        examples=["<OPEN_VOCABULARY_DETECTION>"],
    )
    prompt: Union[WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]] = (
        Field(
            description="The accompanying prompt for the task (comma separated).",
            examples=[["red apple", "blue soda can"], "$inputs.prompt"],
        )
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="raw_output", kind=[BATCH_OF_DICTIONARY_KIND]),
            OutputDefinition(name="structured_output", kind=[WILDCARD_KIND]),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="raw_output", kind=[BATCH_OF_DICTIONARY_KIND]),
            OutputDefinition(name="structured_output", kind=[WILDCARD_KIND]),
        ]
        return result


class Florence2ModelBlock(WorkflowBlock):

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
        vision_task: str,
        prompt: List[str],
        model_id: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return await self.run_locally(
                images=images, vision_task=vision_task, prompt=prompt, model_id=model_id
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise ValueError(
                f"Invalid step execution mode: {self._step_execution_mode}; Florence2ModelBlock only supports local execution."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    async def run_locally(
        self,
        images: Batch[WorkflowImageData],
        vision_task: str,
        prompt: List[str],
        model_id: str,
    ) -> BlockResult:
        predictions = []
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]

        # infer on florence2 model
        predictions = await self.get_florence2_generations_locally(
            image=images_prepared_for_processing,
            prompt=(
                vision_task
                if vision_task not in TASKS_WITH_PROMPT
                else vision_task + " " + "<and>".join(prompt)
            ),
            model_manager=self._model_manager,
            api_key=self._api_key,
            model_id=model_id,
        )

        # convert to sv detections
        for prediction in predictions:
            prediction["structured_output"] = sv.Detections.from_lmm(
                sv.LMM.FLORENCE_2,
                prediction["raw_output"],
                resolution_wh=(
                    prediction["image"]["width"],
                    prediction["image"]["height"],
                ),
            )

        formatted_predictions = [
            {
                **pred,
                "parent_id": image.parent_metadata.parent_id,
                "root_parent_id": image.workflow_root_ancestor_metadata.parent_id,
            }
            for pred, image in zip(predictions, images)
        ]

        return formatted_predictions

    async def get_florence2_generations_locally(
        self,
        image: List[dict],
        prompt: str,
        model_manager: ModelManager,
        api_key: Optional[str],
        model_id: str,
    ) -> List[Dict[str, Any]]:
        serialised_result = []

        # run florence 2 on each image
        for single_image in image:
            loaded_image, _ = load_image(single_image)
            image_metadata = {
                "width": loaded_image.shape[1],
                "height": loaded_image.shape[0],
            }
            inference_request = LMMInferenceRequest(
                model_id=model_id,
                image=single_image,
                prompt=prompt,
                api_key=api_key,
            )
            model_manager.add_model(model_id, api_key=api_key)
            result = await model_manager.infer_from_request(model_id, inference_request)
            serialised_result.append(
                {
                    "raw_output": json.loads(result.response),
                    "image": image_metadata,
                }
            )
        return serialised_result
