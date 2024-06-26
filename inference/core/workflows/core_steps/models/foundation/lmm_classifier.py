from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.core.workflows.constants import (
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.lmm import (
    GPT_4V_MODEL_TYPE,
    LMMConfig,
    get_cogvlm_generations_from_remote_api,
    get_cogvlm_generations_locally,
    run_gpt_4v_llm_prompting,
    turn_raw_lmm_output_into_structured,
)
from inference.core.workflows.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    BATCH_OF_STRING_KIND,
    BATCH_OF_TOP_CLASS_KIND,
    LIST_OF_VALUES_KIND,
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
Classify an image into one or more categories using a Large Multimodal Model (LMM).

You can specify arbitrary classes to an LMMBlock.

The LLMBlock supports two LMMs:

- OpenAI's GPT-4 with Vision, and;
- CogVLM.

You need to provide your OpenAI API key to use the GPT-4 with Vision model. You do not 
need to provide an API key to use CogVLM.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Run a large language model for classification.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["LMMForClassification"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    lmm_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
    )
    classes: Union[List[str], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = (
        Field(
            description="List of classes that LMM shall classify against",
            examples=[["a", "b"], "$inputs.classes"],
        )
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(), description="Configuration of LMM"
    )
    remote_api_key: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and do not require additional API key for CogVLM calls.",
        examples=["xxx-xxx", "$inputs.api_key"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="raw_output", kind=[BATCH_OF_STRING_KIND]),
            OutputDefinition(name="top", kind=[BATCH_OF_TOP_CLASS_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
        ]


class LMMForClassificationBlock(WorkflowBlock):

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
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return await self.run_locally(
                images=images,
                lmm_type=lmm_type,
                classes=classes,
                lmm_config=lmm_config,
                remote_api_key=remote_api_key,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return await self.run_remotely(
                images=images,
                lmm_type=lmm_type,
                classes=classes,
                lmm_config=lmm_config,
                remote_api_key=remote_api_key,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    async def run_locally(
        self,
        images: Batch[WorkflowImageData],
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> BlockResult:
        prompt = (
            f"You are supposed to perform image classification task. You are given image that should be "
            f"assigned one of the following classes: {classes}. "
            f'Your response must be JSON in format: {{"top": "some_class"}}'
        )
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = await run_gpt_4v_llm_prompting(
                image=images_prepared_for_processing,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raw_output = await get_cogvlm_generations_locally(
                image=images_prepared_for_processing,
                prompt=prompt,
                model_manager=self._model_manager,
                api_key=self._api_key,
            )
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output={"top": "name of the class"},
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "top": structured["top"],
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "classification"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions

    async def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> BlockResult:
        prompt = (
            f"You are supposed to   image classification task. You are given image that should be "
            f"assigned one of the following classes: {classes}. "
            f'Your response must be JSON in format: {{"top": "some_class"}}'
        )
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = await run_gpt_4v_llm_prompting(
                image=images_prepared_for_processing,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raw_output = await get_cogvlm_generations_from_remote_api(
                image=images_prepared_for_processing,
                prompt=prompt,
                api_key=self._api_key,
            )
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output={"top": "name of the class"},
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "top": structured["top"],
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "classification"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions
