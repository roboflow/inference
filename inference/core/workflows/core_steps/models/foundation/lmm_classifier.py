from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import AliasChoices, ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.utils import (
    attach_parent_info,
    attach_prediction_type_info,
)
from inference.core.workflows.core_steps.models.foundation.lmm import (
    GPT_4V_MODEL_TYPE,
    LMMConfig,
    get_cogvlm_generations_from_remote_api,
    get_cogvlm_generations_locally,
    run_gpt_4v_llm_prompting,
    turn_raw_lmm_output_into_structured,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    BATCH_OF_STRING_KIND,
    BATCH_OF_TOP_CLASS_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    FlowControl,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
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
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
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
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="raw_output", kind=[BATCH_OF_STRING_KIND]),
            OutputDefinition(name="top", kind=[BATCH_OF_TOP_CLASS_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
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
    ):
        self._model_manager = model_manager
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: List[dict],
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        prompt = (
            f"You are supposed to perform image classification task. You are given image that should be "
            f"assigned one of the following classes: {classes}. "
            f'Your response must be JSON in format: {{"top": "some_class"}}'
        )
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = await run_gpt_4v_llm_prompting(
                image=images,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raw_output = await get_cogvlm_generations_locally(
                image=images,
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
        predictions = attach_parent_info(
            images=images,
            predictions=predictions,
            nested_key=None,
        )
        predictions = attach_prediction_type_info(
            predictions=predictions,
            prediction_type="classification",
        )
        return attach_parent_info(
            images=images,
            predictions=predictions,
            nested_key=None,
        )

    async def run_remotely(
        self,
        images: List[dict],
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        prompt = (
            f"You are supposed to   image classification task. You are given image that should be "
            f"assigned one of the following classes: {classes}. "
            f'Your response must be JSON in format: {{"top": "some_class"}}'
        )
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = await run_gpt_4v_llm_prompting(
                image=images,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raw_output = await get_cogvlm_generations_from_remote_api(
                image=images,
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
        predictions = attach_parent_info(
            images=images,
            predictions=predictions,
            nested_key=None,
        )
        predictions = attach_prediction_type_info(
            predictions=predictions,
            prediction_type="classification",
        )
        return attach_parent_info(
            images=images,
            predictions=predictions,
            nested_key=None,
        )
