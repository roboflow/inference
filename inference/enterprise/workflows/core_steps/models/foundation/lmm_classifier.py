from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.core_steps.common.utils import (
    attach_parent_info,
    attach_prediction_type_info,
)
from inference.enterprise.workflows.core_steps.models.foundation.lmm import (
    GPT_4V_MODEL_TYPE,
    LMMConfig,
    get_cogvlm_generations_from_remote_api,
    get_cogvlm_generations_locally,
    run_gpt_4v_llm_prompting,
    try_parse_lmm_output_to_json,
)
from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BATCH_OF_STRING_KIND,
    IMAGE_METADATA_KIND,
    LIST_OF_VALUES_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    STRING_KIND,
    TOP_CLASS_KIND,
    FlowControl,
    InferenceImageSelector,
    InferenceParameterSelector,
    OutputStepImageSelector,
)
from inference.enterprise.workflows.prototypes.block import (
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

_If you want to ask an LMM an arbitrary question about an image, we recommend using the 
dedicated LMMForClassificationBlock._
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
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    lmm_type: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
    )
    classes: Union[
        List[str], InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        description="List of classes that LMM shall classify against",
        examples=[["a", "b"], "$inputs.classes"],
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(), description="Configuration of LMM"
    )
    remote_api_key: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and do not require additional API key for CogVLM calls.",
        examples=["xxx-xxx", "$inputs.api_key"],
    )


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
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="raw_output", kind=[BATCH_OF_STRING_KIND]),
            OutputDefinition(name="top", kind=[TOP_CLASS_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
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
            raw_output, structured_output = await run_gpt_4v_llm_prompting(
                image=image,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
                expected_output={"top": "name of the class"},
            )
        else:
            raw_output = await get_cogvlm_generations_locally(
                image=image,
                prompt=prompt,
                model_manager=self._model_manager,
                api_key=self._api_key,
            )
            structured_output = [
                try_parse_lmm_output_to_json(
                    output=r["content"],
                    expected_output={"top": "name of the class"},
                )
                for r in raw_output
            ]
        serialised_result = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "top": structured["top"],
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        serialised_result = attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )
        serialised_result = attach_prediction_type_info(
            results=serialised_result,
            prediction_type="classification",
        )
        return attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )

    async def run_remotely(
        self,
        image: List[dict],
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
            raw_output, structured_output = await run_gpt_4v_llm_prompting(
                image=image,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
                expected_output={"top": "name of the class"},
            )
        else:
            raw_output = await get_cogvlm_generations_from_remote_api(
                image=image,
                prompt=prompt,
                api_key=self._api_key,
            )
            structured_output = [
                try_parse_lmm_output_to_json(
                    output=r["content"],
                    expected_output={"top": "name of the class"},
                )
                for r in raw_output
            ]
        serialised_result = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "top": structured["top"],
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        serialised_result = attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )
        serialised_result = attach_prediction_type_info(
            results=serialised_result,
            prediction_type="classification",
        )
        return attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )
