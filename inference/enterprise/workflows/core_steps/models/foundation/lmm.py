import asyncio
import base64
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from openai import AsyncOpenAI
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.env import (
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.enterprise.workflows.core_steps.common.utils import (
    attach_parent_info,
    load_core_model,
)
from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BATCH_OF_DICTIONARY_KIND,
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_STRING_KIND,
    DICTIONARY_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    FlowControl,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches

GPT_4V_MODEL_TYPE = "gpt_4v"
COG_VLM_MODEL_TYPE = "cog_vlm"
NOT_DETECTED_VALUE = "not_detected"

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")


class LMMConfig(BaseModel):
    max_tokens: int = Field(default=450)
    gpt_image_detail: Literal["low", "high", "auto"] = Field(
        default="auto",
        description="To be used for GPT-4V only.",
    )
    gpt_model_version: str = Field(default="gpt-4-vision-preview")


LONG_DESCRIPTION = """
Ask a question to a Large Multimodal Model (LMM) with an image and text.

You can specify arbitrary text prompts to an LMMBlock.

The LLMBlock supports two LMMs:

- OpenAI's GPT-4 with Vision, and;
- CogVLM.

You need to provide your OpenAI API key to use the GPT-4 with Vision model. You do not 
need to provide an API key to use CogVLM.

_If you want to classify an image into one or more categories, we recommend using the 
dedicated LMMForClassificationBlock._
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Run a large language model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["LMM"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    prompt: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Holds unconstrained text prompt to LMM mode",
        examples=["my prompt", "$inputs.prompt"],
    )
    lmm_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
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
    json_output: Optional[Dict[str, str]] = Field(
        default=None,
        description="Holds dictionary that maps name of requested output field into its description",
        examples=[{"count": "number of cats in the picture"}, "$inputs.json_output"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[BATCH_OF_DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[BATCH_OF_STRING_KIND]),
            OutputDefinition(name="*", kind=[WILDCARD_KIND]),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
        ]
        if self.json_output is None:
            return result
        for key in self.json_output.keys():
            result.append(OutputDefinition(name=key, kind=[WILDCARD_KIND]))
        return result


class LMMBlock(WorkflowBlock):

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
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if json_output:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output, indent=4)}"
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
            expected_output=json_output,
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "structured_output": structured,
                **structured,
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        return attach_parent_info(
            images=images,
            predictions=predictions,
            nested_key=None,
        )

    async def run_remotely(
        self,
        images: List[dict],
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if json_output:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output, indent=4)}"
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
            expected_output=json_output,
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "structured_output": structured,
                **structured,
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        return attach_parent_info(
            images=images,
            predictions=predictions,
            nested_key=None,
        )


async def run_gpt_4v_llm_prompting(
    image: List[Dict[str, Any]],
    prompt: str,
    remote_api_key: Optional[str],
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    if remote_api_key is None:
        raise ValueError(
            "Step that involves GPT-4V prompting requires OpenAI API key which was not provided."
        )
    return await execute_gpt_4v_requests(
        image=image,
        remote_api_key=remote_api_key,
        prompt=prompt,
        lmm_config=lmm_config,
    )


async def execute_gpt_4v_requests(
    image: List[dict],
    remote_api_key: str,
    prompt: str,
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    client = AsyncOpenAI(api_key=remote_api_key)
    results = []
    images_batches = list(
        make_batches(
            iterable=image,
            batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
    )
    for image_batch in images_batches:
        batch_coroutines = []
        for image in image_batch:
            coroutine = execute_gpt_4v_request(
                client=client,
                image=image,
                prompt=prompt,
                lmm_config=lmm_config,
            )
            batch_coroutines.append(coroutine)
        batch_results = await asyncio.gather(*batch_coroutines)
        results.extend(batch_results)
    return results


async def execute_gpt_4v_request(
    client: AsyncOpenAI,
    image: Dict[str, Any],
    prompt: str,
    lmm_config: LMMConfig,
) -> Dict[str, str]:
    loaded_image, _ = load_image(image)
    image_metadata = {"width": loaded_image.shape[1], "height": loaded_image.shape[0]}
    base64_image = base64.b64encode(encode_image_to_jpeg_bytes(loaded_image)).decode(
        "ascii"
    )
    response = await client.chat.completions.create(
        model=lmm_config.gpt_model_version,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": lmm_config.gpt_image_detail,
                        },
                    },
                ],
            }
        ],
        max_tokens=lmm_config.max_tokens,
    )
    return {"content": response.choices[0].message.content, "image": image_metadata}


async def get_cogvlm_generations_locally(
    image: List[dict],
    prompt: str,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    serialised_result = []
    for single_image in image:
        loaded_image, _ = load_image(single_image)
        image_metadata = {
            "width": loaded_image.shape[1],
            "height": loaded_image.shape[0],
        }
        inference_request = CogVLMInferenceRequest(
            image=single_image,
            prompt=prompt,
            api_key=api_key,
        )
        model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="cogvlm",
        )
        result = await model_manager.infer_from_request(model_id, inference_request)
        serialised_result.append(
            {
                "content": result.response,
                "image": image_metadata,
            }
        )
    return serialised_result


async def get_cogvlm_generations_from_remote_api(
    image: List[dict],
    prompt: str,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        raise ValueError(
            f"Chosen remote execution of CogVLM model in Roboflow Hosted API mode, but remote execution "
            f"is only possible for self-hosted option."
        )
    client = InferenceHTTPClient.init(
        api_url=LOCAL_INFERENCE_API_URL,
        api_key=api_key,
    )
    raw_output = []
    images_batches = list(
        make_batches(
            iterable=image,
            batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
    )
    for image_batch in images_batches:
        batch_coroutines, batch_image_metadata = [], []
        for image in image_batch:
            loaded_image, _ = load_image(image)
            image_metadata = {
                "width": loaded_image.shape[1],
                "height": loaded_image.shape[0],
            }
            batch_image_metadata.append(image_metadata)
            coroutine = client.prompt_cogvlm_async(
                visual_prompt=image["value"],
                text_prompt=prompt,
            )
            batch_coroutines.append(coroutine)
        batch_results = await asyncio.gather(*batch_coroutines)
        raw_output.extend(
            [
                {"content": br["response"], "image": bm}
                for br, bm in zip(batch_results, batch_image_metadata)
            ]
        )
    return raw_output


def turn_raw_lmm_output_into_structured(
    raw_output: List[Dict[str, Any]],
    expected_output: Optional[Dict[str, str]],
) -> List[dict]:
    if expected_output is None:
        return [{} for _ in range(len(raw_output))]
    return [
        try_parse_lmm_output_to_json(
            output=r["content"],
            expected_output=expected_output,
        )
        for r in raw_output
    ]


def try_parse_lmm_output_to_json(
    output: str, expected_output: Dict[str, str]
) -> Union[list, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(output)
    if len(json_blocks_found) == 0:
        return try_parse_json(output, expected_output=expected_output)
    result = []
    for json_block in json_blocks_found:
        result.append(
            try_parse_json(content=json_block, expected_output=expected_output)
        )
    return result if len(result) > 1 else result[0]


def try_parse_json(content: str, expected_output: Dict[str, str]) -> dict:
    try:
        data = json.loads(content)
        return {key: data.get(key, NOT_DETECTED_VALUE) for key in expected_output}
    except Exception:
        return {key: NOT_DETECTED_VALUE for key in expected_output}
