import asyncio
import base64
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.env import (
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.enterprise.workflows.complier.steps_executors.models import (
    load_core_model,
)
from inference.enterprise.workflows.complier.steps_executors.utils import make_batches
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    DICTIONARY_KIND,
    IMAGE_METADATA_KIND,
    PARENT_ID_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    FlowControl,
    InferenceImageSelector,
    InferenceParameterSelector,
    OutputStepImageSelector,
)
from inference.enterprise.workflows.errors import ExecutionGraphError
from inference.enterprise.workflows.steps.common.utils import attach_parent_info
from inference_sdk import InferenceHTTPClient

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


class BlockManifest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Block that make it possible to use chosen LMM model within `workflows` - with arbitrary prompt and possibility to retrieve structured JSON output.",
            "docs": "https://inference.roboflow.com/workflows/use_lmm/",
            "block_type": "model",
        }
    )
    type: Literal["LMM"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    prompt: Union[InferenceParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Holds unconstrained text prompt to LMM mode",
        examples=["my prompt", "$inputs.prompt"],
    )
    lmm_type: Union[
        InferenceParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
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
    json_output: Optional[
        Union[InferenceParameterSelector(kind=[DICTIONARY_KIND]), Dict[str, str]]
    ] = Field(
        default=None,
        description="Holds dictionary that maps name of requested output field into its description",
        examples=[{"count": "number of cats in the picture"}, "$inputs.json_output"],
    )


class LMMBlock:

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
    def get_input_manifest(cls) -> Type[BaseModel]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
            OutputDefinition(name="*", kind=[WILDCARD_KIND]),
        ]

    def get_actual_outputs(self, manifest: BlockManifest) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
        ]
        if manifest.json_output is None:
            return result
        for key in manifest.json_output.keys():
            result.append(OutputDefinition(name=key, kind=[WILDCARD_KIND]))
        return result

    async def run_locally(
        self,
        image: List[dict],
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if json_output is not None:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output, indent=4)}"
            )
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output, structured_output = await run_gpt_4v_llm_prompting(
                image=image,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
                expected_output=json_output,
            )
        else:
            raw_output = await get_cogvlm_generations_locally(
                image=image,
                prompt=prompt,
                model_manager=self._model_manager,
                api_key=self._api_key,
            )
            if json_output is None:
                structured_output = [{} for _ in range(len(raw_output))]
            else:
                structured_output = [
                    try_parse_lmm_output_to_json(
                        output=r["content"],
                        expected_output=json_output,
                    )
                    for r in raw_output
                ]
        serialised_result = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "structured_output": structured,
                **structured,
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        return attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )

    async def run_remotely(
        self,
        image: List[dict],
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if json_output is not None:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output, indent=4)}"
            )
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output, structured_output = await run_gpt_4v_llm_prompting(
                image=image,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
                expected_output=json_output,
            )
        else:
            raw_output = await get_cogvlm_generations_from_remote_api(
                image=image,
                prompt=prompt,
                api_key=self._api_key,
            )
            if json_output is None:
                structured_output = [{} for _ in range(len(raw_output))]
            else:
                structured_output = [
                    try_parse_lmm_output_to_json(
                        output=r["content"],
                        expected_output=json_output,
                    )
                    for r in raw_output
                ]
        serialised_result = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "structured_output": structured,
                **structured,
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        return attach_parent_info(
            image=image,
            results=serialised_result,
            nested_key=None,
        )


async def run_gpt_4v_llm_prompting(
    image: List[Dict[str, Any]],
    prompt: str,
    remote_api_key: Optional[str],
    lmm_config: LMMConfig,
    expected_output: Optional[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[dict]]:
    if remote_api_key is None:
        raise ExecutionGraphError(
            f"Step that involves GPT-4V prompting requires OpenAI API key which was not provided."
        )
    results = await execute_gpt_4v_requests(
        image=image,
        remote_api_key=remote_api_key,
        prompt=prompt,
        lmm_config=lmm_config,
    )
    if expected_output is None:
        return results, [{} for _ in range(len(results))]
    parsed_output = [
        try_parse_lmm_output_to_json(
            output=r["content"], expected_output=expected_output
        )
        for r in results
    ]
    return results, parsed_output


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
        )
        model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="cogvlm",
            api_key=api_key,
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
        raise ExecutionGraphError(
            f"Chosen remote execution of CogVLM model in Roboflow Hosted API mode, but remote execution "
            f"is only possible for self-hosted option."
        )
    client = InferenceHTTPClient.init(
        api_url=LOCAL_INFERENCE_API_URL,
        api_key=api_key,
    )
    results = []
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
        results.extend(
            [
                {"content": br["response"], "image": bm}
                for br, bm in zip(batch_results, batch_image_metadata)
            ]
        )
    return results


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
