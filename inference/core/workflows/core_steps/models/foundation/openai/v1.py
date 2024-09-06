import base64
import json
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.utils import run_in_parallel
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_METADATA_KIND,
    PARENT_ID_KIND,
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
from inference_sdk.http.utils.iterables import make_batches

NOT_DETECTED_VALUE = "not_detected"
JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")


class LMMConfig(BaseModel):
    max_tokens: int = Field(default=450)
    gpt_image_detail: Literal["low", "high", "auto"] = Field(default="auto")
    gpt_model_version: str = Field(default="gpt-4o")


LONG_DESCRIPTION = """
Ask a question to OpenAI's GPT-4 with Vision model.

You can specify arbitrary text prompts to the OpenAIBlock.

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 

_This model was previously part of the LMM block._
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenAI",
            "version": "v1",
            "short_description": "Run OpenAI's GPT-4 with Vision",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "ChatGPT"],
        }
    )
    type: Literal["roboflow_core/open_ai@v1", "OpenAI"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    prompt: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Text prompt to the OpenAI model",
        examples=["my prompt", "$inputs.prompt"],
    )
    openai_api_key: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        description="Your OpenAI API key",
        examples=["xxx-xxx", "$inputs.openai_api_key"],
        private=True,
    )
    openai_model: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["gpt-4o", "gpt-4o-mini"]
    ] = Field(
        default="gpt-4o",
        description="Model to be used",
        examples=["gpt-4o", "$inputs.openai_model"],
    )
    json_output_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Holds dictionary that maps name of requested output field into its description",
        examples=[
            {"count": "number of cats in the picture"},
            "$inputs.json_output_format",
        ],
    )
    image_detail: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["auto", "high", "low"]
    ] = Field(
        default="auto",
        description="Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity.",
        examples=["auto", "high", "low"],
    )
    max_tokens: int = Field(
        default=450,
        description="Maximum number of tokens the model can generate in it's response.",
        examples=[450],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
            OutputDefinition(name="*", kind=[WILDCARD_KIND]),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
        ]
        if self.json_output_format is None:
            return result
        for key in self.json_output_format.keys():
            result.append(OutputDefinition(name=key, kind=[WILDCARD_KIND]))
        return result

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class OpenAIBlockV1(WorkflowBlock):

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

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        openai_api_key: str,
        openai_model: Optional[str],
        json_output_format: Optional[Dict[str, str]],
        image_detail: Literal["low", "high", "auto"],
        max_tokens: int,
    ) -> BlockResult:
        if json_output_format:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output_format, indent=4)}"
            )
        inference_images = [i.to_inference_format() for i in images]
        raw_output = run_gpt_4v_llm_prompting(
            image=inference_images,
            prompt=prompt,
            openai_api_key=openai_api_key,
            lmm_config=LMMConfig(
                gpt_model_version=openai_model,
                gpt_image_detail=image_detail,
                max_tokens=max_tokens,
            ),
        )
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output=json_output_format,
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
        for prediction, image in zip(predictions, images):
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions


def run_gpt_4v_llm_prompting(
    image: List[Dict[str, Any]],
    prompt: str,
    openai_api_key: Optional[str],
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    if openai_api_key is None:
        raise ValueError(
            "Step that involves GPT-4V prompting requires OpenAI API key which was not provided."
        )
    return execute_gpt_4v_requests(
        image=image,
        openai_api_key=openai_api_key,
        prompt=prompt,
        lmm_config=lmm_config,
    )


def execute_gpt_4v_requests(
    image: List[dict],
    openai_api_key: str,
    prompt: str,
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    client = OpenAI(api_key=openai_api_key)
    tasks = [
        partial(
            execute_gpt_4v_request,
            client=client,
            image=single_image,
            prompt=prompt,
            lmm_config=lmm_config,
        )
        for single_image in image
    ]
    return run_in_parallel(
        tasks=tasks,
        max_workers=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )


def execute_gpt_4v_request(
    client: OpenAI,
    image: Dict[str, Any],
    prompt: str,
    lmm_config: LMMConfig,
) -> Dict[str, str]:
    loaded_image, _ = load_image(image)
    image_metadata = {"width": loaded_image.shape[1], "height": loaded_image.shape[0]}
    base64_image = base64.b64encode(encode_image_to_jpeg_bytes(loaded_image)).decode(
        "ascii"
    )
    response = client.chat.completions.create(
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
