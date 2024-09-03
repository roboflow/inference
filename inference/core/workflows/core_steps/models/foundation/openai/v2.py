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
    BATCH_OF_DICTIONARY_KIND,
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_STRING_KIND,
    DICTIONARY_KIND,
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
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenAI",
            "version": "v2",
            "short_description": "Run OpenAI's GPT-4 with Vision",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "ChatGPT"],
        }
    )
    type: Literal["roboflow_core/open_ai@v2"]
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
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
                    "If not given - block defaults to value configured globally in Workflows Execution Engine. "
                    "Please restrict if you hit OpenAI limits."
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
            OutputDefinition(name="output", kind=[BATCH_OF_STRING_KIND]),
        ]

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
        image_detail: Literal["low", "high", "auto"],
        max_tokens: int,
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_output = run_gpt_4v_llm_prompting(
            image=inference_images,
            prompt=prompt,
            openai_api_key=openai_api_key,
            gpt_model_version=openai_model,
            gpt_image_detail=image_detail,
            max_tokens=max_tokens,
            max_concurrent_requests=max_concurrent_requests,
        )
        predictions = [
            {
                "raw_output": single_output["content"],
                "image": single_output["image"],
            }
            for single_output in raw_output
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
    gpt_model_version: str,
    gpt_image_detail: Literal["auto", "high", "low"],
    max_tokens: int,
    max_concurrent_requests: Optional[int],
) -> List[Dict[str, str]]:
    if openai_api_key is None:
        raise ValueError(
            "Step that involves GPT-4V prompting requires OpenAI API key which was not provided."
        )
    return execute_gpt_4v_requests(
        image=image,
        openai_api_key=openai_api_key,
        prompt=prompt,
        gpt_model_version=gpt_model_version,
        gpt_image_detail=gpt_image_detail,
        max_tokens=max_tokens,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_gpt_4v_requests(
    image: List[dict],
    openai_api_key: str,
    prompt: str,
    gpt_model_version: str,
    gpt_image_detail: Literal["auto", "high", "low"],
    max_tokens: int,
    max_concurrent_requests: Optional[int],
) -> List[Dict[str, str]]:
    client = OpenAI(api_key=openai_api_key)
    tasks = [
        partial(
            execute_gpt_4v_request,
            client=client,
            image=single_image,
            prompt=prompt,
            gpt_model_version=gpt_model_version,
            gpt_image_detail=gpt_image_detail,
            max_tokens=max_tokens,
        )
        for single_image in image
    ]
    max_workers = max_concurrent_requests or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def execute_gpt_4v_request(
    client: OpenAI,
    image: Dict[str, Any],
    prompt: str,
    gpt_model_version: str,
    gpt_image_detail: Literal["auto", "high", "low"],
    max_tokens: int,
) -> Dict[str, str]:
    loaded_image, _ = load_image(image)
    image_metadata = {"width": loaded_image.shape[1], "height": loaded_image.shape[0]}
    base64_image = base64.b64encode(encode_image_to_jpeg_bytes(loaded_image)).decode(
        "ascii"
    )
    response = client.chat.completions.create(
        model=gpt_model_version,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": gpt_image_detail,
                        },
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return {"content": response.choices[0].message.content, "image": image_metadata}
