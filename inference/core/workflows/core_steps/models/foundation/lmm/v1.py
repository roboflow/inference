import base64
import json
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from inference.core.env import (
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    load_core_model,
    run_in_parallel,
)
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
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
    PARENT_ID_KIND,
    SECRET_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

GPT_4V_MODEL_TYPE = "gpt_4v"
NOT_DETECTED_VALUE = "not_detected"

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")


class LMMConfig(BaseModel):
    max_tokens: int = Field(default=450)
    gpt_image_detail: Literal["low", "high", "auto"] = Field(
        default="auto",
        description="To be used for GPT-4V only.",
    )
    gpt_model_version: str = Field(default="gpt-4o")


LONG_DESCRIPTION = """
**⚠️ This block is deprecated.** Use the OpenAI GPT-4 Vision blocks (v1-v4) instead for better functionality and ongoing support. This block allows asking questions to OpenAI's GPT-4 with Vision model using images and text prompts, with optional structured JSON output parsing.

## How This Block Works

This deprecated block uses OpenAI's GPT-4 with Vision model to answer questions about images using natural language prompts. The block:

1. Takes images and a text prompt as input (supports batch processing)
2. Encodes images to base64 format for API transmission
3. Sends the image and prompt to OpenAI's GPT-4 Vision API
4. Receives the model's text response
5. Optionally parses structured JSON output if a JSON schema is specified
6. Returns both raw text output and structured output (if JSON schema was provided)

The block supports arbitrary text prompts, allowing you to ask questions, request descriptions, or instruct the model to analyze images in any way. You can optionally provide a JSON output schema to parse structured data from the model's response, which is useful for extracting specific information in a consistent format.

## Common Use Cases

- **Image Analysis and Description**: Ask questions about image content, scene understanding, or request detailed descriptions of what's in an image
- **Structured Data Extraction**: Extract specific information from images using JSON output schemas (e.g., count objects, identify attributes, extract structured data)
- **Visual Question Answering**: Answer specific questions about image content (e.g., "What is the weather in this image?", "How many people are present?")
- **Content Understanding**: Analyze complex scenes, understand context, or interpret visual information that requires reasoning
- **Custom Vision Tasks**: Perform custom vision tasks with flexible prompts when predefined task types don't fit your needs

## Connecting to Other Blocks

The outputs from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on the text responses or extracted structured data
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log analysis results or extracted structured data
- **Expression blocks** to parse, validate, or transform the text or structured outputs
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts based on the analysis results
- **Webhook blocks** to send analysis results to external systems or APIs

## Deprecation Notice

This block is deprecated. For new workflows, use the **OpenAI GPT-4 Vision blocks (v1-v4)** which provide:

- More comprehensive task type support (OCR, classification, captioning, VQA, structured answering)
- Better prompt handling and structured outputs
- Ongoing updates and support
- Additional model versions and configuration options

If you need image classification specifically, use the **LMM For Classification** block instead.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "LMM",
            "version": "v1",
            "short_description": "Run a large multimodal model such as ChatGPT-4v.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "deprecated": True,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
            },
        }
    )
    type: Literal["roboflow_core/lmm@v1", "LMM"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Text prompt to send to the GPT-4 Vision model. Can be any question, instruction, or request about the image. The model will respond based on this prompt. Be specific and clear for best results (e.g., 'Describe what you see in this image', 'Count the number of cars', 'What is the main subject of this photo?'). If json_output is specified, include instructions in the prompt about what information to extract.",
        examples=[
            "Describe what you see in this image",
            "Count the number of objects",
            "$inputs.prompt",
        ],
        json_schema_extra={
            "multiline": True,
        },
    )
    lmm_type: Union[Selector(kind=[STRING_KIND]), Literal["gpt_4v"]] = Field(
        description="Type of Large Multimodal Model to use. Currently only 'gpt_4v' (GPT-4 with Vision) is supported. This block is deprecated - consider using OpenAI GPT-4 Vision blocks (v1-v4) instead.",
        examples=["gpt_4v", "$inputs.lmm_type"],
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(),
        description="Configuration options for the LMM. Includes max_tokens (maximum length of response, default 450), gpt_image_detail ('low', 'high', or 'auto' - controls image resolution sent to API, 'auto' uses API defaults), and gpt_model_version (e.g., 'gpt-4o', defaults to 'gpt-4o'). Higher image detail provides better accuracy but uses more tokens.",
        examples=[
            {
                "max_tokens": 200,
                "gpt_image_detail": "low",
                "gpt_model_version": "gpt-4o",
            }
        ],
    )
    remote_api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), Optional[str]] = (
        Field(
            default=None,
            description="OpenAI API key required to use GPT-4 with Vision. You can obtain an API key from https://platform.openai.com/api-keys. This field is kept private for security. Required when lmm_type is 'gpt_4v'.",
            examples=["sk-xxx-xxx", "$inputs.openai_api_key", "$secrets.openai_key"],
            private=True,
        )
    )
    json_output: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary defining a JSON schema for structured output parsing. Maps field names to their descriptions (e.g., {'count': 'number of objects', 'color': 'dominant color'}). The block will attempt to parse the model's response as JSON matching this schema, extracting the specified fields. If parsing fails or fields are missing, they will be set to 'not_detected'. Leave as None to get raw text output only.",
        examples=[
            {"count": "number of cats in the picture", "breed": "cat breed"},
            {"objects": "list of objects found"},
            "$inputs.json_output",
        ],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

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
        if self.json_output is None:
            return result
        for key in self.json_output.keys():
            result.append(OutputDefinition(name=key, kind=[WILDCARD_KIND]))
        return result

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class LMMBlockV1(WorkflowBlock):

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

    def run(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                prompt=prompt,
                lmm_type=lmm_type,
                lmm_config=lmm_config,
                remote_api_key=remote_api_key,
                json_output=json_output,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                prompt=prompt,
                lmm_type=lmm_type,
                lmm_config=lmm_config,
                remote_api_key=remote_api_key,
                json_output=json_output,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> BlockResult:
        if json_output:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output, indent=4)}"
            )
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = run_gpt_4v_llm_prompting(
                image=images_prepared_for_processing,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raise ValueError(f"CogVLM has been removed from the Roboflow Core Models.")
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
        for prediction, image in zip(predictions, images):
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        lmm_type: str,
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
        json_output: Optional[Dict[str, str]],
    ) -> BlockResult:
        if json_output:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output, indent=4)}"
            )
        inference_images = [i.to_inference_format() for i in images]
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = run_gpt_4v_llm_prompting(
                image=inference_images,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raise ValueError(f"CogVLM has been removed from the Roboflow Core Models.")
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
        for prediction, image in zip(predictions, images):
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions


def run_gpt_4v_llm_prompting(
    image: List[Dict[str, Any]],
    prompt: str,
    remote_api_key: Optional[str],
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    if remote_api_key is None:
        raise ValueError(
            "Step that involves GPT-4V prompting requires OpenAI API key which was not provided."
        )
    return execute_gpt_4v_requests(
        image=image,
        remote_api_key=remote_api_key,
        prompt=prompt,
        lmm_config=lmm_config,
    )


def execute_gpt_4v_requests(
    image: List[dict],
    remote_api_key: str,
    prompt: str,
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    client = OpenAI(api_key=remote_api_key)
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
