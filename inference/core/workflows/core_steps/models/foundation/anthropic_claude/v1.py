import base64
import json
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import anthropic
from anthropic import NOT_GIVEN
from pydantic import ConfigDict, Field, model_validator

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio
from inference.core.workflows.core_steps.common.utils import run_in_parallel
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INTEGER_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
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

SUPPORTED_TASK_TYPES_LIST = [
    "unconstrained",
    "ocr",
    "structured-answering",
    "classification",
    "multi-label-classification",
    "visual-question-answering",
    "caption",
    "detailed-caption",
    "object-detection",
]
SUPPORTED_TASK_TYPES = set(SUPPORTED_TASK_TYPES_LIST)
RELEVANT_TASKS_METADATA = {
    k: v for k, v in VLM_TASKS_METADATA.items() if k in SUPPORTED_TASK_TYPES
}
RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)
LONG_DESCRIPTION = f"""
Ask a question to Anthropic Claude model with vision capabilities.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

You need to provide your Anthropic API key to use the Claude model. 
"""

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

TASKS_REQUIRING_PROMPT = {
    "unconstrained",
    "visual-question-answering",
}

TASKS_REQUIRING_CLASSES = {
    "classification",
    "multi-label-classification",
    "object-detection",
}

TASKS_REQUIRING_OUTPUT_STRUCTURE = {
    "structured-answering",
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Anthropic Claude",
            "version": "v1",
            "short_description": "Run Anthropic Claude model with vision capabilities",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Claude", "Anthropic"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
        }
    )
    type: Literal["roboflow_core/anthropic_claude@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": {
                "structured-answering": "roboflow_core/json_parser@v1",
                "classification": "roboflow_core/vlm_as_classifier@v1",
                "multi-label-classification": "roboflow_core/vlm_as_classifier@v1",
                "object-detection": "roboflow_core/vlm_as_detector@v1",
            },
            "always_visible": True,
        },
    )
    prompt: Optional[Union[WorkflowParameterSelector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Claude model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
            },
        },
    )
    output_structure: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary with structure of expected JSON response",
        examples=[{"my_key": "description"}, "$inputs.output_structure"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_OUTPUT_STRUCTURE,
                    "required": True,
                },
            },
        },
    )
    classes: Optional[
        Union[WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]]
    ] = Field(
        default=None,
        description="List of classes to be used",
        examples=[["class-a", "class-b"], "$inputs.classes"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_CLASSES,
                    "required": True,
                },
            },
        },
    )
    api_key: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Your Antropic API key",
        examples=["xxx-xxx", "$inputs.antropics_api_key"],
        private=True,
    )
    model_version: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        Literal[
            "claude-3-5-sonnet", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"
        ],
    ] = Field(
        default="claude-3-5-sonnet",
        description="Model to be used",
        examples=["claude-3-5-sonnet", "$inputs.claude"],
    )
    max_tokens: int = Field(
        default=450,
        description="Maximum number of tokens the model can generate in it's response.",
    )
    temperature: Optional[
        Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])]
    ] = Field(
        default=None,
        description="Temperature to sample from the model - value in range 0.0-2.0, the higher - the more "
        'random / "creative" the generations are.',
        ge=0.0,
        le=2.0,
    )
    max_image_size: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        description="Maximum size of the image - if input has larger side, it will be downscaled, keeping aspect ratio",
        default=1024,
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
        "If not given - block defaults to value configured globally in Workflows Execution Engine. "
        "Please restrict if you hit ANtropic API limits.",
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.task_type in TASKS_REQUIRING_PROMPT and self.prompt is None:
            raise ValueError(
                f"`prompt` parameter required to be set for task `{self.task_type}`"
            )
        if self.task_type in TASKS_REQUIRING_CLASSES and self.classes is None:
            raise ValueError(
                f"`classes` parameter required to be set for task `{self.task_type}`"
            )
        if (
            self.task_type in TASKS_REQUIRING_OUTPUT_STRUCTURE
            and self.output_structure is None
        ):
            raise ValueError(
                f"`output_structure` parameter required to be set for task `{self.task_type}`"
            )
        return self

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class AntropicClaudeBlockV1(WorkflowBlock):

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
        task_type: TaskType,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: str,
        model_version: str,
        max_tokens: int,
        temperature: Optional[float],
        max_image_size: int,
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_claude_prompting(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            api_key=api_key,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            max_image_size=max_image_size,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_claude_prompting(
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    api_key: str,
    model_version: str,
    max_tokens: int,
    temperature: Optional[float],
    max_image_size: int,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    prompts = []
    for image in images:
        loaded_image, _ = load_image(image)
        loaded_image = downscale_image_keeping_aspect_ratio(
            image=loaded_image, desired_size=(max_image_size, max_image_size)
        )
        base64_image = base64.b64encode(
            encode_image_to_jpeg_bytes(loaded_image)
        ).decode("ascii")
        generated_prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        prompts.append(generated_prompt)
    return execute_claude_requests(
        api_key=api_key,
        prompts=prompts,
        model_version=model_version,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_claude_requests(
    api_key: str,
    prompts: List[Tuple[Optional[str], List[dict]]],
    model_version: str,
    max_tokens: int,
    temperature: Optional[float],
    max_concurrent_requests: Optional[int],
) -> List[str]:
    tasks = [
        partial(
            execute_claude_request,
            system_prompt=prompt[0],
            messages=prompt[1],
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )
        for prompt in prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


EXACT_MODELS_VERSIONS_MAPPING = {
    "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


def execute_claude_request(
    system_prompt: Optional[str],
    messages: List[dict],
    model_version: str,
    max_tokens: int,
    temperature: Optional[float],
    api_key: str,
) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    if system_prompt is None:
        system_prompt = NOT_GIVEN
    if temperature is None:
        temperature = NOT_GIVEN
    result = client.messages.create(
        system=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        model=EXACT_MODELS_VERSIONS_MAPPING[model_version],
        temperature=temperature,
    )
    return result.content[0].text


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    return None, messages


def prepare_classification_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    serialised_classes = ", ".join(classes)
    system_prompt = (
        "You act as single-class classification model. You must provide reasonable predictions. "
        "You are only allowed to produce JSON document. "
        'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
        "`class-name` must be one of the class names defined by user. You are only allowed to return "
        "single JSON document, even if there are potentially multiple classes. You are not allowed to "
        "return list."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_multi_label_classification_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    serialised_classes = ", ".join(classes)
    system_prompt = (
        "You act as multi-label classification model. You must provide reasonable predictions. "
        "You are only allowed to produce JSON document. "
        'Expected structure of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
        '{"class": "class-name-2", "confidence": 0.7}]}.'
        "`class-name-X` must be one of the class names defined by user and `confidence` is a float value "
        "in range 0.0-1.0 that represents how sure you are that the class is present in the image. "
        "Only return class names that are visible."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_vqa_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    system_prompt = (
        "You act as Visual Question Answering model. Your task is to provide answer to question"
        "submitted by user. If this is open-question - answer with few sentences, for ABCD question, "
        "return only the indicator of the answer."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"Question: {prompt}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_ocr_prompt(
    base64_image: str,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    system_prompt = (
        "You act as OCR model. Your task is to read text from the image and return it in "
        "paragraphs representing the structure of texts in the image. You should only return "
        "recognised text, nothing else."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_caption_prompt(
    base64_image: str,
    short_description: bool,
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    caption_detail_level = "Caption should be short."
    if not short_description:
        caption_detail_level = "Caption should be extensive."
    system_prompt = (
        f"You act as image caption model. Your task is to provide description of the image. "
        f"{caption_detail_level}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_structured_answering_prompt(
    base64_image: str,
    output_structure: Dict[str, str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    output_structure_serialised = json.dumps(output_structure, indent=4)
    system_prompt = (
        "You are supposed to produce responses in JSON. User is to provide you dictionary with "
        "keys and values. Each key must be present in your response. Values in user dictionary "
        "represent descriptions for JSON fields to be generated. Provide only JSON in response."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"Specification of requirements regarding output fields: \n"
                    f"{output_structure_serialised}",
                },
            ],
        }
    ]
    return system_prompt, messages


def prepare_object_detection_prompt(
    base64_image: str,
    classes: List[str],
    **kwargs,
) -> Tuple[Optional[str], List[dict]]:
    serialised_classes = ", ".join(classes)
    system_prompt = (
        "You act as object-detection model. You must provide reasonable predictions. "
        "You are only allowed to produce JSON document. "
        'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, "class_name": "my-class-X", "confidence": 0.7}]} '
        "- remember to close top-level dictionary at the end. "
        "`my-class-X` must be one of the class names defined by user. All coordinates must be in range 0.0-1.0, representing percentage of image dimensions. "
        "`confidence` is a value in range 0.0-1.0 representing your confidence in prediction. You should detect all instances of classes provided by user."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
            ],
        }
    ]
    return system_prompt, messages


PROMPT_BUILDERS = {
    "unconstrained": prepare_unconstrained_prompt,
    "ocr": prepare_ocr_prompt,
    "visual-question-answering": prepare_vqa_prompt,
    "caption": partial(prepare_caption_prompt, short_description=True),
    "detailed-caption": partial(prepare_caption_prompt, short_description=False),
    "classification": prepare_classification_prompt,
    "multi-label-classification": prepare_multi_label_classification_prompt,
    "structured-answering": prepare_structured_answering_prompt,
    "object-detection": prepare_object_detection_prompt,
}
