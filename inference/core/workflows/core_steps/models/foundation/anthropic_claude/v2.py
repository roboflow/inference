import base64
import json
import logging
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import anthropic
import numpy as np
import supervision as sv
from anthropic import NOT_GIVEN
from pydantic import AfterValidator, ConfigDict, Field, model_validator
from supervision.config import CLASS_NAME_DATA_FIELD
from typing_extensions import Annotated

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
    run_in_parallel,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
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

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)

TASKS_DESCRIPTIONS = {
    "unconstrained": "Let you use arbitrary prompt",
    "ocr": "Model recognises text in the image",
    "visual-question-answering": "Model answers the question you submit in the prompt",
    "caption": "Model describes the image",
    "detailed-caption": "Model provides long description of the image",
    "classification": "Model classifies the image content selecting one of many classes you suggested",
    "multi-label-classification": "Model classifies the image content selecting potentially multiple "
    "classes you suggested",
    "object-detection": "Model detect instances of classes you suggested",
    "structured-answering": "Model produces JSON structure that you specify",
}

DESCRIPTIONS_TEXT = "\n\n".join(
    f"- `{key}` - {value}" for key, value in TASKS_DESCRIPTIONS.items()
)

LONG_DESCRIPTION = f"""
Ask a question to Anthropic Claude model with vision capabilities.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{DESCRIPTIONS_TEXT}

You need to provide your Anthropic API key to use the Claude model. 
"""

TaskType = Literal[
    "unconstrained",
    "ocr",
    "visual-question-answering",
    "caption",
    "detailed-caption",
    "classification",
    "multi-label-classification",
    "structured-answering",
    "object-detection",
]

TASKS_REQUIRING_PROMPT = {
    "unconstrained",
    "visual-question-answering",
}

CLASSIFICATION_TASKS = {
    "classification",
    "multi-label-classification",
}

TASKS_REQUIRING_CLASSES = {
    "classification",
    "multi-label-classification",
    "object-detection",
}

TASKS_REQUIRING_OUTPUT_STRUCTURE = {
    "structured-answering",
}


def validate_output_structure(output_structure: Optional[dict]) -> Optional[dict]:
    if output_structure is None:
        return output_structure
    if "error_status" in output_structure or "output" in output_structure:
        raise ValueError(
            "`error_status` and `output` are reserved field names and cannot be "
            "used in `output_structure` of `roboflow_core/anthropic_claude@v2` block."
        )
    return output_structure


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Anthropic Claude",
            "version": "v2",
            "short_description": "Run Anthropic Claude model with vision capabilities",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Claude", "Anthropic"],
        }
    )
    type: Literal["roboflow_core/anthropic_claude@v2"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    task_type: TaskType = Field(
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_descriptions": TASKS_DESCRIPTIONS,
            "impact_on_outputs": {
                "classification": {
                    "type": "StaticSetOfOutputs",
                    "added_outputs": [
                        {
                            "name": "predictions",
                            "kind": [CLASSIFICATION_PREDICTION_KIND.name],
                        },
                        {"name": "inference_id", "kind": [STRING_KIND.name]},
                        {"name": "error_status", "kind": [BOOLEAN_KIND.name]},
                    ],
                },
                "multi-label-classification": {
                    "type": "StaticSetOfOutputs",
                    "added_outputs": [
                        {
                            "name": "predictions",
                            "kind": [CLASSIFICATION_PREDICTION_KIND.name],
                        },
                        {"name": "inference_id", "kind": [STRING_KIND.name]},
                        {"name": "error_status", "kind": [BOOLEAN_KIND.name]},
                    ],
                },
                "object-detection": {
                    "type": "StaticSetOfOutputs",
                    "added_outputs": [
                        {
                            "name": "predictions",
                            "kind": [OBJECT_DETECTION_PREDICTION_KIND.name],
                        },
                        {"name": "inference_id", "kind": [STRING_KIND.name]},
                        {"name": "error_status", "kind": [BOOLEAN_KIND.name]},
                    ],
                },
                "structured-answering": {
                    "type": "DynamicSetOfOutputs",
                    "dependency_type": "DictionaryKeysDictateOutputs",
                    "dependent_on": "output_structure",
                    "assumed_kind": [WILDCARD_KIND.name],
                },
            },
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
    output_structure: Annotated[
        Optional[Dict[str, str]], AfterValidator(validate_output_structure)
    ] = Field(
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
            OutputDefinition(name="*"),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            )
        ]
        if self.task_type in CLASSIFICATION_TASKS:
            result.extend(
                [
                    OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
                    OutputDefinition(
                        name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]
                    ),
                    OutputDefinition(name="inference_id", kind=[STRING_KIND]),
                ]
            )
        if self.task_type == "object-detection":
            result.extend(
                [
                    OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
                    OutputDefinition(
                        name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
                    ),
                    OutputDefinition(name="inference_id", kind=[STRING_KIND]),
                ]
            )
        if self.task_type == "structured-answering":
            for field_name in self.output_structure.keys():
                result.append(OutputDefinition(name=field_name))
        return result

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class AntropicClaudeBlockV2(WorkflowBlock):

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
        if task_type in CLASSIFICATION_TASKS:
            return prepare_classification_results(
                images=images,
                raw_outputs=raw_outputs,
                classes=classes,
            )
        if task_type == "object-detection":
            return prepare_object_detection_results(
                images=images,
                raw_outputs=raw_outputs,
                classes=classes,
            )
        if task_type == "structured-answering":
            return prepare_structured_answering_results(
                raw_outputs=raw_outputs, expected_fields=list(output_structure.keys())
            )
        return [{"output": raw_output} for raw_output in raw_outputs]


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
        prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        prompts.append(prompt)
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


def prepare_classification_results(
    images: Batch[WorkflowImageData],
    raw_outputs: List[str],
    classes: List[str],
) -> List[dict]:
    result = []
    for image, vlm_output in zip(images, raw_outputs):
        inference_id = f"{uuid4()}"
        error_status, parsed_data = string2json(
            raw_json=vlm_output,
        )
        if error_status:
            result.append(
                {
                    "output": vlm_output,
                    "error_status": True,
                    "predictions": None,
                    "inference_id": inference_id,
                }
            )
        elif "class_name" in parsed_data and "confidence" in parsed_data:
            parsed_output = parse_multi_class_classification_results(
                image=image,
                results=parsed_data,
                classes=classes,
                inference_id=inference_id,
            )
            result.append(
                {
                    "output": vlm_output,
                    "error_status": False,
                    "predictions": parsed_output,
                    "inference_id": inference_id,
                }
            )
        elif "predicted_classes" in parsed_data:
            parsed_output = parse_multi_label_classification_results(
                image=image,
                results=parsed_data,
                classes=classes,
                inference_id=inference_id,
            )
            result.append(
                {
                    "output": vlm_output,
                    "error_status": False,
                    "predictions": parsed_output,
                    "inference_id": inference_id,
                }
            )
        else:
            result.append(
                {
                    "output": vlm_output,
                    "error_status": True,
                    "predictions": None,
                    "inference_id": inference_id,
                }
            )
    return result


def string2json(
    raw_json: str,
) -> Tuple[bool, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw_json)
    if len(json_blocks_found) == 0:
        return try_parse_json(raw_json)
    first_block = json_blocks_found[0]
    return try_parse_json(first_block)


def try_parse_json(content: str) -> Tuple[bool, dict]:
    try:
        return False, json.loads(content)
    except Exception as error:
        logging.warning(
            f"Could not parse JSON to dict in `roboflow_core/vlm_as_classifier@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return True, {}


def parse_multi_class_classification_results(
    image: WorkflowImageData,
    results: dict,
    classes: List[str],
    inference_id: str,
) -> dict:
    try:
        class2id_mapping = create_classes_index(classes=classes)
        height, width = image.numpy_image.shape[:2]
        top_class = results["class_name"]
        confidences = {top_class: scale_confidence(results["confidence"])}
        predictions = []
        if top_class not in class2id_mapping:
            predictions.append(
                {
                    "class": top_class,
                    "class_id": -1,
                    "confidence": confidences.get(top_class, 0.0),
                }
            )
        for class_name, class_id in class2id_mapping.items():
            predictions.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidences.get(class_name, 0.0),
                }
            )
        parsed_prediction = {
            "image": {"width": width, "height": height},
            "predictions": predictions,
            "top": top_class,
            "confidence": confidences[top_class],
            "inference_id": inference_id,
            "parent_id": image.parent_metadata.parent_id,
        }
        return {
            "error_status": False,
            "predictions": parsed_prediction,
            "inference_id": inference_id,
        }
    except Exception as error:
        logging.warning(
            f"Could not parse multi-class classification results in `roboflow_core/vlm_as_classifier@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return {"error_status": True, "predictions": None, "inference_id": inference_id}


def parse_multi_label_classification_results(
    image: WorkflowImageData,
    results: dict,
    classes: List[str],
    inference_id: str,
) -> dict:
    try:
        class2id_mapping = create_classes_index(classes=classes)
        height, width = image.numpy_image.shape[:2]
        predicted_classes_confidences = {}
        for prediction in results["predicted_classes"]:
            if prediction["class"] not in class2id_mapping:
                class2id_mapping[prediction["class"]] = -1
            if prediction["class"] in predicted_classes_confidences:
                old_confidence = predicted_classes_confidences[prediction["class"]]
                new_confidence = scale_confidence(value=prediction["confidence"])
                predicted_classes_confidences[prediction["class"]] = max(
                    old_confidence, new_confidence
                )
            else:
                predicted_classes_confidences[prediction["class"]] = scale_confidence(
                    value=prediction["confidence"]
                )
        predictions = {
            class_name: {
                "confidence": predicted_classes_confidences.get(class_name, 0.0),
                "class_id": class_id,
            }
            for class_name, class_id in class2id_mapping.items()
        }
        parsed_prediction = {
            "image": {"width": width, "height": height},
            "predictions": predictions,
            "predicted_classes": list(predicted_classes_confidences.keys()),
            "inference_id": inference_id,
            "parent_id": image.parent_metadata.parent_id,
        }
        return {
            "error_status": False,
            "predictions": parsed_prediction,
            "inference_id": inference_id,
        }
    except Exception as error:
        logging.warning(
            f"Could not parse multi-label classification results in `roboflow_core/vlm_as_classifier@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return {"error_status": True, "predictions": None, "inference_id": inference_id}


def create_classes_index(classes: List[str]) -> Dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def scale_confidence(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def prepare_object_detection_results(
    images: Batch[WorkflowImageData],
    raw_outputs: List[str],
    classes: List[str],
) -> List[dict]:
    result = []
    for image, vlm_output in zip(images, raw_outputs):
        inference_id = f"{uuid4()}"
        error_status, parsed_data = string2json(
            raw_json=vlm_output,
        )
        if error_status:
            result.append(
                {
                    "output": vlm_output,
                    "error_status": True,
                    "predictions": None,
                    "inference_id": inference_id,
                }
            )
            continue
        try:
            predictions = parse_claude_object_detection_response(
                image=image,
                parsed_data=parsed_data,
                classes=classes,
                inference_id=inference_id,
            )
            result.append(
                {
                    "output": vlm_output,
                    "error_status": False,
                    "predictions": predictions,
                    "inference_id": inference_id,
                }
            )
        except Exception as error:
            logging.warning(
                f"Could not parse VLM prediction for model `claude` and task `object-detection`"
                f"in `roboflow_core/vlm_as_detector@v1` block. "
                f"Error type: {error.__class__.__name__}. Details: {error}"
            )
            result.append(
                {
                    "output": vlm_output,
                    "error_status": True,
                    "predictions": None,
                    "inference_id": inference_id,
                }
            )
    return result


def parse_claude_object_detection_response(
    image: WorkflowImageData,
    parsed_data: dict,
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]
    if len(parsed_data["detections"]) == 0:
        return sv.Detections.empty()
    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in parsed_data["detections"]:
        xyxy.append(
            [
                detection["x_min"] * image_width,
                detection["y_min"] * image_height,
                detection["x_max"] * image_width,
                detection["y_max"] * image_height,
            ]
        )
        class_id.append(class_name2id.get(detection["class_name"], -1))
        class_name.append(detection["class_name"])
        confidence.append(scale_confidence(detection.get("confidence", 1.0)))
    xyxy = np.array(xyxy).round(0) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    detection_ids = np.array([str(uuid4()) for _ in range(len(xyxy))])
    dimensions = np.array([[image_height, image_width]] * len(xyxy))
    inference_ids = np.array([inference_id] * len(xyxy))
    prediction_type = np.array(["object-detection"] * len(xyxy))
    data = {
        CLASS_NAME_DATA_FIELD: class_name,
        IMAGE_DIMENSIONS_KEY: dimensions,
        INFERENCE_ID_KEY: inference_ids,
        DETECTION_ID_KEY: detection_ids,
        PREDICTION_TYPE_KEY: prediction_type,
    }
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=None,
        tracker_id=None,
        data=data,
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )


def prepare_structured_answering_results(
    raw_outputs: List[str],
    expected_fields: List[str],
) -> List[dict]:
    result = []
    for raw_json in raw_outputs:
        error_status, parsed_data = string2json_with_expected_fields(
            raw_json=raw_json,
            expected_fields=expected_fields,
        )
        parsed_data["output"] = raw_json
        parsed_data["error_status"] = error_status
        result.append(parsed_data)
    return result


def string2json_with_expected_fields(
    raw_json: str,
    expected_fields: List[str],
) -> Tuple[bool, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw_json)
    if len(json_blocks_found) == 0:
        return try_parse_json_with_expected_fields(
            raw_json, expected_fields=expected_fields
        )
    first_block = json_blocks_found[0]
    return try_parse_json_with_expected_fields(
        first_block, expected_fields=expected_fields
    )


def try_parse_json_with_expected_fields(
    content: str, expected_fields: List[str]
) -> Tuple[bool, dict]:
    try:
        parsed_data = json.loads(content)
        result = {}
        all_fields_find = True
        for field in expected_fields:
            if field not in parsed_data:
                all_fields_find = False
            result[field] = parsed_data.get(field)
        return not all_fields_find, result
    except Exception as error:
        logging.warning(
            f"Could not parse JSON in `roboflow_core/json_parser@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return True, {field: None for field in expected_fields}
