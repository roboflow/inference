import base64
import json
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

from openai import OpenAI
from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.utils import run_in_parallel
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    SECRET_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

MODEL_VERSION_MAPPING = {
    "Qwen 3.5 9B - OpenRouter": "qwen/qwen3.5-9b",
    "Qwen 3.5 27B - OpenRouter": "qwen/qwen3.5-27b",
    "Qwen 3.5 122B A10B - OpenRouter": "qwen/qwen3.5-122b-a10b",
    "Qwen 3.5 397B A17B - OpenRouter": "qwen/qwen3.5-397b-a17b",
    "Qwen 3.5 Flash 02-23 - OpenRouter": "qwen/qwen3.5-flash-02-23",
    "Qwen 3.5 Plus 20260420 - OpenRouter": "qwen/qwen3.5-plus-20260420",
}

ModelVersion = Literal[
    "Qwen 3.5 9B - OpenRouter",
    "Qwen 3.5 27B - OpenRouter",
    "Qwen 3.5 122B A10B - OpenRouter",
    "Qwen 3.5 397B A17B - OpenRouter",
    "Qwen 3.5 Flash 02-23 - OpenRouter",
    "Qwen 3.5 Plus 20260420 - OpenRouter",
]

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
Ask a question to Qwen 3.5 vision-language models served via OpenRouter.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ API providers and model variants

Qwen 3.5 is exposed via [OpenRouter API](https://openrouter.ai/) and we require
passing an [OpenRouter API Key](https://openrouter.ai/docs/api-keys) to run.

Pick a specific model version from the `model_version` dropdown - new Qwen 3.5 releases
will be added to this list as they become available on OpenRouter.

!!! warning "API Usage Charges"

    OpenRouter is an external third party providing access to the model and incurring charges on the usage.
    Please check pricing on [openrouter.ai](https://openrouter.ai/) before use.

#### 💡 Further reading and Acceptable Use Policy

!!! warning "Model license"

    Check the [Qwen license terms](https://huggingface.co/Qwen) before use.
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
            "name": "Qwen 3.5 API",
            "version": "v1",
            "short_description": "Run Qwen 3.5 vision-language models via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Qwen License",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Qwen", "Qwen 3.5", "OpenRouter"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/qwen3_5_openrouter@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": {
                "structured-answering": "roboflow_core/json_parser@v1",
                "classification": "roboflow_core/vlm_as_classifier@v2",
                "multi-label-classification": "roboflow_core/vlm_as_classifier@v2",
                "object-detection": "roboflow_core/vlm_as_detector@v2",
            },
            "always_visible": True,
        },
    )
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Qwen 3.5 model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
            },
            "multiline": True,
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
    classes: Optional[Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]]] = Field(
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
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your OpenRouter API key",
        examples=["xxx-xxx", "$inputs.open_router_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "Qwen 3.5 9B - OpenRouter",
            "Qwen 3.5 27B - OpenRouter",
            "Qwen 3.5 122B A10B - OpenRouter",
            "Qwen 3.5 397B A17B - OpenRouter",
            "Qwen 3.5 Flash 02-23 - OpenRouter",
            "Qwen 3.5 Plus 20260420 - OpenRouter",
        ],
    ] = Field(
        default="Qwen 3.5 27B - OpenRouter",
        description="Model to be used",
        examples=["Qwen 3.5 27B - OpenRouter", "$inputs.qwen_model"],
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens the model can generate in it's response.",
        gt=1,
    )
    temperature: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.1,
        description="Temperature to sample from the model - value in range 0.0-2.0, the higher - the more "
        'random / "creative" the generations are.',
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
        "If not given - block defaults to value configured globally in Workflows Execution Engine. "
        "Please restrict if you hit limits.",
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
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: Union[str, float]) -> Union[str, float]:
        if isinstance(value, str):
            return value
        if value < 0.0 or value > 2.0:
            raise ValueError(
                "'temperature' parameter required to be in range [0.0, 2.0]"
            )
        return value

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

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
        return ">=1.3.0,<2.0.0"


class Qwen35OpenRouterBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
    ):
        self._model_manager = model_manager

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        task_type: TaskType,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: str,
        model_version: ModelVersion,
        max_tokens: int,
        temperature: float,
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        inference_images = [i.to_inference_format() for i in images]
        raw_outputs = run_qwen_llm_prompting(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            qwen_api_key=api_key,
            qwen_model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
        )
        return [
            {"output": raw_output, "classes": classes} for raw_output in raw_outputs
        ]


def run_qwen_llm_prompting(
    images: List[Dict[str, Any]],
    task_type: TaskType,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
    qwen_api_key: Optional[str],
    qwen_model_version: ModelVersion,
    max_tokens: int,
    temperature: float,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Task type: {task_type} not supported.")
    model_version_id = MODEL_VERSION_MAPPING.get(qwen_model_version)
    if model_version_id is None:
        raise ValueError(
            f"Invalid model name: '{qwen_model_version}'. Please use one of {list(MODEL_VERSION_MAPPING.keys())}."
        )
    qwen_prompts = []
    for image in images:
        loaded_image, _ = load_image(image)
        base64_image = base64.b64encode(
            encode_image_to_jpeg_bytes(loaded_image)
        ).decode("ascii")
        generated_prompt = PROMPT_BUILDERS[task_type](
            base64_image=base64_image,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        qwen_prompts.append(generated_prompt)
    return execute_qwen_requests(
        qwen_api_key=qwen_api_key,
        qwen_prompts=qwen_prompts,
        model_version_id=model_version_id,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent_requests=max_concurrent_requests,
    )


def execute_qwen_requests(
    qwen_api_key: str,
    qwen_prompts: List[List[dict]],
    model_version_id: str,
    max_tokens: int,
    temperature: float,
    max_concurrent_requests: Optional[int],
) -> List[str]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=qwen_api_key)
    tasks = [
        partial(
            execute_qwen_request,
            client=client,
            prompt=prompt,
            qwen_model_version=model_version_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for prompt in qwen_prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(
        tasks=tasks,
        max_workers=max_workers,
    )


def execute_qwen_request(
    client: OpenAI,
    prompt: List[dict],
    qwen_model_version: str,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=qwen_model_version,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response.choices is None:
        error_detail = getattr(response, "error", {}).get("message", "N/A")
        raise RuntimeError(
            "OpenRouter provider failed in delivering response. This issue happens from time "
            "to time - raise issue to OpenRouter if that's problematic for you. "
            f"Details: {error_detail}"
        )
    return response.choices[0].message.content


def prepare_unconstrained_prompt(
    base64_image: str,
    prompt: str,
    **kwargs,
) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]


def prepare_classification_prompt(
    base64_image: str, classes: List[str], **kwargs
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": "You act as single-class classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json [...]``` markers. "
            'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
            "`class-name` must be one of the class names defined by user. You are only allowed to return "
            "single JSON document, even if there are potentially multiple classes. You are not allowed to return list. "
            "You cannot discuss the result, you are only allowed to return JSON document.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def prepare_multi_label_classification_prompt(
    base64_image: str, classes: List[str], **kwargs
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": "You act as multi-label classification model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json``` markers. "
            'Expected structure of json: {"predicted_classes": [{"class": "class-name-1", "confidence": 0.9}, '
            '{"class": "class-name-2", "confidence": 0.7}]}. '
            "`class-name-X` must be one of the class names defined by user and `confidence` is a float value in range "
            "0.0-1.0 that represent how sure you are that the class is present in the image. Only return class names "
            "that are visible. You cannot discuss the result, you are only allowed to return JSON document.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def prepare_vqa_prompt(base64_image: str, prompt: str, **kwargs) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "You act as Visual Question Answering model. Your task is to provide answer to question"
            "submitted by user. If this is open-question - answer with few sentences, for ABCD question, "
            "return only the indicator of the answer.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {prompt}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def prepare_ocr_prompt(base64_image: str, **kwargs) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "You act as OCR model. Your task is to read text from the image and return it in "
            "paragraphs representing the structure of texts in the image. You should only return "
            "recognised text, nothing else.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def prepare_caption_prompt(
    base64_image: str, short_description: bool, **kwargs
) -> List[dict]:
    caption_detail_level = "Caption should be short."
    if not short_description:
        caption_detail_level = "Caption should be extensive."
    return [
        {
            "role": "system",
            "content": f"You act as image caption model. Your task is to provide description of the image. "
            f"{caption_detail_level}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def prepare_structured_answering_prompt(
    base64_image: str, output_structure: Dict[str, str], **kwargs
) -> List[dict]:
    output_structure_serialised = json.dumps(output_structure, indent=4)
    return [
        {
            "role": "system",
            "content": "You are supposed to produce responses in JSON wrapped in Markdown markers: "
            "```json\nyour-response\n```. User is to provide you dictionary with keys and values. "
            "Each key must be present in your response. Values in user dictionary represent "
            "descriptions for JSON fields to be generated. Provide only JSON Markdown in response.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Specification of requirements regarding output fields: \n"
                    f"{output_structure_serialised}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


def prepare_object_detection_prompt(
    base64_image: str, classes: List[str], **kwargs
) -> List[dict]:
    serialised_classes = ", ".join(classes)
    return [
        {
            "role": "system",
            "content": "You act as object-detection model. You must provide reasonable predictions. "
            "You are only allowed to produce JSON document in Markdown ```json``` markers. "
            'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, '
            '"class_name": "my-class-X", "confidence": 0.7}]}. '
            "`my-class-X` must be one of the class names defined by user. All coordinates must be in range 0.0-1.0, "
            "representing percentage of image dimensions. `confidence` is a value in range 0.0-1.0 representing your "
            "confidence in prediction. You should detect all instances of classes provided by user. "
            "You cannot discuss the result, you are only allowed to return JSON document.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"List of all classes to be recognised by model: {serialised_classes}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]


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
