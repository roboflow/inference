"""Unified Qwen-VL workflow block.

Subsumes the older per-version Qwen blocks (`qwen25vl@v1`, `qwen3vl@v1`,
`qwen3_5vl@v1`, `qwen3_5_openrouter@v1`, `qwen3_6_openrouter@v1`) into a
single block where the user picks:

* a ``backend`` — "Native (Roboflow)" or "OpenRouter"
* a ``model_version`` (combined version+size selector); each variant is bound
  to one backend
* the standard VLM ``task_type`` surface (unconstrained, OCR, classification,
  detection, etc.) shared with the Gemma/Llama/Kimi blocks

For the OpenRouter backend, all the API Key Passthrough plumbing
(``rf_key:account`` vs custom ``sk-or-...``, the ``privacy_level`` filter,
the proxy/billing flow) is inherited from
``common.openrouter.OpenRouterWorkflowBlockBase``.

For the Native backend, the block dispatches via
``StepExecutionMode.LOCAL``/``REMOTE`` to either ``model_manager`` (local
process) or ``InferenceHTTPClient`` (Roboflow-hosted inference), exactly
like the v1 native qwen blocks did.
"""

import base64
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.openrouter import (
    OpenRouterBlockManifestMixin,
    OpenRouterWorkflowBlockBase,
    RECOMMENDED_PARSERS,
    RELEVANT_TASKS_METADATA,
    SUPPORTED_TASK_TYPES_LIST,
    build_prompts_from_images,
    validate_task_type_required_fields,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

from inference.core.workflows.core_steps.common.openrouter import (
    PRIVACY_LEVEL_LITERAL,
    PRIVACY_LEVEL_METADATA,
)


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------
# Each entry is ``"<friendly label>": {"backend": "native" | "openrouter",
# "model_id": <id used by the chosen backend>}``.
#
# - Native model_ids are Roboflow inference model IDs (e.g. ``qwen3_5-2b``).
# - OpenRouter model_ids are OpenRouter slugs (e.g. ``qwen/qwen3.6-27b``).

MODEL_VARIANTS: Dict[str, Dict[str, str]] = {
    # Native — small models that run on Roboflow infrastructure.
    "Qwen 2.5 VL 7B - Native": {
        "backend": "native",
        "model_id": "qwen25-vl-7b",
    },
    "Qwen 3 VL 2B - Native": {
        "backend": "native",
        "model_id": "qwen3vl-2b-instruct",
    },
    "Qwen 3.5 VL 0.8B - Native": {
        "backend": "native",
        "model_id": "qwen3_5-0.8b",
    },
    "Qwen 3.5 VL 2B - Native": {
        "backend": "native",
        "model_id": "qwen3_5-2b",
    },
    # OpenRouter — large hosted models reached via OpenRouter.
    "Qwen 3.5 9B - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-9b",
    },
    "Qwen 3.5 27B - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-27b",
    },
    "Qwen 3.5 122B A10B - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-122b-a10b",
    },
    "Qwen 3.5 397B A17B - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-397b-a17b",
    },
    "Qwen 3.5 Flash 02-23 - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-flash-02-23",
    },
    "Qwen 3.5 Plus - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.5-plus-20260420",
    },
    "Qwen 3.6 27B - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.6-27b",
    },
    "Qwen 3.6 35B A3B - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.6-35b-a3b",
    },
    "Qwen 3.6 Flash - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.6-flash",
    },
    "Qwen 3.6 Plus - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.6-plus",
    },
    "Qwen 3.6 Max Preview - OpenRouter": {
        "backend": "openrouter",
        "model_id": "qwen/qwen3.6-max-preview",
    },
}

ModelVersion = Literal[tuple(MODEL_VARIANTS.keys())]
TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]
Backend = Literal["native", "openrouter"]


# ---------------------------------------------------------------------------
# Native prompt building
# ---------------------------------------------------------------------------
# The native Qwen API takes a single ``prompt`` string and the image
# separately (via LMMInferenceRequest). It uses the ``<system_prompt>``
# separator convention from the legacy native qwen blocks.
#
# We map each task type to a (system, user-text-template) pair; the
# combined prompt is ``user_text + "<system_prompt>" + system``. This
# mirrors the system prompts used in the OpenRouter messages-array
# builders in ``common.openrouter`` so behavior stays consistent.

_SYSTEM_CLASSIFICATION = (
    "You act as single-class classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    "```json [...]``` markers. Expected structure of json: "
    '{"class_name": "class-name", "confidence": 0.4}. `class-name` must be one '
    "of the class names defined by user. You are only allowed to return single "
    "JSON document, even if there are potentially multiple classes. You are not "
    "allowed to return list. You cannot discuss the result, you are only "
    "allowed to return JSON document."
)

_SYSTEM_MULTI_LABEL = (
    "You act as multi-label classification model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in Markdown "
    '```json``` markers. Expected structure of json: {"predicted_classes": '
    '[{"class": "class-name-1", "confidence": 0.9}, {"class": "class-name-2", '
    '"confidence": 0.7}]}. `class-name-X` must be one of the class names '
    "defined by user and `confidence` is a float value in range 0.0-1.0 that "
    "represent how sure you are that the class is present in the image. Only "
    "return class names that are visible. You cannot discuss the result, you "
    "are only allowed to return JSON document."
)

_SYSTEM_VQA = (
    "You act as Visual Question Answering model. Your task is to provide "
    "answer to question submitted by user. If this is open-question - answer "
    "with few sentences, for ABCD question, return only the indicator of "
    "the answer."
)

_SYSTEM_OCR = (
    "You act as OCR model. Your task is to read text from the image and "
    "return it in paragraphs representing the structure of texts in the "
    "image. You should only return recognised text, nothing else."
)

_SYSTEM_STRUCTURED = (
    "You are supposed to produce responses in JSON wrapped in Markdown "
    "markers: ```json\nyour-response\n```. User is to provide you "
    "dictionary with keys and values. Each key must be present in your "
    "response. Values in user dictionary represent descriptions for JSON "
    "fields to be generated. Provide only JSON Markdown in response."
)

_SYSTEM_DETECTION = (
    "You act as object-detection model. You must provide reasonable "
    "predictions. You are only allowed to produce JSON document in "
    'Markdown ```json``` markers. Expected structure of json: {"detections": '
    '[{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, '
    '"class_name": "my-class-X", "confidence": 0.7}]}. `my-class-X` must be '
    "one of the class names defined by user. All coordinates must be in "
    "range 0.0-1.0, representing percentage of image dimensions. "
    "`confidence` is a value in range 0.0-1.0 representing your confidence "
    "in prediction. You should detect all instances of classes provided by "
    "user. You cannot discuss the result, you are only allowed to return "
    "JSON document."
)


def _coerce_native_response_to_str(response: Any) -> str:
    """Normalize a native Qwen prediction.response into a string for the
    block's `output` field.

    Some Qwen variants return a ``{"thinking": "...", "answer": "..."}`` dict
    when reasoning is enabled. Strip the thinking and surface only the answer
    (matching the v1 native qwen blocks' behavior). For any other non-string
    type, JSON-serialize so downstream parsers (`vlm_as_classifier@v2`,
    `json_parser@v1`) get a string they can parse.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        answer = response.get("answer")
        if isinstance(answer, str) and answer:
            return answer
        return json.dumps(response)
    if response is None:
        return ""
    return json.dumps(response, default=str)


def _build_native_prompt(
    task_type: str,
    prompt: Optional[str],
    output_structure: Optional[Dict[str, str]],
    classes: Optional[List[str]],
) -> str:
    """Build the single-string prompt for the native Qwen API (LMMInferenceRequest).

    Returns ``user_text + "<system_prompt>" + system_text`` per the legacy
    native qwen convention.
    """
    if task_type == "unconstrained":
        # Manifest validation guarantees `prompt` is non-None for unconstrained,
        # so no fallback default is needed. Identity-prime the model as Qwen-VL
        # to match the legacy v1 native qwen blocks (some Qwen variants are
        # sensitive to identity priming).
        user_text = prompt or ""
        system_text = (
            "You are a Qwen vision-language model that can answer questions "
            "about any image."
        )
    elif task_type == "ocr":
        user_text = "Extract the text from this image."
        system_text = _SYSTEM_OCR
    elif task_type == "visual-question-answering":
        user_text = f"Question: {prompt}" if prompt else "Describe this image."
        system_text = _SYSTEM_VQA
    elif task_type == "caption":
        user_text = "Caption this image."
        system_text = (
            "You act as image caption model. Your task is to provide "
            "description of the image. Caption should be short."
        )
    elif task_type == "detailed-caption":
        user_text = "Describe this image in detail."
        system_text = (
            "You act as image caption model. Your task is to provide "
            "description of the image. Caption should be extensive."
        )
    elif task_type == "classification":
        cls_str = ", ".join(classes or [])
        user_text = f"List of all classes to be recognised by model: {cls_str}"
        system_text = _SYSTEM_CLASSIFICATION
    elif task_type == "multi-label-classification":
        cls_str = ", ".join(classes or [])
        user_text = f"List of all classes to be recognised by model: {cls_str}"
        system_text = _SYSTEM_MULTI_LABEL
    elif task_type == "object-detection":
        cls_str = ", ".join(classes or [])
        user_text = f"List of all classes to be recognised by model: {cls_str}"
        system_text = _SYSTEM_DETECTION
    elif task_type == "structured-answering":
        spec = json.dumps(output_structure or {}, indent=4)
        user_text = (
            f"Specification of requirements regarding output fields: \n{spec}"
        )
        system_text = _SYSTEM_STRUCTURED
    else:
        raise ValueError(f"Task type: {task_type} not supported.")
    return user_text + "<system_prompt>" + system_text


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)

LONG_DESCRIPTION = f"""
Run any Qwen vision-language model — natively on Roboflow infrastructure or via OpenRouter.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ Backend selection

* **Native (Roboflow)** — small Qwen-VL models (0.8B–7B) run on the same infrastructure as
  your other Roboflow models. Lower latency, no per-token billing. Recommended for tasks
  like OCR, captioning, and visual question answering.

* **OpenRouter** — large hosted Qwen models (9B–397B) reached via [OpenRouter](https://openrouter.ai/).
  Defaults to a Roboflow-managed API key and bills your Roboflow credits. Paste your own
  `sk-or-...` key in the `api_key` field to bypass Roboflow billing. Recommended for
  structured tasks that benefit from larger models (classification, object-detection,
  structured-answering).

The `model_version` dropdown lists every supported variant; each is bound to one backend.
A validator catches mismatches between your selected backend and model.

#### 🔒 Privacy filter (OpenRouter only)

* **No data collection** *(default)* – providers may not train on your inputs.
* **Allow data collection** – broader provider pool.
* **Zero data retention** – strictest, restricts to providers that retain nothing.
"""


class BlockManifest(OpenRouterBlockManifestMixin):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Qwen-VL (Unified)",
            "version": "v1",
            "short_description": "Run any Qwen vision model — natively or via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Qwen",
                "qwen-vl",
                "qwen3.5",
                "qwen3.6",
                "VLM",
                "Alibaba",
                "OpenRouter",
            ],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/qwen_vlm@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    backend: Backend = Field(
        default="native",
        description=(
            "Where to run inference. Native = Roboflow infrastructure. "
            "OpenRouter = large hosted Qwen models via OpenRouter."
        ),
        json_schema_extra={
            "values_metadata": {
                "native": {
                    "name": "Native (Roboflow)",
                    "description": (
                        "Runs locally on the inference server, or remotely "
                        "via Roboflow Hosted Inference. Smaller models, "
                        "lower latency, no per-token billing."
                    ),
                },
                "openrouter": {
                    "name": "OpenRouter",
                    "description": (
                        "Routes to large hosted Qwen models via OpenRouter. "
                        "Defaults to a Roboflow-managed key (billed in "
                        "credits)."
                    ),
                },
            },
            "always_visible": True,
        },
    )

    task_type: TaskType = Field(
        default="unconstrained",
        description="Task type to be performed by model. Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": RECOMMENDED_PARSERS,
            "always_visible": True,
        },
    )

    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Qwen model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["unconstrained", "visual-question-answering"],
                    "required": True,
                },
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
                "task_type": {"values": ["structured-answering"], "required": True},
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
                    "values": [
                        "classification",
                        "multi-label-classification",
                        "object-detection",
                    ],
                    "required": True,
                },
            },
        },
    )

    model_version: Union[Selector(kind=[STRING_KIND]), ModelVersion] = Field(
        default="Qwen 3.5 VL 2B - Native",
        description="Qwen model variant. Each variant is bound to one backend; mismatched selections are rejected.",
        examples=["Qwen 3.5 VL 2B - Native", "Qwen 3.6 27B - OpenRouter"],
    )

    # --- Override inherited OpenRouter fields with relevant_for=openrouter ---
    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description=(
            "OpenRouter API key (only used when backend=openrouter). Defaults "
            "to Roboflow's managed key. Provide your own `sk-or-...` key to "
            "call OpenRouter directly without Roboflow billing."
        ),
        examples=["rf_key:account", "sk-or-...", "$inputs.openrouter_api_key"],
        private=True,
        json_schema_extra={
            "relevant_for": {"backend": {"values": ["openrouter"], "required": False}},
        },
    )
    privacy_level: PRIVACY_LEVEL_LITERAL = Field(
        default="deny",
        description=(
            "Provider privacy filter (only used when backend=openrouter). "
            "Stricter levels reduce the pool of providers and may increase "
            "per-call cost on the managed key."
        ),
        json_schema_extra={
            "values_metadata": PRIVACY_LEVEL_METADATA,
            "relevant_for": {"backend": {"values": ["openrouter"], "required": False}},
        },
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        validate_task_type_required_fields(
            task_type=self.task_type,
            prompt=self.prompt,
            classes=self.classes,
            output_structure=self.output_structure,
        )
        # Cross-check: the selected model must support the selected backend.
        # Skip when model_version is a selector reference (starts with $).
        if isinstance(self.model_version, str) and not self.model_version.startswith("$"):
            variant = MODEL_VARIANTS.get(self.model_version)
            if variant is not None and variant["backend"] != self.backend:
                raise ValueError(
                    f"Model '{self.model_version}' requires backend "
                    f"'{variant['backend']}', but '{self.backend}' was selected. "
                    f"Either change the backend or pick a model bound to "
                    f"'{self.backend}'."
                )
        return self

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
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

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


# ---------------------------------------------------------------------------
# Block class
# ---------------------------------------------------------------------------


class QwenVlmBlockV1(OpenRouterWorkflowBlockBase):
    """Unified Qwen-VL block. Inherits OpenRouter routing/execution from base
    and adds the native local/remote dispatch on top.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        super().__init__(model_manager=model_manager, api_key=api_key)
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        backend: str,
        model_version: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: str,
        privacy_level: str,
        max_tokens: int,
        temperature: float,
        max_concurrent_requests: Optional[int],
    ) -> BlockResult:
        variant = MODEL_VARIANTS.get(model_version)
        if variant is None:
            raise ValueError(
                f"Unknown Qwen variant '{model_version}'. Pick one of: "
                f"{list(MODEL_VARIANTS.keys())}"
            )
        if variant["backend"] != backend:
            raise ValueError(
                f"Model '{model_version}' requires backend "
                f"'{variant['backend']}', but '{backend}' was selected."
            )
        model_id = variant["model_id"]

        if backend == "openrouter":
            inference_images = [i.to_inference_format() for i in images]
            prompts = build_prompts_from_images(
                images=inference_images,
                task_type=task_type,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            raw_outputs = self.execute_openrouter_batch(
                openrouter_api_key=api_key,
                model=model_id,
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                privacy_level=privacy_level,
                max_concurrent_requests=max_concurrent_requests,
            )
        elif backend == "native":
            raw_outputs = self._run_native(
                images=images,
                model_id=model_id,
                task_type=task_type,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return [{"output": o, "classes": classes} for o in raw_outputs]

    # ----------------------- Native dispatch -----------------------

    def _run_native(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        max_tokens: Optional[int],
    ) -> List[str]:
        combined_prompt = _build_native_prompt(
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
        )
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self._run_native_locally(
                images=images,
                model_id=model_id,
                combined_prompt=combined_prompt,
                max_new_tokens=max_tokens,
            )
        if self._step_execution_mode == StepExecutionMode.REMOTE:
            return self._run_native_remotely(
                images=images,
                model_id=model_id,
                combined_prompt=combined_prompt,
                max_new_tokens=max_tokens,
            )
        raise ValueError(
            f"Unknown step_execution_mode: {self._step_execution_mode}"
        )

    def _run_native_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        combined_prompt: str,
        max_new_tokens: Optional[int],
    ) -> List[str]:
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        self._model_manager.add_model(
            model_id=model_id, api_key=self._roboflow_api_key
        )
        outputs: List[str] = []
        for image in inference_images:
            request_kwargs: Dict[str, Any] = dict(
                api_key=self._roboflow_api_key,
                model_id=model_id,
                image=image,
                source="workflow-execution",
                prompt=combined_prompt,
            )
            if max_new_tokens is not None:
                request_kwargs["max_new_tokens"] = max_new_tokens
            request = LMMInferenceRequest(**request_kwargs)
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_id, request=request
            )
            outputs.append(_coerce_native_response_to_str(prediction.response))
        return outputs

    def _run_native_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        combined_prompt: str,
        max_new_tokens: Optional[int],
    ) -> List[str]:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url, api_key=self._roboflow_api_key
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        outputs: List[str] = []
        for image in images:
            kwargs: Dict[str, Any] = dict(
                inference_input=image.base64_image,
                model_id=model_id,
                prompt=combined_prompt,
                model_id_in_path=True,
            )
            if max_new_tokens is not None:
                kwargs["max_new_tokens"] = max_new_tokens
            result = client.infer_lmm(**kwargs)
            response_text = result.get("response", result)
            outputs.append(_coerce_native_response_to_str(response_text))
        return outputs
