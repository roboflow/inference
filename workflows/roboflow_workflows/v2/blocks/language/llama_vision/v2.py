"""Ask Llama 3.2 Vision about each image (deprecated, as in V1).

Canonical type: ``roboflow_core/llama_vision@v2``.

Constructor resources come from ``OpenRouterBlockBase`` (``openrouter/v3.py``):
``platform_client`` and ``api_key`` (the host's Roboflow key, used for managed
``rf_key:`` keys), ``openrouter_base_url`` and ``openrouter_http_client``. Its
``ask_all`` owns key handling, the Roboflow proxy, privacy routing, the
reasoning retry and request concurrency; this module checks the inputs, builds
the messages and reads the answers.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.language._shared.openrouter_vendors import (
    DECODING_TASKS,
    VendorModel,
    resolve_model,
)
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    check_task_fields,
    third_party_model,
)
from roboflow_workflows.v2.blocks.language.decoding import checked_classes
from roboflow_workflows.v2.blocks.language.openrouter.v3 import (
    OPENROUTER_TASKS,
    PRIVACY_LEVEL_METADATA,
    OpenRouterBlockBase,
    OpenRouterTask,
    PrivacyLevel,
    openrouter_task_messages,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.engine.declaration import (
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

LLAMA_MODELS: Dict[str, VendorModel] = {
    "11B - OpenRouter": VendorModel("meta-llama/llama-3.2-11b-vision-instruct"),
}


class LlamaVisionBlock(OpenRouterBlockBase):
    """Ask Llama 3.2 Vision about each image (deprecated, as in V1).

    OpenRouter no longer hosts Llama 3.2 Vision, so real requests are
    expected to fail; the block stays for existing workflows. It returns the
    raw answer only. Classification can use ``roboflow_core/vlm_as_classifier@v2``.
    Object detection returns raw normalized 0-1 ``x_min``/``y_min``/``x_max``/
    ``y_max`` fields in ``output``; no matching supported Llama ``model_type``
    exists in ``roboflow_core/vlm_as_detector@v2``. Use ``roboflow_core/meta_vlm@v3`` or
    ``roboflow_core/openrouter@v3`` for native decoded ``predictions``.
    """

    type = "roboflow_core/llama_vision@v2"
    metadata = {
        "name": "Llama 3.2 Vision",
        "port_of": "roboflow_core/llama_vision@v2",
        "section": "model",
        "license": "Llama 3.2 Community License",
        "deprecated": True,
        "deprecation_message": (
            "OpenRouter no longer hosts Llama 3.2 Vision. Use the Meta block "
            "(`roboflow_core/meta_vlm@v3`) for Muse Spark and Muse Glimmer; it decodes "
            "detection and classification predictions in-block."
        ),
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": {
            "structured-answering": "roboflow_core/json_parser@v1",
            "classification": "roboflow_core/vlm_as_classifier@v2",
            "multi-label-classification": "roboflow_core/vlm_as_classifier@v2",
        },
    }
    outputs = {
        "output": Output(
            STRING_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            source="images",
            description="The model's raw answer to this image.",
        ),
        "classes": Output(
            LIST_OF_VALUES_KIND, source="images", description="The step's classes."
        ),
    }

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to ask about; one request per image."
        )
        task_type: OpenRouterTask = Field(
            default="unconstrained",
            description="What to ask: a free prompt, OCR, captions, classification, structured answers or detection.",
            examples=["object-detection"],
        )
        prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="Prompt for 'unconstrained' and question for 'visual-question-answering'.",
            examples=["my prompt", "$inputs.prompt"],
        )
        output_structure: Optional[Dict[str, StrictStr]] = Field(
            default=None,
            description="For 'structured-answering': answer field names and their descriptions.",
            examples=[{"my_key": "description"}],
        )
        classes: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Class names for classification and detection.",
            examples=[["class-a", "class-b"], "$inputs.classes"],
        )
        api_key: ApiKey = Field(
            default=DEFAULT_MANAGED_KEY,
            description=(
                "OpenRouter key as a secret or string, or 'rf_key:account' / "
                "'rf_key:user:ID' to use a key stored with Roboflow."
            ),
            examples=["rf_key:account", "sk-or-...", "$inputs.openrouter_api_key"],
        )
        privacy_level: PrivacyLevel = Field(
            default="deny",
            description=(
                "Provider privacy filter. Stricter levels reduce the pool of "
                "providers and may increase per-call cost on the managed key."
            ),
            examples=["deny"],
        )
        max_concurrent_requests: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description="Most requests in flight for one batch; None means 8.",
            examples=[4],
        )
        max_tokens: StrictInt = Field(
            default=500,
            gt=1,
            description="Most output tokens the model may generate.",
            examples=[500],
        )
        temperature: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.1,
            ge=0.0,
            le=2.0,
            description="Sampling temperature.",
            examples=[0.1],
        )
        model_version: Literal[tuple(LLAMA_MODELS)] | Ref(STRING_KIND) = Field(
            default="11B - OpenRouter",
            description="Llama 3.2 Vision model to run.",
            examples=["11B - OpenRouter", "$inputs.llama_model"],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "LlamaVisionBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )

            return self

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the OpenRouter model the step calls; a selector as is."""
        model_id = params.model_version
        if model_id in LLAMA_MODELS:
            model_id = LLAMA_MODELS[model_id].slug

        resources = [
            third_party_model(
                model_id, provider="openrouter", resource="openrouter_http_client"
            )
        ]

        return resources

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[Any]:
        """Report remote inference and image encoding."""
        operations = list(WORK_OPERATIONS)

        return operations

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> List[Any]:
        """Declare that the step has no runtime restrictions."""
        return []

    def run(
        self,
        *,
        images: Batch,
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: Any,
        privacy_level: str,
        max_concurrent_requests: Optional[int],
        max_tokens: int,
        temperature: float,
        model_version: str,
    ) -> List[Dict[str, Any]]:
        """Send one request per image; return the raw answers.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.
            max_tokens: Output token limit.
            temperature: Sampling temperature.
            model_version: Llama model label.

        Returns:
            One ``{"output", "classes"}`` per image.

        Raises:
            ValueError: On a missing key, an unknown model, a missing task
                field or invalid classes; nothing is sent.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails.
        """
        if api_key is None:
            # The OpenAI SDK would fall back to the OPENAI_API_KEY environment
            # variable and send that key to OpenRouter.
            raise ValueError(
                "`api_key` resolved to None; give an OpenRouter key or "
                "'rf_key:account'"
            )

        model = resolve_model(
            model_version, models=LLAMA_MODELS, family="Llama Vision model"
        )
        check_task_fields(
            task_type,
            prompt=prompt,
            classes=classes,
            output_structure=output_structure,
        )
        if task_type in DECODING_TASKS:
            # Validates only; prompts and outputs use classes as given.
            checked_classes(classes)

        prompts = [
            openrouter_task_messages(
                task_type,
                image=image,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
                detection_format=model.box_format,
            )
            for image in images
        ]
        answers = self.ask_all(
            prompts,
            api_key=api_key,
            model=model.slug,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            reasoning_effort=None,
            max_concurrent_requests=max_concurrent_requests,
        )

        results = [{"output": answer.text, "classes": classes} for answer in answers]

        return results
