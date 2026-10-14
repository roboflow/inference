"""Ask Meta Muse vision models about each image.

Canonical type: ``roboflow_core/meta_vlm@v3``.

Constructor resources come from ``OpenRouterBlockBase`` (``openrouter/v3.py``):
``platform_client`` and ``api_key`` (the host's Roboflow key, used for managed
``rf_key:`` keys), ``openrouter_base_url`` and ``openrouter_http_client``. Its
``ask_all`` owns key handling, the Roboflow proxy, privacy routing, the
reasoning retry and request concurrency; this module checks the inputs, builds
the messages and reads the answers.
"""

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.language._shared.openrouter_vendors import (
    DECODING_TASKS,
    VendorModel,
    encode_capped_jpeg,
    image_first_task_messages,
    resolve_model,
)
from roboflow_workflows.v2.blocks.language.chat import (
    DEFAULT_MANAGED_KEY,
    TASK_METADATA,
    WORK_OPERATIONS,
    ApiKey,
    check_reasoning_effort,
    check_task_fields,
    third_party_model,
    token_outputs,
)
from roboflow_workflows.v2.blocks.language.decoding import (
    checked_classes,
    decode_vlm_output,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.blocks.language.openrouter.v3 import (
    OPENROUTER_TASKS,
    PRIVACY_LEVEL_METADATA,
    OpenRouterBlockBase,
    OpenRouterTask,
    PrivacyLevel,
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

MUSE_SPARK_LEVELS = ("minimal", "low", "medium", "high", "xhigh")

META_MODELS: Dict[str, VendorModel] = {
    "Muse Spark 1.1": VendorModel(
        "meta/muse-spark-1.1",
        box_format="named_0_1000",
        reasoning_levels=MUSE_SPARK_LEVELS,
    ),
    "Muse Spark 1.2": VendorModel(
        "meta/muse-spark-1.2",
        box_format="named_0_1000",
        reasoning_levels=MUSE_SPARK_LEVELS,
    ),
    "Muse Spark 1.3": VendorModel(
        "meta/muse-spark-1.3",
        box_format="named_0_1000",
        reasoning_levels=MUSE_SPARK_LEVELS,
    ),
    "Muse Glimmer": VendorModel(
        "meta/muse-glimmer-30b",
        box_format="named_0_1000",
        reasoning_levels=("low", "medium", "high", "xhigh"),
    ),
}

MUSE_REASONING_METADATA = {
    "minimal": {
        "name": "Minimal",
        "description": (
            "Shortest reasoning pass. Supported by Muse Spark models only — "
            "Muse Glimmer starts at low."
        ),
    },
    "low": {
        "name": "Low (recommended)",
        "description": (
            "Small reasoning budget. Muse models require reasoning; this is "
            "the lowest effort every variant accepts."
        ),
    },
    "medium": {
        "name": "Medium",
        "description": "Moderate reasoning budget before answering.",
    },
    "high": {
        "name": "High",
        "description": (
            "Large reasoning budget. Slow and expensive; consider raising "
            "`max_tokens` so reasoning does not crowd out the answer."
        ),
    },
    "xhigh": {
        "name": "Extra high",
        "description": (
            "Maximum reasoning depth. Slowest and most expensive; raise "
            "`max_tokens` accordingly."
        ),
    },
}


class MetaVlmBlock(OpenRouterBlockBase):
    """Ask Meta Muse vision models about each image.

    Example: ``model_version="Muse Glimmer"``, ``task_type="object-detection"``
    and ``classes=["car"]`` asks for 0-1000 ``x_min``/``y_min``/``x_max``/
    ``y_max`` boxes and returns them as detections in image pixels. Muse
    always reasons; ``reasoning_effort`` defaults to ``low``.
    """

    type = "roboflow_core/meta_vlm@v3"
    metadata = {
        "name": "Meta",
        "port_of": "roboflow_core/meta_vlm@v3",
        "section": "model",
        "license": "Apache-2.0",
        "task_metadata": {task: TASK_METADATA[task] for task in OPENROUTER_TASKS},
        "model_version_metadata": {
            label: {"name": label, "reasoning_levels": list(model.reasoning_levels)}
            for label, model in META_MODELS.items()
        },
        "reasoning_effort_metadata": MUSE_REASONING_METADATA,
        "privacy_level_metadata": PRIVACY_LEVEL_METADATA,
        "recommended_parsers": {"structured-answering": "roboflow_core/json_parser@v1"},
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
        "thinking": Output(
            STRING_KIND,
            source="images",
            description="Reasoning trace when OpenRouter returns one, else empty.",
        ),
        **token_outputs(source="images"),
        **vlm_prediction_outputs("unconstrained", source="images"),
    }
    output_fields = ("task_type",)

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
        max_tokens: Optional[StrictInt] = Field(
            default=2048,
            gt=1,
            description=(
                "Most output tokens; None means 2048. Raise it (e.g. 8192) for long "
                "answers; billing counts generated tokens only."
            ),
            examples=[2048],
        )
        temperature: Optional[StrictFloat] | Ref(FLOAT_KIND, allow_none=True) = Field(
            default=None,
            ge=0.0,
            le=2.0,
            description="Sampling temperature; None keeps the provider default.",
            examples=[0.2],
        )
        model_version: Literal[tuple(META_MODELS)] | Ref(STRING_KIND) = Field(
            default="Muse Spark 1.3",
            description=(
                "Muse model to run. Spark 1.1, Spark 1.2, Spark 1.3, and Glimmer "
                "are the current image-capable Muse chat models on OpenRouter."
            ),
            examples=["Muse Spark 1.3", "Muse Glimmer"],
        )
        reasoning_effort: Literal[MUSE_SPARK_LEVELS] = Field(
            default="low",
            description=(
                "Reasoning budget, always sent: Muse rejects disabled reasoning. "
                "'minimal' is Spark-only."
            ),
            examples=["low"],
        )

        @model_validator(mode="after")
        def _task_fields_present(self) -> "MetaVlmBlock.Params":
            check_task_fields(
                self.task_type,
                prompt=self.prompt,
                classes=self.classes,
                output_structure=self.output_structure,
            )

            return self

        @model_validator(mode="after")
        def _reasoning_supported(self) -> "MetaVlmBlock.Params":
            levels = {
                label: model.reasoning_levels for label, model in META_MODELS.items()
            }
            check_reasoning_effort(
                self.model_version, self.reasoning_effort, levels_by_model=levels
            )

            return self

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Dict[str, Output]:
        """Narrow ``predictions`` to the kind the configured task decodes."""
        outputs = {
            **cls.outputs,
            **vlm_prediction_outputs(params.task_type, source="images"),
        }

        return outputs

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the OpenRouter model the step calls; a selector as is."""
        model_id = params.model_version
        if model_id in META_MODELS:
            model_id = META_MODELS[model_id].slug

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
        max_tokens: Optional[int],
        temperature: Optional[float],
        model_version: str,
        reasoning_effort: str,
    ) -> List[Dict[str, Any]]:
        """Send one image-first request per image and decode each answer.

        Args:
            images: The batch of images.
            task_type: What to ask.
            prompt: Prompt or question.
            output_structure: Fields of a structured answer.
            classes: Class names.
            api_key: OpenRouter key, secret or managed ``rf_key:`` reference.
            privacy_level: ``allow``, ``deny`` or ``zdr``.
            max_concurrent_requests: Requests in flight; ``None`` means 8.
            max_tokens: Output token limit; ``None`` means 2048.
            temperature: Sampling temperature; ``None`` keeps the provider's.
            model_version: Muse model label.
            reasoning_effort: Reasoning level the model accepts.

        Returns:
            One ``{"output", "classes", "thinking", "input_tokens",
            "output_tokens", "predictions", "error_status", "inference_id"}``
            per image.

        Raises:
            ValueError: On a missing key, an unknown model, a rejected
                reasoning level, a missing task field or invalid classes;
                nothing is sent.
            SecretUnavailableError: When a key was read back from a recording.
            ChatRequestError: When a request fails; nothing is decoded.
        """
        if api_key is None:
            # The OpenAI SDK would fall back to the OPENAI_API_KEY environment
            # variable and send that key to OpenRouter.
            raise ValueError(
                "`api_key` resolved to None; give an OpenRouter key or "
                "'rf_key:account'"
            )

        model = resolve_model(model_version, models=META_MODELS, family="Muse variant")
        check_reasoning_effort(
            model_version,
            reasoning_effort,
            levels_by_model={model_version: model.reasoning_levels},
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
            image_first_task_messages(
                task_type,
                upload=encode_capped_jpeg(image),
                model=model,
                prompt=prompt,
                output_structure=output_structure,
                classes=classes,
            )
            for image in images
        ]
        answers = self.ask_all(
            prompts,
            api_key=api_key,
            model=model.slug,
            max_tokens=2048 if max_tokens is None else max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            reasoning_effort=reasoning_effort,
            max_concurrent_requests=max_concurrent_requests,
        )

        results = []
        for image, answer in zip(images, answers):
            inference_id = str(uuid.uuid4())
            error_status, predictions = decode_vlm_output(
                task_type,
                answer.text,
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format=model.box_format,
            )
            results.append(
                {
                    "output": answer.text,
                    "classes": classes,
                    "thinking": answer.thinking,
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "predictions": predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )

        return results
