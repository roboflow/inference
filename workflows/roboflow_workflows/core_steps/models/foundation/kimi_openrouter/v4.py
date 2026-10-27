"""Kimi v4 with model-specific reasoning and normalized XYXY detection."""

import json
from functools import partial
from typing import Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

from openai import OpenAI
from pydantic import ConfigDict, Field, model_validator
from roboflow_workflows.core_steps.common.openrouter import (
    _PROXY_ERROR_HANDLERS,
    PROXY_MAX_TOKENS_CEILING,
    RELEVANT_TASKS_METADATA,
    SUPPORTED_TASK_TYPES_LIST,
    OpenRouterBlockManifestMixin,
    OpenRouterWorkflowBlockBase,
    build_prompts_from_images,
    build_provider_routing,
    validate_task_type_required_fields,
)
from roboflow_workflows.core_steps.common.utils import run_in_parallel
from roboflow_workflows.core_steps.common.vlm_decoding import (
    actual_vlm_prediction_outputs,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
    extract_json,
    get_detection_box_format,
)
from roboflow_workflows.environment import (
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlockManifest,
    is_workflow_selector,
    third_party_model,
)

MODEL_VARIANTS = {
    "Kimi K2.6 - OpenRouter": {
        "model_id": "moonshotai/kimi-k2.6",
        "reasoning_levels": [],
        "reasoning_toggle": True,
    },
    "Kimi K3.1 Pro - OpenRouter": {
        "model_id": "moonshotai/kimi-k3.1-pro",
        "reasoning_levels": ["none", "low", "high", "max"],
        "reasoning_toggle": False,
    },
}
MODEL_VERSION_MAPPING = {
    label: variant["model_id"] for label, variant in MODEL_VARIANTS.items()
}
MODEL_VERSION_METADATA = {
    label: {"name": label, **variant} for label, variant in MODEL_VARIANTS.items()
}
ModelVersion = Literal[tuple(MODEL_VARIANTS)]

TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

# Keep the prompt and decoder on the same XYXY 0–1000 contract.
DETECTION_BOX_FORMAT = "xyxy_0_1000"
DETECTION_PROMPT = (
    "Detect all instances of the specified object classes in this image. "
    'Output a JSON list. Each detection must contain "label", "box_2d", and "confidence". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: '
    "the top-left and bottom-right corners as integers between 0 and 1000, "
    "normalized to the image width (x) and height (y). "
    "Confidence is a number between 0 and 1. Return only the JSON, with no extra text. "
    "Only use these labels: {class_list}"
)


# Detections and classifications are decoded in-block from this version on, so
# only the structured-answering parser stays relevant.
RECOMMENDED_PARSERS = {
    "structured-answering": "roboflow_core/json_parser@v1",
}

RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)

LONG_DESCRIPTION = f"""
Ask a question to Moonshot AI Kimi vision-language models served via OpenRouter.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

#### 🛠️ API providers and model variants

Kimi is exposed via [OpenRouter](https://openrouter.ai/). By default this block uses
the **Roboflow-managed OpenRouter key** and bills your Roboflow credits — no extra
setup needed. To bypass Roboflow billing, paste your own `sk-or-...` key into the
`api_key` field.

The `privacy_level` field controls which OpenRouter providers may serve the request:

* **No data collection** *(default)* – providers may not train on your inputs.
* **Allow data collection** – broader provider pool.
* **Zero data retention** – strictest, restricts to providers that retain nothing.

#### 💡 Further reading and Acceptable Use Policy

!!! warning "Model license"

    Check the [Moonshot AI Kimi license terms](https://huggingface.co/moonshotai) before use.

## Version Differences

This version (v4) adds Kimi K3.1 Pro as the default and retains Kimi K2.6.
Kimi K2.5 remains available in earlier block versions; it is omitted here because
OpenRouter no longer lists an official Moonshot route, not because the model is
globally discontinued.

Object detection requests a JSON list of `label`, `box_2d` in
`[x_min, y_min, x_max, y_max]` order normalized to 0–1000, and `confidence`.
This coordinate contract scored highest for both models in our 100-image format
comparison. Small differences between the leading formats remain uncertain.

Kimi K2.6 uses `reasoning_enabled` (on/off). Kimi K3.1 Pro uses
`reasoning_effort`: `none`, `low`, `high`, or `max` (default `high`).
Unsupported reasoning is reported as an error rather than retried without it.

Requests are pinned to the official Moonshot AI provider, without fallback.
Managed keys require a Roboflow proxy that forwards provider routing and reasoning.

Decoded detections follow the shared block convention: unknown labels are kept
with `class_id=-1` (the benchmark instead excluded those labels from scoring),
and out-of-range coordinates are clamped. Model confidence is preserved.

The output contract is unchanged from v3:

* **`predictions`** - classification and object-detection answers are parsed here,
  so the deprecated `VLM as Detector` / `VLM as Classifier` blocks are no longer
  needed. The output kind follows `task_type`: object detection predictions for
  `object-detection`, classification predictions for `classification` and
  `multi-label-classification`, and `None` for every other task.
* **`error_status`** - `True` when the model answer could not be parsed.
* **`inference_id`** - identifier generated per image and attached to the decoded
  predictions.
"""


def _base_outputs() -> List[OutputDefinition]:
    """Outputs the block produces regardless of the selected task."""
    return [
        OutputDefinition(name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]),
        OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
    ]


class BlockManifest(OpenRouterBlockManifestMixin):
    """Configure Kimi model, reasoning mode and task inputs."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "MoonshotAI Kimi",
            "version": "v4",
            "short_description": "Run Moonshot AI Kimi vision-language models via OpenRouter.",
            "long_description": LONG_DESCRIPTION,
            "license": "Moonshot AI Kimi License",
            "block_type": "model",
            "search_keywords": ["LMM", "VLM", "Kimi", "Moonshot", "OpenRouter"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/kimi_openrouter@v4"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
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
        description="Text prompt to the Kimi model",
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
        default="Kimi K3.1 Pro - OpenRouter",
        description="Model to be used",
        examples=["Kimi K3.1 Pro - OpenRouter", "$inputs.kimi_model"],
        json_schema_extra={"values_metadata": MODEL_VERSION_METADATA},
    )

    temperature: Literal[None] = Field(
        default=None,
        description="Use the official provider's model-specific sampling temperature. Kimi v4 does not send a temperature override.",
    )
    max_tokens: int = Field(
        default=16384,
        gt=1,
        description="Combined reasoning and answer token limit. Increase for dense detection outputs; the managed proxy accepts at most 16384.",
    )
    reasoning_enabled: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=True,
        description="Enable reasoning for Kimi K2.6. This model supports an on/off toggle, not effort levels.",
        json_schema_extra={
            "relevant_for": {"model_version": {"values": ["Kimi K2.6 - OpenRouter"]}}
        },
    )
    reasoning_effort: Union[
        Selector(kind=[STRING_KIND]), Literal["none", "low", "high", "max"]
    ] = Field(
        default="high",
        description="Reasoning effort for Kimi K3.1 Pro: none disables reasoning; low, high and max set its effort.",
        json_schema_extra={
            "relevant_for": {
                "model_version": {"values": ["Kimi K3.1 Pro - OpenRouter"]}
            }
        },
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        """Validate task-dependent inputs.

        Returns:
            The validated manifest.

        Raises:
            ValueError: If required task inputs are missing.
        """
        validate_task_type_required_fields(
            task_type=self.task_type,
            prompt=self.prompt,
            classes=self.classes,
            output_structure=self.output_structure,
        )
        return self

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        """Describe connectivity requirements.

        Returns:
            Unavailable offline because Kimi requires an external API.
        """
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        """List batched inputs.

        Returns:
            The images parameter.
        """
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        """Describe all possible task outputs.

        Returns:
            Raw output and prediction definitions.
        """
        outputs = _base_outputs() + describe_vlm_prediction_outputs()
        return outputs

    def get_actual_outputs(self) -> List[OutputDefinition]:
        """Resolve output kinds for the selected task.

        Returns:
            Task-specific output definitions.
        """
        outputs = _base_outputs() + actual_vlm_prediction_outputs(self.task_type)
        return outputs

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        """Declare supported engine versions.

        Returns:
            The supported semantic version range.
        """
        return ">=1.3.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        """Resolve the selected OpenRouter model dependency.

        Returns:
            A model dependency, with deferred resolution for selectors.
        """
        if is_workflow_selector(self.model_version):
            # Friendly-label selector returned verbatim; the attached resolver
            # performs the MODEL_VERSION_MAPPING lookup once the input value
            # is substituted.
            return [
                third_party_model(
                    provider="openrouter",
                    model_id=self.model_version,
                    model_id_resolver=lambda label: MODEL_VERSION_MAPPING[label],
                )
            ]
        return [
            third_party_model(
                provider="openrouter",
                model_id=MODEL_VERSION_MAPPING[self.model_version],
            )
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        """Describe runtime operations.

        Returns:
            Inference, network access and image encoding operations.
        """
        return [
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.IMAGE_ENCODING,
        ]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Describe runtime restrictions.

        Args:
            ignore_environment_restrictions: Whether to ignore host restrictions.

        Returns:
            An empty, complete restriction set.
        """
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class KimiOpenrouterBlockV4(OpenRouterWorkflowBlockBase):
    """Run Kimi with explicit reasoning and decode predictions in-block."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        """Return the v4 input manifest.

        Returns:
            The manifest class.
        """
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        """Declare supported engine versions.

        Returns:
            The supported semantic version range.
        """
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        task_type: str,
        prompt: Optional[str],
        output_structure: Optional[Dict[str, str]],
        classes: Optional[List[str]],
        api_key: str,
        privacy_level: str,
        model_version: ModelVersion,
        max_tokens: int,
        temperature: Optional[float],
        max_concurrent_requests: Optional[int],
        reasoning_enabled: bool = True,
        reasoning_effort: str = "high",
    ) -> BlockResult:
        """Run the selected Kimi task for each input image.

        Args:
            images: Input workflow images.
            task_type: Built-in task or unconstrained prompting.
            prompt: Text for unconstrained or question-answering tasks.
            output_structure: Expected structured-answering fields.
            classes: Allowed detection or classification labels.
            api_key: Caller OpenRouter key or managed-key selector.
            privacy_level: OpenRouter data-collection policy.
            model_version: Supported friendly model label.
            max_tokens: Combined reasoning and answer token limit.
            temperature: Sampling temperature.
            max_concurrent_requests: Maximum simultaneous image requests.
            reasoning_enabled: K2.6 reasoning toggle.
            reasoning_effort: K3.1 Pro native reasoning effort.

        Returns:
            Raw answers and decoded predictions with error status and IDs.

        Raises:
            ValueError: If the resolved model or reasoning mode is unsupported.
            RuntimeError: If the provider returns no usable completion or the
                request is routed outside the required Moonshot provider.
        """
        if temperature is not None:
            raise ValueError(
                "Kimi v4 requires temperature=None to use provider defaults"
            )

        if model_version not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported Kimi model: {model_version}")

        if model_version == "Kimi K2.6 - OpenRouter":
            if not isinstance(reasoning_enabled, bool):
                raise ValueError("Kimi K2.6 reasoning_enabled must be a boolean")
        elif reasoning_effort not in MODEL_VARIANTS[model_version]["reasoning_levels"]:
            raise ValueError(
                f"Unsupported Kimi K3.1 Pro reasoning effort: {reasoning_effort}"
            )

        inference_images = [i.to_inference_format() for i in images]
        prompts = build_prompts_from_images(
            images=inference_images,
            task_type=task_type,
            prompt=prompt,
            output_structure=output_structure,
            classes=classes,
            detection_box_format=DETECTION_BOX_FORMAT,
        )
        reasoning = (
            {"enabled": reasoning_enabled}
            if model_version == "Kimi K2.6 - OpenRouter"
            else {"effort": reasoning_effort}
        )
        if task_type == "object-detection":
            for messages in prompts:
                messages[0]["content"][0]["text"] = DETECTION_PROMPT.format(
                    class_list=", ".join(classes)
                )

        raw_outputs = self._execute_kimi_batch(
            openrouter_api_key=api_key,
            model=MODEL_VERSION_MAPPING[model_version],
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            max_concurrent_requests=max_concurrent_requests,
            reasoning=reasoning,
        )
        predictions = []
        for image, raw_output in zip(images, raw_outputs):
            inference_id = str(uuid4())
            error_status, decoded_predictions = decode_vlm_output(
                task_type=task_type,
                raw_output=(
                    _detection_with_confidence(raw_output)
                    if task_type == "object-detection"
                    else raw_output
                ),
                image=image,
                classes=classes,
                inference_id=inference_id,
                box_format="named_normalized",
            )
            predictions.append(
                {
                    "output": raw_output,
                    "classes": classes,
                    "predictions": decoded_predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )
        return predictions

    def _execute_kimi_batch(
        self,
        *,
        openrouter_api_key,
        model,
        prompts,
        max_tokens,
        temperature,
        privacy_level,
        max_concurrent_requests,
        reasoning,
    ):
        # Keep strict reasoning scoped to v4: the shared legacy executor retries
        # unsupported settings without reasoning, which changes the chosen mode.
        request = partial(
            self._execute_kimi_request,
            openrouter_api_key=openrouter_api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            privacy_level=privacy_level,
            reasoning=reasoning,
        )
        outputs = run_in_parallel(
            tasks=[partial(request, messages=messages) for messages in prompts],
            max_workers=max_concurrent_requests
            or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        return outputs

    def _execute_kimi_request(
        self,
        *,
        openrouter_api_key,
        model,
        messages,
        max_tokens,
        temperature,
        privacy_level,
        reasoning,
    ):
        provider = build_provider_routing(privacy_level) or {}
        provider.update(
            only=["moonshotai" if model.endswith("k2.6") else "moonshotai/mxfp4"],
            allow_fallbacks=False,
            require_parameters=True,
        )
        request = dict(model=model, messages=messages, max_tokens=max_tokens)
        if temperature is not None:
            request["temperature"] = temperature

        if openrouter_api_key.startswith(("rf_key:account", "rf_key:user:")):
            request.update(
                openrouter_api_key=openrouter_api_key,
                privacy_level=privacy_level,
                reasoning=reasoning,
                provider=provider,
                max_tokens=(
                    max_tokens if max_tokens is not None else PROXY_MAX_TOKENS_CEILING
                ),
            )
            response = self._platform_client.post(
                endpoint="apiproxy/openrouter",
                api_key=self._roboflow_api_key,
                payload=request,
                http_errors_handlers=_PROXY_ERROR_HANDLERS,
            )
        else:
            with OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key
            ) as client:
                response = client.chat.completions.create(
                    **request,
                    extra_body={"reasoning": reasoning, "provider": provider},
                ).model_dump()

        choices = response.get("choices") or []
        if (
            not choices
            or choices[0].get("error")
            or choices[0].get("finish_reason") == "error"
        ):
            raise RuntimeError("Kimi provider failed to return a completion")

        if response.get("provider") != "Moonshot AI":
            raise RuntimeError(
                "Kimi request was not served by the required Moonshot AI provider; "
                "the Roboflow proxy must forward provider routing before this model can use a managed key"
            )

        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content:
            raise RuntimeError(
                "Kimi returned no answer; increase max_tokens if reasoning exhausted the output budget"
            )

        return content


def _detection_with_confidence(raw_output: str) -> str:
    # The legacy XYXY decoder intentionally replaces confidence with 1.0.
    # Reuse its coordinate converter, then the shared named decoder to retain
    # Kimi's requested confidence without changing any shipped format behavior.
    error, entries = extract_json(raw_output)
    if error or not isinstance(entries, list):
        return "invalid detection JSON"

    converter = get_detection_box_format(DETECTION_BOX_FORMAT)
    detections = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        box = converter.to_pixel_xyxy(entry=entry, image_width=1, image_height=1)
        if box is None:
            continue

        detections.append(
            dict(
                zip(("x_min", "y_min", "x_max", "y_max"), box),
                class_name=entry.get("label", ""),
                confidence=entry.get("confidence", 1.0),
            )
        )
    if entries and not detections:
        return "invalid detection boxes"

    normalized_output = json.dumps({"detections": detections})
    return normalized_output
