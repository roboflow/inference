"""Local Cosmos 3 Edge scene reasoning with V1 prompt composition.

The optional system prompt follows <system_prompt> only when explicitly set;
unset keeps the backend's base or fine-tune system default. Every image is a
separate conversation, in input order, with the call's one model and system
prompt. Roboflow hosted execution awaits the new server contract.
"""

import uuid
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Field, StrictInt, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._local_vlm import (
    check_backend_call,
    checked_text,
    generation_options,
    image_row,
    per_row,
    require_rgb_images,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    model_request,
    models_cache,
    one_per_call,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

DEFAULT_MODEL_ID = "nvidia/cosmos-3-edge"

DEFAULT_PROMPT = "Describe what's in this image."

SYSTEM_PROMPT_MARKER = "<system_prompt>"


def _checked_instruction(text: Any, *, name: str) -> Optional[str]:
    text = checked_text(text, name=name)
    if text and SYSTEM_PROMPT_MARKER in text:
        raise ValueError(
            f"{name} contains {SYSTEM_PROMPT_MARKER!r}, which Cosmos reads as "
            "the start of the system prompt; remove it from the text"
        )

    return text


def _combine_prompt(prompt: Any, system_prompt: Optional[str]) -> str:
    # system_prompt was checked once for the call.
    prompt = _checked_instruction(prompt, name="prompt") or DEFAULT_PROMPT
    if system_prompt:
        prompt = f"{prompt}{SYSTEM_PROMPT_MARKER}{system_prompt}"

    return prompt


class Cosmos3EdgeBlock(Block):
    """Ask Cosmos 3 Edge about each image's physical scene."""

    type = "roboflow_core/cosmos3_edge@v1"
    metadata = {"name": "Cosmos 3", "port_of": "roboflow_core/cosmos3_edge@v1"}
    outputs = {
        # Temporary STRING_KIND convention, matching pinned Q3 c2.
        "output": Output(
            STRING_KIND, source="image", description="Text answer for this image."
        ),
        "model_id": Output(
            STRING_KIND, source="image", description="Model id that served this image."
        ),
        "inference_id": Output(
            STRING_KIND,
            source="image",
            description=(
                "Fresh id of this image's result; for results with rows, also the "
                "namespace of their row identity."
            ),
        ),
    }

    class Params(BlockParams):
        """Image, local request and optional user/system instructions."""

        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=DEFAULT_MODEL_ID,
            description=(
                "Model id (or local path) of the model in the 'models' resource, "
                "one for every image of the step; a selector may choose it per "
                "run, not per image. To serve images with different models, "
                "route them to one step per model."
            ),
            examples=[DEFAULT_MODEL_ID, "$inputs.model_id"],
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Exact model package to load, one for every image of the step; "
                    "None lets the loader choose. A selector may choose it per run, "
                    "not per image. A model supplied without a package id rejects a "
                    "package. List a model's package ids with "
                    'inference_models.AutoModel.describe_model("<model_id>").'
                ),
                examples=[None, "$inputs.model_package_id"],
            )
        )
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images; each starts a separate conversation."
        )
        prompt: Optional[StrictStr] | Ref(
            STRING_KIND, batch="if_varying", allow_none=True
        ) = Field(
            default=None,
            description=f"User question; omitted, None or empty uses {DEFAULT_PROMPT!r}. Reserved marker is rejected.",
            examples=["Is the walkway clear?", "$inputs.prompt"],
        )
        system_prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="System instruction for every image of the step; a selector may choose it per run, not per image. Omitted, None or empty preserves backend/fine-tune default. Reserved marker is rejected.",
            examples=["You are a safety inspector.", "$inputs.system_prompt"],
        )
        max_new_tokens: Optional[StrictInt] | Ref(INTEGER_KIND, allow_none=True) = (
            Field(
                default=None,
                ge=1,
                description="Positive generation limit; omitted or None leaves prompt()'s backend default.",
                examples=[256, "$inputs.tokens"],
            )
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        """Bind the host-owned models resource.

        Args:
            models: Loaded model mapping or shared ModelCache; nothing loads here.
        """
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Describe the local model request.

        Args:
            params: Validated step parameters.

        Returns:
            Model resource declaration, including any selected package.
        """
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": "vlm",
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    def run(
        self,
        *,
        image: Batch,
        model_id: str = DEFAULT_MODEL_ID,
        model_package_id: Optional[str] = None,
        prompt: Union[None, str, Batch] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Answer each image independently, preserving input order.

        Args:
            image: RGB images as a Batch.
            model_id: Model id for every image.
            model_package_id: Exact package for every image, or None.
            prompt: User question, per-image questions, or None for the V1 default.
            system_prompt: System instruction for every image, or None.
            max_new_tokens: Positive limit or None to use the backend default.

        Returns:
            Text output, model id and fresh inference id for every image.

        Raises:
            ValueError: On invalid parameters or reserved markers (naming the
                image's logical index), a per-image Batch whose indices differ
                from the image batch's, a backend whose prompt() cannot take the
                call, or non-single-text answers.
            LookupError: When the host cannot serve the exact model request.
        """
        require_rgb_images(image, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        one_per_call(system_prompt, name="system_prompt", block=self.type)
        try:
            system_prompt = _checked_instruction(system_prompt, name="system_prompt")
        except ValueError as error:
            raise ValueError(f"{self.type}: {error}") from error
        prompts = []
        for index, text in zip(
            image.indices, per_row(prompt, image=image, name="prompt")
        ):
            with image_row(index, block=self.type):
                prompts.append(_combine_prompt(text, system_prompt))
        options = generation_options(max_new_tokens)
        # An empty batch loads no model.
        if not len(image):
            return []

        model = self._models.get(request)
        # Bind the backend signature before the first model invocation.
        check_backend_call(
            model,
            "prompt",
            block=self.type,
            request=request,
            options=options,
            prompt="",
            input_color_format="rgb",
        )

        results = []
        for member, text in zip(image, prompts):
            answers = model.prompt(
                [member.tensor_image], prompt=text, input_color_format="rgb", **options
            )
            if (
                not isinstance(answers, list)
                or len(answers) != 1
                or not isinstance(answers[0], str)
            ):
                raise ValueError(
                    f"{self.type}: model {str(request)!r} must return a list of one text answer"
                )
            results.append(
                {
                    "output": answers[0],
                    "model_id": request.model_id,
                    "inference_id": uuid.uuid4().hex,
                }
            )

        return results


BLOCKS = [Cosmos3EdgeBlock]
