"""Ask Qwen3-VL a question about each image.

Canonical type: ``roboflow_core/qwen3vl@v1``. The model, system prompt and
``max_new_tokens`` are one value per call; the prompt may differ per image.
The backend builds one conversation with one image slot, so the block makes
one ``prompt`` call per image, as V1 does, after every check has passed.
"""

import uuid
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Field, StrictInt, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._local_vlm import per_row
from roboflow_workflows.v2.blocks.models._shared.vlm import (
    DEFAULT_PROMPT,
    SYSTEM_PROMPT_MARKER,
    prompt_with_system_prompt,
    single_text_answer,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    model_request,
    models_cache,
    one_per_call,
    require_rgb,
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

DEFAULT_MODEL_ID = "qwen3vl-2b-instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are a Qwen3-VL model that can answer questions about any image."
)


class Qwen3VLBlock(Block):
    """Ask Qwen3-VL a question about each image."""

    type = "roboflow_core/qwen3vl@v1"
    metadata = {"name": "Qwen3-VL", "port_of": "roboflow_core/qwen3vl@v1"}
    default_system_prompt = DEFAULT_SYSTEM_PROMPT
    outputs = {
        "parsed_output": Output(
            STRING_KIND,
            source="image",
            description="The model's answer to this image's prompt, as text.",
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
            description="RGB images to ask about; one model call per image."
        )
        prompt: Optional[StrictStr] | Ref(
            STRING_KIND, batch="if_varying", allow_none=True
        ) = Field(
            default=None,
            description=(
                f"Question or instruction for each image. None or empty: "
                f"{DEFAULT_PROMPT!r}. A selector may choose it per run or per image."
            ),
            examples=["What is in this image?", "$inputs.prompt"],
        )
        max_new_tokens: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            ge=1,
            description=(
                "Maximum number of generated tokens per image. None leaves "
                "max_new_tokens out of the model's prompt() call, so the backend's "
                "prompt() default applies: 512, fixed."
            ),
            examples=[256, "$inputs.max_new_tokens"],
        )
        system_prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "System prompt for every image of the step. None or empty: "
                f"{DEFAULT_SYSTEM_PROMPT!r}. A selector may choose it per run, not "
                f"per image. Must not contain {SYSTEM_PROMPT_MARKER!r}."
            ),
            examples=["You are a helpful assistant.", "$inputs.system_prompt"],
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
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
        model_id: str,
        model_package_id: Optional[str] = None,
        prompt: Union[None, str, Batch] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Ask every image of the batch its prompt, one model call per image.

        Args:
            image: The batch of RGB images.
            model_id: Model id for every image.
            model_package_id: Package id for every image, or ``None``.
            prompt: Prompt for every image, a ``Batch`` with one per image, or
                ``None`` for the default prompt.
            system_prompt: System prompt for every image, or ``None`` for the
                default one.
            max_new_tokens: Generation limit; ``None`` leaves the keyword out
                of the ``prompt`` call, so the backend's default applies.

        Returns:
            One ``{"parsed_output", "model_id", "inference_id"}`` per image, in
            batch order.

        Raises:
            ValueError: Before any model loads, on a non-RGB image, a ``Batch``
                for a one-per-call field, a prompt batch whose length or
                logical indices differ from the image batch's, or a prompt
                containing the system prompt marker. After it, on a model
                returning other than one text answer.
        """
        for member in image:
            require_rgb(member, block=self.type)
        one_per_call(system_prompt, name="system_prompt", block=self.type)
        one_per_call(max_new_tokens, name="max_new_tokens", block=self.type)
        prompts = tuple(
            prompt_with_system_prompt(
                row_prompt,
                system_prompt,
                default_system_prompt=self.default_system_prompt,
                family="Qwen3-VL",
                block=self.type,
            )
            for row_prompt in per_row(prompt, image=image, name="prompt")
        )
        request = model_request(model_id, model_package_id, block=self.type)
        # An empty batch loads no model.
        if not len(image):
            return []

        model = self._models.get(request)
        options = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}

        results = []
        for member, row_prompt in zip(image, prompts):
            answers = model.prompt(
                [member.tensor_image],
                prompt=row_prompt,
                input_color_format="rgb",
                **options,
            )
            results.append(
                {
                    "parsed_output": single_text_answer(
                        answers, block=self.type, request=request
                    ),
                    "model_id": request.model_id,
                    "inference_id": uuid.uuid4().hex,
                }
            )

        return results
