"""Local Cosmos 3 Edge scene reasoning with V1 prompt composition.

The optional system prompt follows <system_prompt> only when explicitly set.
Leaving it unset preserves the backend's base or fine-tune system default.
Every image is a separate conversation, in input order; RGB tensors retain
their identity/device. Per-image values must carry the image batch's indices
(see _local_vlm). Models belong to the host and may be reused by cache.
Roboflow hosted execution awaits the new server contract.
"""

import uuid
from typing import Any, Dict, List, Mapping, Union

from pydantic import Field
from roboflow_workflows.v2.blocks.models._local_vlm import (
    TextField,
    TextParam,
    TokensField,
    TokensParam,
    _check_backend_call,
    _image_row,
    _model_requests,
    _require_rgb,
    checked_text,
    generation_options,
    per_row,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    INFERENCE_ID_OUTPUT,
    MODEL_ID_DESCRIPTION,
    MODEL_ID_OUTPUT,
    ModelIdField,
    ModelSelectionParams,
    RequestField,
    model_dependency,
    models_cache,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
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


def _combine_prompt(prompt: Any, system_prompt: Any) -> str:
    prompt = checked_text(prompt, name="prompt") or DEFAULT_PROMPT
    system_prompt = checked_text(system_prompt, name="system_prompt")
    for name, text in (("prompt", prompt), ("system_prompt", system_prompt)):
        if text and SYSTEM_PROMPT_MARKER in text:
            raise ValueError(
                f"{name} contains {SYSTEM_PROMPT_MARKER!r}, which Cosmos reads as "
                "the start of the system prompt; remove it from the text"
            )

    if system_prompt:
        prompt = f"{prompt}{SYSTEM_PROMPT_MARKER}{system_prompt}"

    return prompt


class Cosmos3EdgeBlock(Block):
    """Ask Cosmos 3 Edge about each image's physical scene."""

    type = "v2/cosmos3_edge@v1"
    metadata = {"name": "Cosmos 3", "port_of": "roboflow_core/cosmos3_edge@v1"}
    outputs = {
        # Temporary STRING_KIND convention, matching pinned Q3 c2.
        "output": Output(
            STRING_KIND, source="image", description="Text answer for this image."
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(ModelSelectionParams):
        """Image, local request and optional user/system instructions."""

        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images; each starts a separate conversation."
        )
        model_id: ModelIdField = Field(
            default=DEFAULT_MODEL_ID,
            description=MODEL_ID_DESCRIPTION,
            examples=[DEFAULT_MODEL_ID, "$inputs.model_id"],
        )
        prompt: TextParam = Field(
            default=None,
            description=f"User question; omitted, None or empty uses {DEFAULT_PROMPT!r}. Reserved marker is rejected.",
            examples=["Is the walkway clear?", "$inputs.prompt"],
        )
        system_prompt: TextParam = Field(
            default=None,
            description="System instruction; omitted, None or empty preserves backend/fine-tune default. Reserved marker is rejected.",
            examples=["You are a safety inspector.", "$inputs.system_prompt"],
        )
        max_new_tokens: TokensParam = Field(
            default=None,
            ge=1,
            description="Positive generation limit; omitted or None leaves prompt()'s backend default.",
            examples=[256, "$inputs.tokens"],
        )

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
        resources = model_dependency(params, task_type="vlm")

        return resources

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField = DEFAULT_MODEL_ID,
        model_package_id: RequestField = None,
        prompt: TextField = None,
        system_prompt: TextField = None,
        max_new_tokens: TokensField = None,
    ) -> List[Dict[str, Any]]:
        """Answer each image independently, preserving input order.

        Args:
            image: RGB images as a Batch.
            model_id: Model id, or a Batch with one id per image.
            model_package_id: Exact package, per-image packages, or None.
            prompt: User question, per-image questions, or None for the V1 default.
            system_prompt: System instruction, per-image instructions, or None.
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
        _require_rgb(image, block=self.type)
        requests = _model_requests(
            image,
            model_id=model_id,
            model_package_id=model_package_id,
            block=self.type,
        )
        prompts = []
        for index, text, system in zip(
            image.indices,
            per_row(prompt, image=image, name="prompt"),
            per_row(system_prompt, image=image, name="system_prompt"),
        ):
            with _image_row(index, block=self.type):
                prompts.append(_combine_prompt(text, system))
        options = generation_options(max_new_tokens)
        models = {
            request: self._models.get(request) for request in dict.fromkeys(requests)
        }
        # Bind every backend signature before the first model invocation.
        for request, model in models.items():
            _check_backend_call(
                model,
                "prompt",
                block=self.type,
                request=request,
                options=options,
                prompt="",
                input_color_format="rgb",
            )

        results = []
        for member, request, text in zip(image, requests, prompts):
            answers = models[request].prompt(
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
