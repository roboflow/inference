"""Prompt local vision-language models (VLMs) with host-owned models.

Ports of the latest V1 local VLM blocks (Qwen2.5-VL, Qwen3-VL, Qwen3.5 v2,
SmolVLM2). Each step asks every image one text question and returns the
model's answer as text::

    run   check images, prompts and per-image requests
          resolve every row's model from the 'models' resource
          per image: model.prompt([tensor_image], prompt=..., **options)
          one {"parsed_output", "model_id", "inference_id"} per image

Every ``inference_models`` VLM builds one conversation with one image slot,
so the block makes one ``prompt`` call per image, as V1 does. All models are
resolved and all prompts checked before the first call. Images are passed as
their RGB ``(3, H, W)`` ``uint8`` tensors on their own device.

Qwen models read the system prompt from the text after a ``<system_prompt>``
marker in ``prompt``; the block appends it there, as V1 does. A prompt or
system prompt that already contains the marker would be split in the wrong
place, so it fails the step. SmolVLM2 has no system prompt.
"""

import uuid
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union

from pydantic import Field, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
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
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

TASK_TYPE = "vlm"
SYSTEM_PROMPT_MARKER = "<system_prompt>"
DEFAULT_PROMPT = "Describe what's in this image."
# V1 local defaults, always sent explicitly. Qwen3VLHF's own default differs
# ("... a helpful assistant for any visual task."); V1 never used it.
QWEN25_SYSTEM_PROMPT = (
    "You are a Qwen2.5-VL model that can answer questions about any image."
)
QWEN3_SYSTEM_PROMPT = (
    "You are a Qwen3-VL model that can answer questions about any image."
)
QWEN3_5_SYSTEM_PROMPT = "You are a helpful assistant."

# A literal, or a selector; text that varies per image arrives as a Batch.
TextParam = Optional[StrictStr] | Ref(STRING_KIND, batch="if_varying", allow_none=True)
TokensParam = Optional[StrictInt] | Ref(INTEGER_KIND)

# Provisional kind: plain text until V2 has a kind for language model answers
# (V1 declares LANGUAGE_MODEL_OUTPUT_KIND).
PARSED_OUTPUT = Output(
    STRING_KIND,
    source="image",
    description="The model's answer to this image's prompt, as text.",
)


def _max_new_tokens_field(backend_default: Optional[str] = None) -> Any:
    description = (
        "Maximum number of generated tokens per image. None leaves "
        "max_new_tokens out of the model's prompt() call, so the backend's "
        "prompt() default applies"
    )
    if backend_default is not None:
        description = f"{description}: {backend_default}"
    field = Field(
        default=None,
        ge=1,
        description=f"{description}.",
        examples=[256, "$inputs.max_new_tokens"],
    )

    return field


class VLMParams(ModelSelectionParams):
    """Image, prompt and generation length of every local VLM block."""

    # noqa F821: flake8 parses the string "always" in an annotation as a name.
    image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
        description="RGB images to ask about; one model call per image."
    )
    prompt: TextParam = Field(
        default=None,
        description=(
            f"Question or instruction for each image. None or empty: "
            f"{DEFAULT_PROMPT!r}. A selector may choose it per run or per image."
        ),
        examples=["What is in this image?", "$inputs.prompt"],
    )
    # Each block's Params repeats this field with its backend default.
    max_new_tokens: TokensParam = _max_new_tokens_field()


def _system_prompt_field(default_system_prompt: str) -> Any:
    field = Field(
        default=None,
        description=(
            "System prompt for each image. None or empty: "
            f"{default_system_prompt!r}. A selector may choose it per run or per "
            f"image. Must not contain {SYSTEM_PROMPT_MARKER!r}."
        ),
        examples=["You are a helpful assistant.", "$inputs.system_prompt"],
    )

    return field


def _model_id_field(default_model_id: str, *examples: str) -> Any:
    field = Field(
        default=default_model_id,
        description=MODEL_ID_DESCRIPTION,
        examples=[default_model_id, *examples, "$inputs.model_id"],
    )

    return field


class _LocalVLMBlock(Block):
    """Shared run of the local VLM blocks; subclasses set the family values."""

    family: ClassVar[str]
    # None: the model has no system prompt (SmolVLM2).
    default_system_prompt: ClassVar[Optional[str]] = None
    # Fixed prompt() keywords of the family, e.g. Qwen3.5's enable_thinking.
    fixed_options: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    outputs = {
        "parsed_output": PARSED_OUTPUT,
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=TASK_TYPE)

        return resources

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        model_package_id: RequestField = None,
        prompt: RequestField = None,
        system_prompt: RequestField = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Ask every image of the batch its prompt, one model call per image.

        Args:
            image: The batch of RGB images.
            model_id: Model id for every image, or a ``Batch`` with one per
                image.
            model_package_id: Package id for every image, a ``Batch`` with
                one per image, or ``None``.
            prompt: Prompt for every image, a ``Batch`` with one per image, or
                ``None`` for the default prompt.
            system_prompt: Like ``prompt``, for the system prompt; only for
                families that have one.
            max_new_tokens: Generation limit; ``None`` leaves the keyword out
                of the ``prompt`` call, so the backend's default applies.

        Returns:
            One ``{"parsed_output", "model_id", "inference_id"}`` per image, in
            batch order.

        Raises:
            ValueError: On a non-RGB image, a batch of the wrong length, a
                prompt containing the system prompt marker, a system prompt
                for a family without one, or a model returning other than one
                text answer.
        """
        row_count = len(image)
        for member in image:
            _require_rgb(member, block=self.type)
        model_ids = _per_row(model_id, row_count=row_count, name="model_id")
        package_ids = _per_row(
            model_package_id, row_count=row_count, name="model_package_id"
        )
        prompts = tuple(
            self._full_prompt(row_prompt, row_system_prompt)
            for row_prompt, row_system_prompt in zip(
                _per_row(prompt, row_count=row_count, name="prompt"),
                _per_row(system_prompt, row_count=row_count, name="system_prompt"),
            )
        )
        requests = tuple(map(ModelRequest, model_ids, package_ids))
        models = {
            request: self._models.get(request) for request in dict.fromkeys(requests)
        }

        options = dict(self.fixed_options)
        if max_new_tokens is not None:
            options["max_new_tokens"] = max_new_tokens

        results = []
        for member, request, row_prompt in zip(image, requests, prompts):
            answers = models[request].prompt(
                [member.tensor_image],
                prompt=row_prompt,
                input_color_format="rgb",
                **options,
            )
            results.append(
                {
                    "parsed_output": self._single_answer(answers, request=request),
                    "model_id": request.model_id,
                    "inference_id": uuid.uuid4().hex,
                }
            )

        return results

    def _full_prompt(self, prompt: Optional[str], system_prompt: Optional[str]) -> str:
        prompt = prompt or DEFAULT_PROMPT
        if self.default_system_prompt is None:
            if system_prompt is not None:
                raise ValueError(
                    f"{self.type}: {self.family} has no system prompt; put the "
                    "instruction into prompt"
                )
            return prompt

        system_prompt = system_prompt or self.default_system_prompt
        for name, text in (("prompt", prompt), ("system_prompt", system_prompt)):
            if SYSTEM_PROMPT_MARKER in text:
                raise ValueError(
                    f"{self.type}: {name} contains {SYSTEM_PROMPT_MARKER!r}, which "
                    f"{self.family} reads as the start of the system prompt; "
                    "remove it from the text"
                )
        full_prompt = f"{prompt}{SYSTEM_PROMPT_MARKER}{system_prompt}"

        return full_prompt

    def _single_answer(self, answers: Any, *, request: ModelRequest) -> str:
        # Qwen3.5 answers with a dict only in thinking mode, which stays off.
        if (
            not isinstance(answers, list)
            or len(answers) != 1
            or not isinstance(answers[0], str)
        ):
            raise ValueError(
                f"{self.type}: model {str(request)!r} must answer one image with "
                f"a list of one text, got {answers!r}"
            )

        return answers[0]


class Qwen25VLBlock(_LocalVLMBlock):
    """Ask Qwen2.5-VL a question about each image."""

    type = "v2/qwen25vl@v1"
    family = "Qwen2.5-VL"
    default_system_prompt = QWEN25_SYSTEM_PROMPT
    metadata = {"name": "Qwen2.5-VL", "port_of": "roboflow_core/qwen25vl@v1"}

    class Params(VLMParams):
        model_id: ModelIdField = _model_id_field("qwen25-vl-7b")
        system_prompt: TextParam = _system_prompt_field(QWEN25_SYSTEM_PROMPT)
        max_new_tokens: TokensParam = _max_new_tokens_field(
            "512, or the INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS "
            "environment variable when set"
        )


class Qwen3VLBlock(_LocalVLMBlock):
    """Ask Qwen3-VL a question about each image."""

    type = "v2/qwen3vl@v1"
    family = "Qwen3-VL"
    default_system_prompt = QWEN3_SYSTEM_PROMPT
    metadata = {"name": "Qwen3-VL", "port_of": "roboflow_core/qwen3vl@v1"}

    class Params(VLMParams):
        model_id: ModelIdField = _model_id_field("qwen3vl-2b-instruct")
        system_prompt: TextParam = _system_prompt_field(QWEN3_SYSTEM_PROMPT)
        max_new_tokens: TokensParam = _max_new_tokens_field("512, fixed")


class Qwen35VLBlock(_LocalVLMBlock):
    """Ask Qwen3.5 a question about each image, with thinking mode off."""

    type = "v2/qwen3_5vl@v2"
    family = "Qwen3.5"
    default_system_prompt = QWEN3_5_SYSTEM_PROMPT
    # V1 v2 always disables thinking; the answer is then plain text.
    fixed_options = MappingProxyType({"enable_thinking": False})
    metadata = {"name": "Qwen3.5", "port_of": "roboflow_core/qwen3_5vl@v2"}

    class Params(VLMParams):
        model_id: ModelIdField = _model_id_field(
            "qwen3_5-0.8b", "qwen3_5-2b", "qwen3_5-4b"
        )
        system_prompt: TextParam = _system_prompt_field(QWEN3_5_SYSTEM_PROMPT)
        max_new_tokens: TokensParam = _max_new_tokens_field("512, fixed")


class SmolVLM2Block(_LocalVLMBlock):
    """Ask SmolVLM2 a question about each image; it has no system prompt."""

    type = "v2/smolvlm2@v1"
    family = "SmolVLM2"
    metadata = {"name": "SmolVLM2", "port_of": "roboflow_core/smolvlm2@v1"}

    class Params(VLMParams):
        model_id: ModelIdField = _model_id_field("smolvlm2/smolvlm-2.2b-instruct")
        max_new_tokens: TokensParam = _max_new_tokens_field(
            "400, or the INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS "
            "environment variable when set"
        )


BLOCKS = (Qwen25VLBlock, Qwen3VLBlock, Qwen35VLBlock, SmolVLM2Block)


# Copies of the private common.py helpers of the same names; import them
# instead once common.py makes them public.
def _per_row(value: RequestField, *, row_count: int, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, Batch):
        return (value,) * row_count

    if len(value) != row_count:
        raise ValueError(f"{name} has {len(value)} values for {row_count} images")

    return value.content


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")
