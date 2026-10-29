"""Recognize text, tables and formulas with a host-owned local GLM-OCR model.

Port of the latest V1 ``roboflow_core/glm_ocr@v1``. Each step builds one
prompt per image from the call's task type and asks the model for text::

    run   check images, then the task type and each row's required field
          (this is the direct-call order; workflow validation runs first)
          build each row's prompt exactly as V1 does
          resolve the call's one model from the 'models' resource
          per image: model.prompt(tensor_image, prompt=..., **options)
          one {"parsed_output", "model_id", "inference_id"} per image

Task types and the prompt each one sends (V1 strings)::

    text-recognition       "Text Recognition:"
    table-recognition      "Table Recognition:"
    formula-recognition    "Formula Recognition:"
    structured-answering   STRUCTURED_ANSWERING_PROMPT_TEMPLATE + output_structure
    custom                 the step's prompt

Params validates task inputs when the workflow compiles (selectors count as
present), then again per image on resolved values before calling run(), so
workflow task-field errors precede run()'s RGB check. All checks precede model
loads. ``GlmOcrHF`` puts its ``images`` argument into one image slot of one
conversation, so the block passes each RGB tensor by itself, one call per image.
"""

import json
import uuid
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator
from roboflow_workflows.v2.blocks.models._local_vlm import per_row
from roboflow_workflows.v2.blocks.models._shared.vlm import single_text_answer
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    model_request,
    models_cache,
    one_per_call,
    require_rgb,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

DEFAULT_MODEL_ID = "glm-ocr"

DEFAULT_TASK_TYPE = "text-recognition"

STRUCTURED_ANSWERING_PROMPT_TEMPLATE = (
    "You are supposed to produce responses in JSON wrapped in Markdown markers: "
    "```json\nyour-response\n```. Below is a dictionary with keys and values. "
    "Each key must be present in your response. Values represent descriptions "
    "for JSON fields to be generated. Provide only JSON Markdown in response.\n\n"
    "Specification of requirements regarding output fields:\n{output_structure}"
)

FIXED_TASK_PROMPTS: Mapping[str, str] = {
    "text-recognition": "Text Recognition:",
    "table-recognition": "Table Recognition:",
    "formula-recognition": "Formula Recognition:",
}

TASK_TYPES = (*FIXED_TASK_PROMPTS, "structured-answering", "custom")


class GlmOcrBlock(Block):
    """Recognize text, a table or a formula in each image with GLM-OCR."""

    type = "roboflow_core/glm_ocr@v1"
    metadata = {"name": "GLM-OCR", "port_of": "roboflow_core/glm_ocr@v1"}
    outputs = {
        "parsed_output": Output(
            STRING_KIND,
            source="image",
            description=(
                "The model's text for this image. For 'structured-answering' it is a "
                "JSON document in Markdown markers, ready for "
                "roboflow_core/json_parser@v1."
            ),
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
            description="RGB images to read; one model call per image."
        )
        # One task type per call; the custom prompt may vary per image.
        task_type: Literal[TASK_TYPES] | Ref(STRING_KIND) = Field(
            default=DEFAULT_TASK_TYPE,
            description=(
                "What to recognize in every image of the step; decides the "
                f"prompt. One of {', '.join(TASK_TYPES)}. 'custom' needs prompt; "
                "'structured-answering' needs output_structure. A selector may "
                "choose it per run, not per image; then the step checks it and "
                "its required field when it runs, before any model loads."
            ),
            examples=["table-recognition", "$inputs.task_type"],
        )
        prompt: Optional[StrictStr] | Ref(
            STRING_KIND, batch="if_varying", allow_none=True
        ) = Field(
            default=None,
            description=(
                "Prompt sent as is when task_type is 'custom'; required and "
                "non-empty there, ignored for other task types. A selector may "
                "choose it per run or per image."
            ),
            examples=["Read the serial number.", "$inputs.prompt"],
        )
        output_structure: Optional[Dict[StrictStr, StrictStr]] | Ref(
            DICTIONARY_KIND
        ) = Field(
            default=None,
            description=(
                "JSON field names mapped to descriptions of their values; "
                "required and non-empty when task_type is 'structured-answering', "
                "ignored otherwise. A selector may choose it per run."
            ),
            examples=[{"serial_number": "The serial number on the label"}],
        )
        max_new_tokens: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            ge=1,
            description=(
                "Maximum number of generated tokens per image. None omits the "
                "keyword, using the backend default (8192 unless overridden by "
                "INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS)."
            ),
            examples=[256, "$inputs.max_new_tokens"],
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

        @model_validator(mode="after")
        def _validate_task_inputs(self) -> "GlmOcrBlock.Params":
            # Runs at compile and per image after selector resolution, before
            # run(). Unresolved task selectors defer checking; field selectors
            # count as present. run() repeats checks for direct callers.
            if not _is_selector(self.task_type):
                _check_task_fields(
                    self.task_type,
                    prompt=self.prompt,
                    output_structure=self.output_structure,
                )

            return self

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
        task_type: str = DEFAULT_TASK_TYPE,
        prompt: Union[None, str, Batch] = None,
        output_structure: Optional[Mapping[str, str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read every image of the batch, one model call per image.

        Args:
            image: The batch of RGB images.
            model_id: Model id for every image.
            model_package_id: Package id for every image, or ``None``.
            task_type: Task type for every image.
            prompt: Custom prompt for every image, a ``Batch`` with one per
                image, or ``None``.
            output_structure: Field descriptions for 'structured-answering'.
            max_new_tokens: Generation limit; ``None`` keeps the model default.

        Returns:
            One ``{"parsed_output", "model_id", "inference_id"}`` per image, in
            batch order.

        Raises:
            ValueError: On a non-RGB image, a prompt batch whose length or
                logical indices differ from the image batch's, a ``Batch``
                for a one-per-call field,
                an unknown task type, a missing field that the task type needs,
                or a model returning other than one text answer.
        """
        for member in image:
            require_rgb(member, block=self.type)
        for name, value in (
            ("task_type", task_type),
            ("output_structure", output_structure),
            ("max_new_tokens", max_new_tokens),
        ):
            one_per_call(value, name=name, block=self.type)
        prompts = tuple(
            _task_prompt(
                task_type, prompt=row_prompt, output_structure=output_structure
            )
            for row_prompt in per_row(prompt, image=image, name="prompt")
        )
        request = model_request(model_id, model_package_id, block=self.type)
        # An empty batch loads no model.
        if not len(image):
            return []

        model = self._models.get(request)

        # GlmOcrHF ignores input_color_format today: its processor reads
        # tensors as RGB, which these are. The keyword states the contract.
        options: Dict[str, Any] = {"input_color_format": "rgb"}
        if max_new_tokens is not None:
            options["max_new_tokens"] = max_new_tokens

        results = []
        for member, row_prompt in zip(image, prompts):
            answers = model.prompt(member.tensor_image, prompt=row_prompt, **options)
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


BLOCKS = (GlmOcrBlock,)


def _task_prompt(
    task_type: Any,
    *,
    prompt: Optional[str],
    output_structure: Optional[Mapping[str, str]],
) -> str:
    _check_task_fields(task_type, prompt=prompt, output_structure=output_structure)
    if task_type == "custom":
        return prompt

    if task_type == "structured-answering":
        _require_text_mapping(output_structure)
        task_prompt = STRUCTURED_ANSWERING_PROMPT_TEMPLATE.format(
            output_structure=json.dumps(dict(output_structure), indent=4),
        )

        return task_prompt

    return FIXED_TASK_PROMPTS[task_type]


def _check_task_fields(
    task_type: Any,
    *,
    prompt: Optional[str],
    output_structure: Optional[Mapping[str, str]],
) -> None:
    # The V1 messages; empty values count as missing, as in V1.
    if task_type not in TASK_TYPES:
        raise ValueError(
            f"Unsupported GLM-OCR task_type {task_type!r}. Expected one of: "
            f"{', '.join(sorted(TASK_TYPES))}."
        )

    if task_type == "custom" and not prompt:
        raise ValueError("`prompt` is required when task_type is 'custom'.")

    if task_type == "structured-answering" and not output_structure:
        raise ValueError(
            "`output_structure` is required when task_type is "
            "'structured-answering'."
        )


def _require_text_mapping(output_structure: Mapping[Any, Any]) -> None:
    # A literal is checked when compiling; a selected mapping only here.
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in output_structure.items()
    ):
        raise ValueError(
            "`output_structure` must map field names to text descriptions, got "
            f"{dict(output_structure)!r}"
        )


def _is_selector(value: Any) -> bool:
    selector = isinstance(value, str) and value.startswith("$")

    return selector
