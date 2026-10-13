"""Recognize text, tables and formulas with a host-owned local GLM-OCR model.

Port of the latest V1 ``roboflow_core/glm_ocr@v1``. Each step builds one
prompt per image from its task type and asks the model for text::

    run   check images, then each row's task type and its required field
          (this is the direct-call order; workflow validation runs first)
          build each row's prompt exactly as V1 does
          resolve every row's model from the 'models' resource
          per image: model.prompt(tensor_image, prompt=..., **options)
          one {"parsed_output", "model_id", "inference_id"} per image

Task types and the prompt each one sends (V1 strings)::

    text-recognition       "Text Recognition:"
    table-recognition      "Table Recognition:"
    formula-recognition    "Formula Recognition:"
    structured-answering   STRUCTURED_ANSWERING_PROMPT_TEMPLATE + output_structure
    custom                 the step's prompt

Params validates task inputs when the workflow compiles (selectors count as
present), then again per image on resolved values before calling run(). Thus
workflow task-field errors precede run()'s RGB check. Direct run() callers
get the RGB check first, then repeated task-field checks. All checks precede
model loads. ``GlmOcrHF`` puts its ``images`` argument into one
image slot of one conversation, so the block passes each RGB ``(3, H, W)``
``uint8`` tensor by itself, on its own device, one call per image.
"""

import json
import uuid
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import Field, StrictInt, StrictStr, model_validator
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
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

TASK_TYPE = "vlm"
DEFAULT_MODEL_ID = "glm-ocr"
DEFAULT_TASK_TYPE = "text-recognition"

# V1 strings, byte for byte.
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
# Literals, or selectors; values that vary per image arrive as a Batch.
TaskTypeParam = Literal[TASK_TYPES] | Ref(STRING_KIND, batch="if_varying")
TextParam = Optional[StrictStr] | Ref(STRING_KIND, batch="if_varying", allow_none=True)

PARSED_OUTPUT = Output(
    STRING_KIND,
    source="image",
    description=(
        "The model's text for this image. For 'structured-answering' it is a "
        "JSON document in Markdown markers, ready for v2/json_parser@v1."
    ),
)


class GlmOcrBlock(Block):
    """Recognize text, a table or a formula in each image with GLM-OCR."""

    type = "v2/glm_ocr@v1"
    metadata = {"name": "GLM-OCR", "port_of": "roboflow_core/glm_ocr@v1"}

    class Params(ModelSelectionParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to read; one model call per image."
        )
        model_id: ModelIdField = Field(
            default=DEFAULT_MODEL_ID,
            description=MODEL_ID_DESCRIPTION,
            examples=[DEFAULT_MODEL_ID, "$inputs.model_id"],
        )
        task_type: TaskTypeParam = Field(
            default=DEFAULT_TASK_TYPE,
            description=(
                "What to recognize; decides the prompt. One of "
                f"{', '.join(TASK_TYPES)}. 'custom' needs prompt; "
                "'structured-answering' needs output_structure. A selector may "
                "choose it per run or per image; then the step checks it and its "
                "required field when it runs, before any model loads."
            ),
            examples=["table-recognition", "$inputs.task_type"],
        )
        prompt: TextParam = Field(
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
        task_type: RequestField = DEFAULT_TASK_TYPE,
        prompt: RequestField = None,
        output_structure: Optional[Mapping[str, str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read every image of the batch, one model call per image.

        Args:
            image: The batch of RGB images.
            model_id: Model id for every image, or a ``Batch`` with one per
                image.
            model_package_id: Package id for every image, a ``Batch`` with
                one per image, or ``None``.
            task_type: Task type for every image, or a ``Batch`` with one per
                image.
            prompt: Custom prompt for every image, a ``Batch`` with one per
                image, or ``None``.
            output_structure: Field descriptions for 'structured-answering'.
            max_new_tokens: Generation limit; ``None`` keeps the model default.

        Returns:
            One ``{"parsed_output", "model_id", "inference_id"}`` per image, in
            batch order.

        Raises:
            ValueError: On a non-RGB image, a batch of the wrong length, an
                unknown task type, a missing field that the task type needs,
                or a model returning other than one text answer.
        """
        row_count = len(image)
        for member in image:
            _require_rgb(member, block=self.type)
        prompts = tuple(
            _task_prompt(row_task, prompt=row_prompt, output_structure=output_structure)
            for row_task, row_prompt in zip(
                _per_row(task_type, row_count=row_count, name="task_type"),
                _per_row(prompt, row_count=row_count, name="prompt"),
            )
        )
        requests = tuple(
            map(
                ModelRequest,
                _per_row(model_id, row_count=row_count, name="model_id"),
                _per_row(
                    model_package_id, row_count=row_count, name="model_package_id"
                ),
            )
        )
        # Resolve each distinct request once, in first-appearance order,
        # before calling any model.
        distinct_requests = dict.fromkeys(requests)
        models = {request: self._models.get(request) for request in distinct_requests}

        # GlmOcrHF ignores input_color_format today: its processor reads
        # tensors as RGB, which these are. The keyword states the contract.
        options: Dict[str, Any] = {"input_color_format": "rgb"}
        if max_new_tokens is not None:
            options["max_new_tokens"] = max_new_tokens

        results = []
        for member, request, row_prompt in zip(image, requests, prompts):
            answers = models[request].prompt(
                member.tensor_image, prompt=row_prompt, **options
            )
            results.append(
                {
                    "parsed_output": self._single_answer(answers, request=request),
                    "model_id": request.model_id,
                    "inference_id": uuid.uuid4().hex,
                }
            )

        return results

    def _single_answer(self, answers: Any, *, request: ModelRequest) -> str:
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


def _per_row(value: RequestField, *, row_count: int, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, Batch):
        return (value,) * row_count

    if len(value) != row_count:
        raise ValueError(f"{name} has {len(value)} values for {row_count} images")

    return value.content


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")
