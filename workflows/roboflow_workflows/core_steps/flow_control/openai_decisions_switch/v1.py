"""OpenAI Decisions Switch workflow block (v1).

Routes each image to a branch by asking OpenAI's Decisions API one ``choice``
question over the block's cases, plus an option meaning none of them. An answer
that names no case, falls below the minimum confidence or is a refusal goes to
the default steps.
"""

import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from roboflow_workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from roboflow_workflows.core_steps.common.utils import run_in_parallel
from roboflow_workflows.core_steps.models.foundation.openai_decisions.v1 import (
    MAX_CHOICES,
    MODEL_VERSION_IDS,
    MODEL_VERSION_METADATA,
    build_decision_input,
    execute_decision_request,
)
from roboflow_workflows.environment import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
    StepSelector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.execution_engine.v1.entities import FlowControl
from roboflow_workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    third_party_model,
)
from roboflow_workflows.prototypes.platform_client import (
    OFFLINE_PLATFORM_CLIENT,
    RoboflowPlatformClient,
)

if ENABLE_TENSOR_DATA_REPRESENTATION:
    from roboflow_workflows.core_steps.common.serializers_tensor import (
        serialize_wildcard_kind,
    )
else:
    from roboflow_workflows.core_steps.common.serializers import serialize_wildcard_kind

logger = get_logger(__name__)

NONE_OPTION = "none of these"
NONE_OPTION_DESCRIPTION = "None of the other options fits."
"""Option the model always has, because a ``choice`` must pick one of its options."""

MAX_CASES = MAX_CHOICES - 1
DEFAULT_MIN_CONFIDENCE = 0.9

LONG_DESCRIPTION = f"""
Route each image to one of several branches by asking GPT-6 Luna a question through
OpenAI's Decisions API. Where Switch Case matches a value that an earlier step computed,
this block decides from the image itself and from what earlier steps found in it.

## How This Block Works

1. Asks `question` about each image, as one choice over the keys of `cases` plus the option
   "{NONE_OPTION}". `case_descriptions` tells the model what a case means.
2. Sends `context` along with the image as JSON, so the decision can use text and values from
   earlier steps, such as OCR text or a detection count. Each image gets its own values.
3. Continues to the step of the chosen case when the model gives it a probability of at least
   `min_confidence`, and terminates the other branches.
4. Continues to `default_next_steps` when the model chooses "{NONE_OPTION}", stays below
   `min_confidence`, returns an invalid probability, refuses to answer or returns an answer
   the block cannot read. When `default_next_steps` is empty, the branch terminates for that image.

A failed request fails the workflow run, as it does in the other OpenAI blocks.

Flow control blocks have no outputs, so the answer is not available to later steps. To keep the
probabilities, use the OpenAI Decisions block with Switch Case instead.

Provide your OpenAI API key or set the value to ``rf_key:account`` (or ``rf_key:user:<id>``) to
proxy requests through Roboflow's API.

## Common Use Cases

- **Triage**: send photos of damaged goods to a claim flow, photos of intact goods to storage and
  anything else to manual review
- **Saving model calls**: run an expensive model only on the images that need it
- **Decisions over several signals**: use the image together with detections and OCR text from
  earlier steps to pick an action

## Connecting to Other Blocks

- **After image inputs or crops**: each image or crop gets its own decision
- **After detection, classification or OCR blocks**: pass what they found through `context`, using
  `context_operations` to reduce it, for example to a count of detections
- **Before any downstream blocks** that should only run for a specific case: each target becomes
  the head of its own execution branch
- **With a default branch** wired through `default_next_steps`, for the images the model is not sure
  about
"""

SHORT_DESCRIPTION = "Route each image to the branch that GPT-6 Luna picks for it."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenAI Decisions Switch",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "search_keywords": [
                "LMM",
                "VLM",
                "GPT",
                "OpenAI",
                "Decisions",
                "switch",
                "router",
            ],
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-split",
                "blockPriority": 5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/open_ai_decisions_switch@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    question: Union[Selector(kind=[STRING_KIND]), str] = Field(
        title="Question",
        description="What the model decides about each image, answered by picking one of "
        "`cases`. It is the same for every image; use `context` for per-image values.",
        examples=["Which line should handle this item?", "$inputs.question"],
        json_schema_extra={"multiline": True, "always_visible": True},
    )
    cases: Dict[str, StepSelector] = Field(
        title="Cases",
        description="Mapping of each option the model can pick to the step that runs when it "
        'does, e.g. {"damaged": "$steps.on_damaged", "intact": "$steps.on_intact"}. Each '
        "target step may appear at most once across `cases` and `default_next_steps`.",
        examples=[{"damaged": "$steps.on_damaged", "intact": "$steps.on_intact"}],
        json_schema_extra={"always_visible": True},
    )
    case_descriptions: Dict[str, str] = Field(
        title="Case Descriptions",
        description="Optional text telling the model what a case means. Keys are keys of `cases`.",
        default_factory=dict,
        examples=[{"damaged": "The box is torn, crushed or wet"}],
    )
    default_next_steps: List[StepSelector] = Field(
        title="Default Steps",
        description="Steps to run when the model picks no case, stays below `min_confidence` or "
        "refuses to answer. Leave empty to terminate the branch for that image.",
        default_factory=list,
        examples=[["$steps.manual_review"]],
    )
    min_confidence: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = (
        Field(
            title="Minimum Confidence",
            description="Probability the model must give the chosen case for its step to run. "
            "Images below it go to `default_next_steps`.",
            default=DEFAULT_MIN_CONFIDENCE,
            examples=[DEFAULT_MIN_CONFIDENCE, "$inputs.min_confidence"],
        )
    )
    context: Dict[str, Union[Selector(), str, int, float, bool]] = Field(
        title="Context",
        description="Named values sent with each image as JSON for the model to weigh, from "
        "workflow inputs, step outputs or literals. Values from batch-oriented steps are "
        "sent per image. The API rejects more than 10,000 values per image, which fails the "
        "run, so reduce large values such as segmentation masks with `context_operations`.",
        default_factory=dict,
        examples=[{"label_text": "$steps.ocr.result", "line": "$inputs.line"}],
    )
    context_operations: Dict[str, List[AllOperationsType]] = Field(
        title="Context Operations",
        description="Optional operations that transform `context` values before they are sent. "
        "Keys are keys of `context`, values are lists of operations (same as Property "
        "Definition block).",
        default_factory=dict,
        examples=[{"boxes": [{"type": "SequenceLength"}]}],
    )
    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description="Your OpenAI API key",
        examples=["xxx-xxx", "$inputs.openai_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal[tuple(MODEL_VERSION_IDS)],
    ] = Field(
        default="gpt-6-luna",
        description="Model to be used. The Decisions API serves GPT-6 Luna.",
        examples=["gpt-6-luna", "$inputs.openai_model"],
        json_schema_extra={
            "values_metadata": MODEL_VERSION_METADATA,
        },
    )
    image_detail: Union[
        Selector(kind=[STRING_KIND]), Literal["auto", "high", "low", "original"]
    ] = Field(
        default="auto",
        description="Detail the model sees each image at: 'low' costs fewer "
        "tokens, 'high' and 'original' help with small details.",
        examples=["auto", "low"],
    )
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description="Number of concurrent requests that can be executed by block when batch of input images provided. "
        "If not given - block defaults to value configured globally in Workflows Execution Engine. "
        "Please restrict if you hit OpenAI limits.",
    )

    @model_validator(mode="after")
    def validate_cases_and_context(self) -> "BlockManifest":
        if not 1 <= len(self.cases) <= MAX_CASES:
            raise ValueError(f"`cases` must hold 1 to {MAX_CASES} cases")
        if any(not case.strip() for case in self.cases):
            raise ValueError("`cases` must not have an empty key")
        if any(case.strip().lower() == NONE_OPTION for case in self.cases):
            raise ValueError(
                f"`cases` must not name '{NONE_OPTION}': the model always has that "
                f"option, and it leads to `default_next_steps`"
            )
        targets = list(self.cases.values()) + list(self.default_next_steps)
        duplicated = {target for target in targets if targets.count(target) > 1}
        if duplicated:
            raise ValueError(
                f"Each step may be targeted at most once across `cases` and "
                f"`default_next_steps`. Duplicated targets: {sorted(duplicated)}"
            )
        unknown_cases = set(self.case_descriptions).difference(self.cases)
        if unknown_cases:
            raise ValueError(
                f"`case_descriptions` names unknown cases: {sorted(unknown_cases)}"
            )
        unknown_values = set(self.context_operations).difference(self.context)
        if unknown_values:
            raise ValueError(
                f"`context_operations` names values missing from `context`: "
                f"{sorted(unknown_values)}"
            )
        return self

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["context"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        # Requires per-key execution branches for dict step selectors, added in
        # Execution Engine v1.11.0.
        return ">=1.11.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [third_party_model(provider="openai", model_id=self.model_version)]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [
            WorkOperation.FLOW_CONTROL,
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.IMAGE_ENCODING,
            WorkOperation.DATA_TRANSFORMATION,
        ]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class OpenAIDecisionsSwitchBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
        platform_client: RoboflowPlatformClient = OFFLINE_PLATFORM_CLIENT,
    ):
        self._api_key = api_key
        self._platform_client = platform_client

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "platform_client"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        question: str,
        cases: Dict[str, str],
        case_descriptions: Dict[str, str],
        default_next_steps: List[str],
        min_confidence: float,
        context: Dict[str, Any],
        context_operations: Dict[str, List[AllOperationsType]],
        model_version: str,
        image_detail: Literal["auto", "high", "low", "original"],
        max_concurrent_requests: Optional[int],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        choice_question = build_choice_question(
            question, cases=cases, case_descriptions=case_descriptions
        )

        tasks = [
            partial(
                execute_decision_request,
                roboflow_api_key=self._api_key,
                platform_client=self._platform_client,
                openai_api_key=api_key,
                decision_input=build_switch_input(
                    image,
                    context=build_image_context(
                        context, index=index, context_operations=context_operations
                    ),
                    image_detail=image_detail,
                ),
                questions=[choice_question],
                model_version=model_version,
            )
            for index, image in enumerate(images)
        ]
        decisions = run_in_parallel(
            tasks=tasks,
            max_workers=max_concurrent_requests
            or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )

        results = [
            route_decision(
                decision,
                cases=cases,
                default_next_steps=default_next_steps,
                min_confidence=min_confidence,
            )
            for decision in decisions
        ]

        return results


def build_choice_question(
    question: str,
    *,
    cases: Dict[str, str],
    case_descriptions: Dict[str, str],
) -> Dict[str, Any]:
    """Build the ``choice`` question the block asks about every image.

    Args:
        question: Instructions of the question.
        cases: Mapping of case to the step it leads to.
        case_descriptions: Text telling the model what a case means.

    Returns:
        A Decisions API question over the cases and the option meaning none of them.
    """
    choices = [
        (
            {"value": case, "description": case_descriptions[case]}
            if case in case_descriptions
            else {"value": case}
        )
        for case in cases
    ]
    choices.append({"value": NONE_OPTION, "description": NONE_OPTION_DESCRIPTION})

    choice_question = {
        "type": "choice",
        "name": "route",
        "instructions": question,
        "choices": choices,
    }

    return choice_question


def build_image_context(
    context: Dict[str, Any],
    *,
    index: int,
    context_operations: Dict[str, List[AllOperationsType]],
) -> Dict[str, Any]:
    """Build the JSON-ready context values of one image.

    Args:
        context: Named values, each a batch with one value per image or a single value.
        index: Position of the image in the batch.
        context_operations: Operations transforming named values before they are sent.

    Returns:
        The values for that image, after operations, with workflow data such as
        detections serialized.
    """
    values = {
        name: value[index] if isinstance(value, Batch) else value
        for name, value in context.items()
    }
    for name, operations in context_operations.items():
        operations_chain = build_operations_chain(operations=operations)
        values[name] = operations_chain(values[name], global_parameters={})

    image_context = _normalize_numpy_values(serialize_wildcard_kind(values))

    return image_context


def _normalize_numpy_values(value: Any) -> Any:
    """Convert nested NumPy arrays and scalars to Python values for JSON encoding."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, dict):
        value = {key: _normalize_numpy_values(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_normalize_numpy_values(item) for item in value]

    return value


def build_switch_input(
    image: WorkflowImageData,
    *,
    context: Dict[str, Any],
    image_detail: str,
) -> List[Dict[str, Any]]:
    """Build the Decisions API input for one image.

    Args:
        image: Workflow image the question is about.
        context: JSON-ready values sent with the image.
        image_detail: Detail level the model sees the image at.

    Returns:
        One user message holding the image, followed by the context when there is one.
    """
    decision_input = build_decision_input(image, image_detail=image_detail, prompt=None)
    if context:
        decision_input[0]["content"].append({"type": "input_json", "data": context})

    return decision_input


def route_decision(
    decision: Any,
    *,
    cases: Dict[str, str],
    default_next_steps: List[str],
    min_confidence: float,
) -> FlowControl:
    """Turn a Decisions API response into the flow control of one image.

    Args:
        decision: Parsed Decisions API response.
        cases: Mapping of case to the step it leads to.
        default_next_steps: Steps for an answer that selects no case.
        min_confidence: Probability the chosen case needs.

    Returns:
        Flow control selecting the case's step, the default steps, or terminating
        the branch when there are no default steps.
    """
    selected_case = read_selected_case(decision, min_confidence=min_confidence)
    if selected_case in cases:
        flow_control = FlowControl(mode="select_step", context=[cases[selected_case]])
    elif default_next_steps:
        flow_control = FlowControl(mode="select_step", context=default_next_steps)
    else:
        flow_control = FlowControl(mode="terminate_branch")

    return flow_control


def read_selected_case(decision: Any, *, min_confidence: float) -> Optional[str]:
    """Read the option a Decisions API response selects.

    Never raises, like the OpenAI Decisions block's decoding: a response that
    cannot be read selects no option.

    Args:
        decision: Parsed Decisions API response.
        min_confidence: Probability the chosen option needs.

    Returns:
        The chosen option, or ``None`` when the response holds no choice, its
        probabilities cannot be read, or the choice's probability is non-finite,
        outside [0, 1], or below ``min_confidence``.
    """
    answers = decision.get("answers") if isinstance(decision, dict) else None
    answer = answers[0] if isinstance(answers, list) and answers else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        logger.warning("Decisions API response holds no choice for the switch.")
        return None

    # The API's own `confidence` rescales this probability by the number of options. The
    # probability is the number the OpenAI Decisions block reports as a class's confidence.
    try:
        probabilities = {
            entry["value"]: float(entry["probability"])
            for entry in answer.get("probabilities") or []
        }
    except (KeyError, TypeError, ValueError) as error:
        logger.warning(
            "Could not read the Decisions API probabilities for the switch. "
            "Error type: %s. Details: %s",
            error.__class__.__name__,
            error,
        )
        return None

    choice = answer.get("choice")
    probability = probabilities.get(choice, 0.0)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None

    if probability < min_confidence:
        return None

    return choice
