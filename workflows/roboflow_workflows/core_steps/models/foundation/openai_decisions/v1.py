"""OpenAI Decisions workflow block (v1).

Asks OpenAI's Decisions API typed questions about each image. Where the OpenAI
block generates text, the Decisions API returns one answer per question, with
probabilities: a ``predicate`` gives the probability that a statement holds, a
``choice`` picks one of the supplied values and gives the probability of each,
and a ``score`` rates the input against ordered levels.

The classification tasks build the questions from ``classes`` - one ``choice``
over every class for single-label classification, one ``predicate`` per class
for multi-label classification - and decode the answers into the
classification prediction of the OpenAI v7 block, with OpenAI's probabilities
as the confidences.
"""

import base64
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import requests
from openai import OpenAI
from pydantic import ConfigDict, Field, model_validator
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.token_usage import (
    TOKEN_OUTPUT_DEFINITIONS,
    parse_responses_api_usage,
)
from roboflow_workflows.core_steps.common.utils import run_in_parallel
from roboflow_workflows.core_steps.common.vlm_decoding import (
    create_classes_index,
    describe_vlm_prediction_outputs,
    to_tensor_native_predictions,
)
from roboflow_workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from roboflow_workflows.environment import (
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
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
    WorkflowBlock,
    WorkflowBlockManifest,
    third_party_model,
)
from roboflow_workflows.prototypes.platform_client import (
    OFFLINE_PLATFORM_CLIENT,
    RoboflowPlatformClient,
)
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes

logger = get_logger(__name__)

MODEL_VERSION_METADATA = {"gpt-6-luna": {"name": "GPT-6 Luna"}}
MODEL_VERSION_IDS = list(MODEL_VERSION_METADATA)

CLASSIFICATION_TASK = "classification"
MULTI_LABEL_CLASSIFICATION_TASK = "multi-label-classification"
QUESTIONS_TASK = "questions"

SUPPORTED_TASK_TYPES_LIST = [
    CLASSIFICATION_TASK,
    MULTI_LABEL_CLASSIFICATION_TASK,
    QUESTIONS_TASK,
]
TaskType = Literal[tuple(SUPPORTED_TASK_TYPES_LIST)]

TASKS_REQUIRING_CLASSES = {CLASSIFICATION_TASK, MULTI_LABEL_CLASSIFICATION_TASK}

TASKS_METADATA = {
    CLASSIFICATION_TASK: VLM_TASKS_METADATA[CLASSIFICATION_TASK],
    MULTI_LABEL_CLASSIFICATION_TASK: VLM_TASKS_METADATA[
        MULTI_LABEL_CLASSIFICATION_TASK
    ],
    QUESTIONS_TASK: {
        "name": "Decisions API Questions",
        "description": "Model answers the provided predicate, choice and score questions",
    },
}
TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{metadata['name']}** (`{task}`) - {metadata['description']}"
    for task, metadata in TASKS_METADATA.items()
)

QUESTION_TYPES = ("predicate", "choice", "score")
MAX_QUESTIONS = 400
MAX_CHOICES = 255

SINGLE_LABEL_INSTRUCTIONS = "Which of these classes best describes the image?"
MULTI_LABEL_INSTRUCTIONS_TEMPLATE = "The image shows {class_name}."

PREDICTED_CLASS_MIN_PROBABILITY = 0.5
"""Probability from which a multi-label class counts as present in the image."""

LONG_DESCRIPTION = f"""
Ask OpenAI's Decisions API typed questions about images and get probabilities
back instead of generated text.

The Decisions API runs on GPT-6 Luna and answers three kinds of questions: a
`predicate` returns the probability that a statement about the image is true, a
`choice` picks one of the supplied values and returns the probability of each,
and a `score` rates the image against ordered levels. It bills input tokens
only.

The block supports the following tasks:

{TASKS_DOCS_DESCRIPTION}

For the classification tasks, `predictions` holds the classification prediction
of the OpenAI block's v7:

* `classification` asks one `choice` question over `classes`. Each class gets
the probability OpenAI returns for it as its confidence, and `top` is the
chosen class.
* `multi-label-classification` asks one `predicate` per class ("The image
shows <class>."). Each class gets the probability of its statement, and
`predicted_classes` lists the classes whose probability is at least
{PREDICTED_CLASS_MIN_PROBABILITY}.

The `questions` task sends your own questions, and `predictions` is `None`.

`answers` holds OpenAI's answers for every task, one per question in question
order. `error_status` is `True` when OpenAI refused a question or its answers
could not be decoded, and `predictions` is then `None`.

The optional `prompt` is sent with each image as context for every question,
for instance what the images show or what the classes mean.

Provide your OpenAI API key or set the value to ``rf_key:account`` (or
``rf_key:user:<id>``) to proxy requests through Roboflow's API.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OpenAI Decisions",
            "version": "v1",
            "short_description": "Answer typed questions about images with probabilities.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "LMM",
                "VLM",
                "GPT",
                "OpenAI",
                "Decisions",
                "classification",
                "probability",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/open_ai_decisions@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    task_type: TaskType = Field(
        default=CLASSIFICATION_TASK,
        description="Task to be performed. The classification tasks ask about "
        "`classes`; `questions` asks the provided Decisions API questions.",
        json_schema_extra={
            "values_metadata": TASKS_METADATA,
            "always_visible": True,
        },
    )
    classes: Optional[Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]]] = Field(
        default=None,
        description="Classes the images are classified into.",
        examples=[["cat", "dog"], "$inputs.classes"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_CLASSES, "required": True},
            },
        },
    )
    questions: Optional[
        Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[Dict[str, Any]]]
    ] = Field(
        default=None,
        description="Decisions API questions asked about each image: objects with "
        "a `type` of `predicate`, `choice` or `score`, their `instructions`, an "
        "optional `name` and, for `choice` and `score`, their `choices` or `levels`.",
        examples=[
            [
                {
                    "type": "predicate",
                    "name": "helmets",
                    "instructions": "Every person in the image wears a helmet.",
                }
            ],
            "$inputs.questions",
        ],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": {QUESTIONS_TASK}, "required": True},
            },
        },
    )
    prompt: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Optional text sent with each image as context for every "
        "question, for instance what the images show or what the classes mean.",
        examples=[
            "Photos of steel surfaces; the classes are defect types.",
            "$inputs.prompt",
        ],
        json_schema_extra={
            "multiline": True,
        },
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
    def validate(self) -> "BlockManifest":
        if self.task_type in TASKS_REQUIRING_CLASSES:
            if self.classes is None:
                raise ValueError(
                    f"`classes` parameter required to be set for task `{self.task_type}`"
                )
            if isinstance(self.classes, list):
                validate_classes(self.classes, task_type=self.task_type)
        if self.task_type == QUESTIONS_TASK:
            if self.questions is None:
                raise ValueError(
                    f"`questions` parameter required to be set for task `{self.task_type}`"
                )
            if isinstance(self.questions, list):
                validate_questions(self.questions)
        return self

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="answers", kind=[LIST_OF_VALUES_KIND]),
            *TOKEN_OUTPUT_DEFINITIONS,
            *describe_vlm_prediction_outputs(SUPPORTED_TASK_TYPES_LIST),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [third_party_model(provider="openai", model_id=self.model_version)]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.IMAGE_ENCODING,
        ]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class OpenAIDecisionsBlockV1(WorkflowBlock):

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

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def run(
        self,
        images: Batch[WorkflowImageData],
        task_type: TaskType,
        classes: Optional[List[str]],
        questions: Optional[List[Dict[str, Any]]],
        prompt: Optional[str],
        model_version: str,
        image_detail: Literal["auto", "high", "low", "original"],
        max_concurrent_requests: Optional[int],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        decision_questions = build_questions(
            task_type=task_type, classes=classes, questions=questions
        )

        tasks = [
            partial(
                execute_decision_request,
                roboflow_api_key=self._api_key,
                platform_client=self._platform_client,
                openai_api_key=api_key,
                decision_input=build_decision_input(
                    image, image_detail=image_detail, prompt=prompt
                ),
                questions=decision_questions,
                model_version=model_version,
            )
            for image in images
        ]
        decisions = run_in_parallel(
            tasks=tasks,
            max_workers=max_concurrent_requests
            or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )

        results = []
        for image, decision in zip(images, decisions):
            inference_id = str(uuid4())
            error_status, answers, predictions = decode_decision(
                decision,
                task_type=task_type,
                image=image,
                classes=classes,
                question_count=len(decision_questions),
                inference_id=inference_id,
            )
            usage = decision.get("usage") if isinstance(decision, dict) else None
            input_tokens, output_tokens = parse_responses_api_usage(usage)
            results.append(
                {
                    "answers": answers,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "predictions": predictions,
                    "error_status": error_status,
                    "inference_id": inference_id,
                }
            )

        return results


def validate_classes(classes: List[Any], *, task_type: str) -> None:
    """Check classes against the questions the task builds from them.

    Args:
        classes: Class names the task classifies into.
        task_type: One of the classification tasks.

    Raises:
        ValueError: If a class is empty or repeated, or there are fewer or more
            classes than the task's questions accept.
    """
    if any(
        not isinstance(class_name, str) or not class_name.strip()
        for class_name in classes
    ):
        raise ValueError("`classes` must contain non-empty strings")
    if len(set(classes)) != len(classes):
        raise ValueError("`classes` must not repeat a class")
    if task_type == CLASSIFICATION_TASK and not 2 <= len(classes) <= MAX_CHOICES:
        raise ValueError(f"`{task_type}` needs 2 to {MAX_CHOICES} classes")
    if (
        task_type == MULTI_LABEL_CLASSIFICATION_TASK
        and not 1 <= len(classes) <= MAX_QUESTIONS
    ):
        raise ValueError(f"`{task_type}` needs 1 to {MAX_QUESTIONS} classes")


def validate_questions(questions: List[Any]) -> None:
    """Check the questions of the ``questions`` task before sending them.

    Only the question count and types are checked; OpenAI validates the rest.

    Args:
        questions: Decisions API questions.

    Raises:
        ValueError: If there are no questions or more than the API accepts, or a
            question has no known ``type``.
    """
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"`questions` must hold 1 to {MAX_QUESTIONS} questions")
    if any(
        not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES
        for question in questions
    ):
        raise ValueError(
            f"Every question needs a `type` of {', '.join(QUESTION_TYPES)}"
        )


def build_questions(
    *,
    task_type: str,
    classes: Optional[List[str]],
    questions: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build the Decisions API questions the block asks about every image.

    Single-label classification asks one ``choice`` over the classes,
    multi-label classification one ``predicate`` per class, and the
    ``questions`` task the questions it was given.

    Args:
        task_type: Task the block is configured to run.
        classes: Class names, for the classification tasks.
        questions: Decisions API questions, for the ``questions`` task.

    Returns:
        Questions in the order their answers come back.

    Raises:
        ValueError: If the task's classes or questions are missing or invalid.
    """
    if task_type == QUESTIONS_TASK:
        if questions is None:
            raise ValueError(f"`questions` are required for task `{task_type}`")
        validate_questions(questions)
        return questions

    if classes is None:
        raise ValueError(f"`classes` are required for task `{task_type}`")
    validate_classes(classes, task_type=task_type)

    if task_type == CLASSIFICATION_TASK:
        decision_questions = [
            {
                "type": "choice",
                "name": "class",
                "instructions": SINGLE_LABEL_INSTRUCTIONS,
                "choices": [{"value": class_name} for class_name in classes],
            }
        ]
    else:
        decision_questions = [
            {
                "type": "predicate",
                "name": f"class_{class_id}",
                "instructions": MULTI_LABEL_INSTRUCTIONS_TEMPLATE.format(
                    class_name=class_name
                ),
            }
            for class_id, class_name in enumerate(classes)
        ]

    return decision_questions


def build_decision_input(
    image: WorkflowImageData,
    *,
    image_detail: str,
    prompt: Optional[str],
) -> List[Dict[str, Any]]:
    """Build the Decisions API input for one image.

    Args:
        image: Workflow image the questions are about.
        image_detail: Detail level the model sees the image at.
        prompt: Optional context sent with the image.

    Returns:
        One user message holding the image as a JPEG data URL, followed by the
        prompt when there is one.
    """
    base64_image = base64.b64encode(
        encode_image_to_jpeg_bytes(image.numpy_image)
    ).decode("ascii")
    content = [
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_image}",
            "detail": image_detail,
        }
    ]
    if prompt:
        content.append({"type": "input_text", "text": prompt})

    decision_input = [{"role": "user", "content": content}]

    return decision_input


def execute_decision_request(
    *,
    roboflow_api_key: Optional[str],
    platform_client: RoboflowPlatformClient,
    openai_api_key: str,
    decision_input: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model_version: str,
) -> Any:
    """Ask the Decisions API the questions about one input.

    Requests are proxied through Roboflow when the OpenAI key is a Roboflow
    managed ``rf_key:`` key; otherwise the OpenAI API is called directly.

    Args:
        roboflow_api_key: Roboflow API key, required for proxied execution.
        platform_client: Client for the Roboflow API proxy.
        openai_api_key: OpenAI API key or Roboflow-proxied ``rf_key:`` key.
        decision_input: Decisions API input built by ``build_decision_input``.
        questions: Decisions API questions.
        model_version: OpenAI model identifier.

    Returns:
        The Decisions API response as parsed JSON.

    Raises:
        ValueError: If a managed key is used without a Roboflow API key.
        RuntimeError: If the Roboflow proxy cannot be reached.
    """
    if openai_api_key.startswith(("rf_key:account", "rf_key:user:")):
        if not roboflow_api_key:
            raise ValueError(
                "Roboflow API key is required when using a Roboflow-managed OpenAI API key."
            )
        decision = _execute_proxied_decision_request(
            roboflow_api_key=roboflow_api_key,
            platform_client=platform_client,
            openai_api_key=openai_api_key,
            decision_input=decision_input,
            questions=questions,
            model_version=model_version,
        )
    else:
        decision = _execute_direct_decision_request(
            openai_api_key=openai_api_key,
            decision_input=decision_input,
            questions=questions,
            model_version=model_version,
        )

    return decision


def _execute_proxied_decision_request(
    roboflow_api_key: str,
    platform_client: RoboflowPlatformClient,
    openai_api_key: str,
    decision_input: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model_version: str,
) -> Any:
    payload = {
        "model": model_version,
        "input": decision_input,
        "questions": questions,
        "openai_api_key": openai_api_key,
    }
    try:
        decision = platform_client.post(
            endpoint="apiproxy/openai/decisions",
            api_key=roboflow_api_key,
            payload=payload,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Roboflow proxy: {e}") from e

    return decision


def _execute_direct_decision_request(
    openai_api_key: str,
    decision_input: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model_version: str,
) -> Dict[str, Any]:
    client = _get_openai_client(openai_api_key)
    decision = client.decisions.create(
        model=model_version,
        input=decision_input,
        questions=questions,
        # Each request carries a different image, so a cache write would almost
        # never be read again, and cache writes bill above the input rate.
        # Explicit mode with no breakpoints turns caching off.
        prompt_cache_options={"mode": "explicit"},
    )
    decision_payload = decision.to_dict()

    return decision_payload


def decode_decision(
    decision: Any,
    *,
    task_type: str,
    image: WorkflowImageData,
    classes: Optional[List[str]],
    question_count: int,
    inference_id: str,
) -> Tuple[bool, Optional[List[Any]], Any]:
    """Decode a Decisions API response into the block's outputs for one image.

    Never raises: a response that cannot be decoded sets ``error_status``.

    Args:
        decision: Parsed Decisions API response.
        task_type: Task the block is configured to run.
        image: Workflow image the answers are about.
        classes: Class names the classification questions were built from.
        question_count: Number of questions the request asked.
        inference_id: Identifier attached to the prediction.

    Returns:
        Tuple of ``(error_status, answers, predictions)``. ``predictions`` is
        ``None`` for the ``questions`` task and whenever ``error_status`` is
        ``True``; ``answers`` is ``None`` when the response holds none.
    """
    answers = decision.get("answers") if isinstance(decision, dict) else None
    if not isinstance(answers, list):
        logger.warning("Decisions API response holds no answers.")
        return True, None, None
    if len(answers) != question_count:
        logger.warning(
            "Decisions API response has %s answers for %s questions.",
            len(answers),
            question_count,
        )
        return True, answers, None
    if any(
        not isinstance(answer, dict) or answer.get("type") == "refusal"
        for answer in answers
    ):
        return True, answers, None
    if task_type == QUESTIONS_TASK:
        return False, answers, None

    try:
        if task_type == CLASSIFICATION_TASK:
            prediction = build_single_label_prediction(
                answers[0], image=image, classes=classes, inference_id=inference_id
            )
        else:
            prediction = build_multi_label_prediction(
                answers, image=image, classes=classes, inference_id=inference_id
            )
        predictions = to_tensor_native_predictions(
            predictions=prediction,
            image=image,
            classes=classes,
            inference_id=inference_id,
        )
    except Exception as error:
        logger.warning(
            "Could not decode Decisions API %s answers. Error type: %s. Details: %s",
            task_type,
            error.__class__.__name__,
            error,
        )
        return True, answers, None

    return False, answers, predictions


def build_single_label_prediction(
    answer: Dict[str, Any],
    *,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
) -> dict:
    """Build the single-label classification prediction from a ``choice`` answer.

    Args:
        answer: Answer to the ``choice`` question over the classes.
        image: Workflow image the prediction refers to.
        classes: Class names, in class id order.
        inference_id: Identifier attached to the prediction.

    Returns:
        The classification prediction dict: every class with its probability as
        confidence, and the chosen class as ``top``.

    Raises:
        ValueError: If the answer does not choose one of the classes with a
            probability.
    """
    probabilities = {
        entry["value"]: float(entry["probability"])
        for entry in answer.get("probabilities") or []
    }
    top_class = answer.get("choice")
    if answer.get("type") != "choice" or top_class not in classes:
        raise ValueError("the answer does not choose one of the classes")
    if top_class not in probabilities:
        raise ValueError("the answer has no probability for the chosen class")

    height, width = image.numpy_image.shape[:2]
    prediction = {
        "image": {"width": width, "height": height},
        "predictions": [
            {
                "class": class_name,
                "class_id": class_id,
                "confidence": probabilities.get(class_name, 0.0),
            }
            for class_name, class_id in create_classes_index(classes=classes).items()
        ],
        "top": top_class,
        "confidence": probabilities[top_class],
        "inference_id": inference_id,
        "parent_id": image.parent_metadata.parent_id,
    }

    return prediction


def build_multi_label_prediction(
    answers: List[Dict[str, Any]],
    *,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
) -> dict:
    """Build the multi-label classification prediction from ``predicate`` answers.

    Args:
        answers: One ``predicate`` answer per class, in class order.
        image: Workflow image the prediction refers to.
        classes: Class names, in class id order.
        inference_id: Identifier attached to the prediction.

    Returns:
        The multi-label classification prediction dict: every class with the
        probability of its statement as confidence, and the classes whose
        probability reaches ``PREDICTED_CLASS_MIN_PROBABILITY`` as
        ``predicted_classes``.

    Raises:
        ValueError: If an answer is not a ``predicate`` answer.
    """
    if any(answer.get("type") != "predicate" for answer in answers):
        raise ValueError("every class needs a predicate answer")
    probabilities = {
        class_name: float(answer["probability"])
        for class_name, answer in zip(classes, answers)
    }

    height, width = image.numpy_image.shape[:2]
    prediction = {
        "image": {"width": width, "height": height},
        "predictions": {
            class_name: {"confidence": probabilities[class_name], "class_id": class_id}
            for class_name, class_id in create_classes_index(classes=classes).items()
        },
        "predicted_classes": [
            class_name
            for class_name in classes
            if probabilities[class_name] >= PREDICTED_CLASS_MIN_PROBABILITY
        ],
        "inference_id": inference_id,
        "parent_id": image.parent_metadata.parent_id,
    }

    return prediction


def _get_openai_client(api_key: str) -> OpenAI:
    client = _openai_client_cache.get(api_key)
    if client is None:
        client = OpenAI(api_key=api_key)
        _openai_client_cache[api_key] = client
    return client


_openai_client_cache: Dict[str, OpenAI] = {}
