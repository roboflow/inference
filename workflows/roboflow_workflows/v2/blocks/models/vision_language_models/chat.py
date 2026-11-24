"""Shared parts of the remote chat model blocks, independent of any wire protocol.

Every remote chat block (OpenAI, OpenAI-compatible servers, OpenRouter and the
other vendors) does the same few things around its own request format::

    params ──▶ check_task_fields, check_reasoning_effort      before any request
    api_key, roboflow api_key ──▶ reveal_keys ──▶ ChatKeys
        keys.managed ("rf_key:account", "rf_key:user:ID")
            yes ─▶ platform_client.post(<vendor proxy endpoint>, keys.roboflow_key)
            no  ─▶ vendor SDK with keys.provider_key
    each image ──▶ encode_image ──▶ ChatPrompt ──▶ vendor request format
    calls ──▶ run_concurrently(calls, max_concurrent_requests)   results in order
    scrubbed_call(request, keys=keys, ...) fails ──▶ ChatRequestError, keys scrubbed

The vendor modules build their own requests: ``openai.py`` the OpenAI
Responses API and chat completions, ``openrouter.py`` OpenRouter's chat
completions. Nothing here imports a vendor SDK.

``scrubbed_call`` replaces every error of a request with a
``ChatRequestError``. Its message has each revealed key replaced by
``<secret NAME>``, and it has neither ``__cause__`` nor ``__context__``: an
SDK exception can hold the request with its authorization header, so it is
not chained. (A ``with`` block cannot drop ``__context__``, hence a call.)

Differences from V1, applied by every block that uses these helpers:

* A failed request raises ``ChatRequestError`` with the keys scrubbed. V1
  raised the vendor's exception, whose text may contain the key.
* The default concurrency is ``DEFAULT_MAX_CONCURRENT_REQUESTS`` (8, V1's
  default) instead of an environment variable. After the first failure,
  requests that have not started are cancelled.
"""

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import cv2
import numpy as np
from pydantic import StrictStr
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Revealed,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import DependentResource, Output, Ref

ResultT = TypeVar("ResultT")
SizeWH = Tuple[int, int]

MANAGED_KEY_PREFIXES = ("rf_key:account", "rf_key:user:")
"""Provider keys stored by Roboflow; requests go through the platform proxy."""

DEFAULT_MANAGED_KEY = "rf_key:account"
DEFAULT_MAX_CONCURRENT_REQUESTS = 8
JPEG_QUALITY = 90
"""V1 ``encode_image_to_jpeg_bytes`` default quality."""

ApiKey = StrictStr | Ref(SECRET_KIND, STRING_KIND)
"""A provider key parameter: a literal, or a secret or string selector."""

WORK_OPERATIONS = (
    WorkOperation.MODEL_INFERENCE,
    WorkOperation.EXTERNAL_REQUEST,
    WorkOperation.IMAGE_ENCODING,
)
"""What every remote chat call does."""

TASK_METADATA: Dict[str, Dict[str, str]] = {
    "unconstrained": {
        "name": "Open Prompt",
        "description": "Use any prompt to generate a raw response",
    },
    "ocr": {
        "name": "Text Recognition (OCR)",
        "description": "Model recognizes text in the image",
    },
    "visual-question-answering": {
        "name": "Visual Question Answering",
        "description": "Model answers the question you submit in the prompt",
    },
    "caption": {
        "name": "Captioning (short)",
        "description": "Model provides a short description of the image",
    },
    "detailed-caption": {
        "name": "Captioning",
        "description": "Model provides a long description of the image",
    },
    "classification": {
        "name": "Single-Label Classification",
        "description": "Model classifies the image content as one of the provided classes",
    },
    "multi-label-classification": {
        "name": "Multi-Label Classification",
        "description": "Model classifies the image content as one or more of the provided classes",
    },
    "object-detection": {
        "name": "Unprompted Object Detection",
        "description": "Model detects and returns the bounding boxes for prominent objects in the image",
    },
    "instance-segmentation": {
        "name": "Instance Segmentation",
        "description": "Model segments every instance of the provided classes and returns a polygon outline per instance",
    },
    "structured-answering": {
        "name": "Structured Output Generation",
        "description": "Model returns a JSON response with the specified fields",
    },
}
"""V1 ``VLM_TASKS_METADATA`` entries of the tasks remote chat blocks run."""

TASKS_REQUIRING_PROMPT = frozenset({"unconstrained", "visual-question-answering"})
TASKS_REQUIRING_CLASSES = frozenset(
    {
        "classification",
        "multi-label-classification",
        "object-detection",
        "instance-segmentation",
    }
)
TASKS_REQUIRING_OUTPUT_STRUCTURE = frozenset({"structured-answering"})

REASONING_EFFORT_OPTIONS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
"""V1 shared reasoning levels; each model supports a subset."""


class ChatRequestError(RuntimeError):
    """A remote chat request failed; the message names no revealed key."""


@dataclass(frozen=True)
class ChatAnswer:
    """One model answer.

    Args:
        text: The answer text.
        input_tokens: Prompt tokens the provider reported; ``None`` when it
            reported none, never a made-up zero.
        output_tokens: Generated tokens the provider reported, or ``None``.
        thinking: Reasoning trace when the provider returns one, else ``""``.
    """

    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking: str = ""


@dataclass(frozen=True)
class ChatPrompt:
    """What one request asks, before a vendor formats it.

    Args:
        system: System instructions, or ``None``.
        text: User text, or ``None`` when only the image is sent.
        image_first: Whether the image precedes the text in the user message.
    """

    system: Optional[str]
    text: Optional[str]
    image_first: bool = False


@dataclass(frozen=True)
class EncodedImage:
    """An image as uploaded: base64 file content and its pixel size.

    Args:
        base64: Base64 text of the encoded file.
        media_type: ``image/jpeg`` or ``image/png``.
        size_wh: ``(width, height)`` of the uploaded pixels.
    """

    base64: str
    media_type: str
    size_wh: SizeWH

    @property
    def data_url(self) -> str:
        """The image as a ``data:`` URL."""
        url = f"data:{self.media_type};base64,{self.base64}"

        return url


@dataclass(frozen=True, repr=False)
class ChatKeys:
    """Revealed provider and Roboflow keys of one step call.

    Args:
        revealed: ``{"api_key": ..., "roboflow_api_key": ...}`` in plaintext,
            with the secrets to scrub from messages.
    """

    revealed: Revealed

    @property
    def provider_key(self) -> Optional[str]:
        """The vendor key, or a managed ``rf_key:`` reference."""
        key = self.revealed.value["api_key"]

        return key

    @property
    def roboflow_key(self) -> Optional[str]:
        """The host's Roboflow API key, for managed keys."""
        key = self.revealed.value["roboflow_api_key"]

        return key

    @property
    def managed(self) -> bool:
        """Whether the request goes through the Roboflow platform proxy."""
        managed = is_managed_key(self.provider_key)

        return managed

    def scrub(self, text: str) -> str:
        """Replace every revealed key in ``text`` with its ``<secret NAME>``.

        Args:
            text: Any message.

        Returns:
            The scrubbed message.
        """
        scrubbed = self.revealed.scrub(text)

        return scrubbed

    def __repr__(self) -> str:
        return f"ChatKeys(managed={self.managed})"


def is_managed_key(api_key: Optional[str]) -> bool:
    """Tell whether a provider key is a Roboflow-managed ``rf_key:`` reference.

    Args:
        api_key: Revealed provider key or ``None``.

    Returns:
        ``True`` for ``rf_key:account`` and ``rf_key:user:ID`` keys.
    """
    managed = isinstance(api_key, str) and api_key.startswith(MANAGED_KEY_PREFIXES)

    return managed


def reveal_keys(
    api_key: Union[None, str, Secret],
    *,
    roboflow_api_key: Union[None, str, Secret],
) -> ChatKeys:
    """Reveal a step's provider key and the host's Roboflow key together.

    Plain strings are wrapped first, so they are scrubbed like secrets.

    Args:
        api_key: The step's provider key: a literal, a ``Secret`` or ``None``.
        roboflow_api_key: The host's ``api_key`` resource.

    Returns:
        The revealed keys.

    Raises:
        SecretUnavailableError: When a key was read back from a recording.
        TypeError: When a key is neither a string nor a secret.
    """
    keys = {
        "api_key": to_secret(api_key, name="api_key"),
        "roboflow_api_key": to_secret(roboflow_api_key, name="roboflow_api_key"),
    }
    revealed = ChatKeys(revealed=reveal_secrets(keys))

    return revealed


def scrubbed_call(
    request: Callable[[], ResultT], *, keys: ChatKeys, provider: str
) -> ResultT:
    """Run a request; turn any error into a ``ChatRequestError`` without keys.

    A ``ChatRequestError`` raised by ``request`` keeps its text (scrubbed
    again); other errors get ``"<provider> request failed: "`` in front. The
    new error is raised after the original one is no longer being handled,
    so it is chained to nothing.

    Args:
        request: Zero-argument callable sending the request.
        keys: The revealed keys of the call.
        provider: Provider name for the message, e.g. ``"OpenAI"``.

    Returns:
        What ``request`` returns.

    Raises:
        ChatRequestError: For any error ``request`` raises.
    """
    try:
        result = request()
    except ChatRequestError as error:
        message = keys.scrub(str(error))
    except Exception as error:
        message = keys.scrub(f"{provider} request failed: {error}")
    else:
        return result

    raise ChatRequestError(message)


def run_concurrently(
    calls: Sequence[Callable[[], ResultT]],
    *,
    max_concurrent_requests: Optional[int],
) -> List[ResultT]:
    """Run request calls on a thread pool and return results in call order.

    One call runs in the calling thread. The first failure in call order is
    raised once the running calls finish; calls not yet started are
    cancelled, so no further paid request is sent.

    Args:
        calls: Zero-argument callables, one per request.
        max_concurrent_requests: Most calls at once; ``None`` uses
            ``DEFAULT_MAX_CONCURRENT_REQUESTS``.

    Returns:
        One result per call, in order.

    Raises:
        ValueError: When ``max_concurrent_requests`` is below 1.
    """
    limit = (
        DEFAULT_MAX_CONCURRENT_REQUESTS
        if max_concurrent_requests is None
        else max_concurrent_requests
    )
    if limit < 1:
        raise ValueError(f"max_concurrent_requests must be at least 1, got {limit}")
    if len(calls) <= 1:
        results = [call() for call in calls]
        return results

    # Workers take calls in order, so a skipped call always follows the failed
    # one, whose error is raised first below.
    failed = threading.Event()

    def guarded(call: Callable[[], ResultT]) -> Callable[[], Optional[ResultT]]:
        def run() -> Optional[ResultT]:
            if failed.is_set():
                return None
            try:
                result = call()
            except BaseException:
                failed.set()
                raise
            return result

        return run

    executor = ThreadPoolExecutor(max_workers=min(limit, len(calls)))
    try:
        futures = [executor.submit(guarded(call)) for call in calls]
        results = [future.result() for future in futures]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return results


def check_task_fields(
    task_type: str,
    *,
    prompt: Any,
    classes: Any,
    output_structure: Any,
) -> None:
    """Reject a task configuration that misses the field the task needs.

    Args:
        task_type: The configured task.
        prompt: The ``prompt`` value or ``None``.
        classes: The ``classes`` value or ``None``.
        output_structure: The ``output_structure`` value or ``None``.

    Raises:
        ValueError: With V1's message, naming the missing field.
    """
    required = (
        ("prompt", prompt, TASKS_REQUIRING_PROMPT),
        ("classes", classes, TASKS_REQUIRING_CLASSES),
        ("output_structure", output_structure, TASKS_REQUIRING_OUTPUT_STRUCTURE),
    )
    for name, value, tasks in required:
        if task_type in tasks and value is None:
            raise ValueError(
                f"`{name}` parameter required to be set for task `{task_type}`"
            )


def check_reasoning_effort(
    model: Any,
    level: Any,
    *,
    levels_by_model: Mapping[str, Sequence[str]],
) -> None:
    """Reject a reasoning level the selected model does not support.

    Unset values, selectors and models outside ``levels_by_model`` pass, so
    newer models keep working, as in V1.

    Args:
        model: Model id, a selector or ``None``.
        level: Reasoning level, a selector or ``None``.
        levels_by_model: Supported levels per known model; empty means the
            model has no reasoning setting.

    Raises:
        ValueError: With V1's message when the model rejects the level.
    """
    if model is None or level is None or _is_selector(model) or _is_selector(level):
        return
    if model not in levels_by_model:
        return

    supported = list(levels_by_model[model])
    if level in supported:
        return
    if supported:
        raise ValueError(
            f"Model {model} supports reasoning_effort values {supported}, got {level!r}."
        )
    raise ValueError(
        f"Model {model} does not support configurable reasoning_effort (got {level!r})."
    )


def encode_image(
    image: ImageData,
    *,
    image_format: Literal["jpeg", "png"] = "jpeg",
    size_wh: Optional[SizeWH] = None,
) -> EncodedImage:
    """Encode an image for upload; the host boundary of every remote call.

    The pixels leave their device here, once per request. Resizing uses
    OpenCV's Lanczos filter, as V1's detection downscale does.

    Args:
        image: RGB or single-channel image.
        image_format: ``"jpeg"`` (quality 90) or lossless ``"png"``.
        size_wh: Upload ``(width, height)``; ``None`` keeps the image size.

    Returns:
        The encoded image and its uploaded size.

    Raises:
        ValueError: On an unknown format or when OpenCV cannot encode.
    """
    if image_format not in ("jpeg", "png"):
        raise ValueError(f"image_format must be 'jpeg' or 'png', got {image_format!r}")

    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    if pixels.shape[2] == 3:
        pixels = cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2BGR)
    else:
        pixels = np.ascontiguousarray(pixels[:, :, 0])
    height, width = pixels.shape[:2]
    if size_wh is not None and tuple(size_wh) != (width, height):
        pixels = cv2.resize(pixels, tuple(size_wh), interpolation=cv2.INTER_LANCZOS4)
        height, width = pixels.shape[:2]

    if image_format == "jpeg":
        encoded, data = cv2.imencode(
            ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
    else:
        encoded, data = cv2.imencode(".png", pixels)
    if not encoded:
        raise ValueError(f"OpenCV could not encode an image of shape {pixels.shape}")

    text = base64.b64encode(data.tobytes()).decode("ascii")
    encoded_image = EncodedImage(
        base64=text, media_type=f"image/{image_format}", size_wh=(width, height)
    )

    return encoded_image


def token_count(value: Any) -> Optional[int]:
    """Read one usage count; anything but a real ``int`` becomes ``None``.

    Args:
        value: A count from a provider's usage object.

    Returns:
        The count, or ``None`` for a missing, boolean or non-integer value.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None

    return value


def token_outputs(*, source: str) -> Dict[str, Output]:
    """Declare the ``input_tokens`` and ``output_tokens`` outputs.

    Args:
        source: Field whose values give each answer its context, e.g. ``images``.

    Returns:
        Both outputs by name.
    """
    outputs = {
        "input_tokens": Output(
            INTEGER_KIND,
            source=source,
            description="Prompt tokens the provider reported; None when it reported none.",
        ),
        "output_tokens": Output(
            INTEGER_KIND,
            source=source,
            description="Generated tokens the provider reported; None when it reported none.",
        ),
    }

    return outputs


def third_party_model(
    model_id: Any, *, provider: str, resource: str
) -> DependentResource:
    """Describe the remote model a step calls, for workload introspection.

    Args:
        model_id: Model id, or the selector that chooses it at run time.
        provider: Who serves the model, e.g. ``"openai"`` or a base URL.
        resource: Name of the block resource that carries the requests.

    Returns:
        A ``third_party_model`` dependent resource.
    """
    dependency = DependentResource(
        resource_type="third_party_model",
        identifier=model_id,
        details={"provider": provider, "resource": resource},
    )

    return dependency


def _is_selector(value: Any) -> bool:
    selector = isinstance(value, str) and value.startswith("$")

    return selector
