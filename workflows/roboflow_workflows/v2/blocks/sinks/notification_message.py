"""Message helpers shared by the e-mail, Slack and Twilio notification blocks.

Used by ``email.py`` (``roboflow_core/email_notification@v2``), ``slack.py``
(``roboflow_core/slack_notification@v1``) and ``twilio.py``
(``roboflow_core/twilio_sms_notification@v2``). A change here changes all three::

    message ── referenced_parameters ──▶ names
            ── ParameterOperations.apply (UQL, compiled once per step) ──▶ values
            ── render_message (one scan; values are not re-expanded) ──▶ text
    ImageData ── image_to_jpeg / image_to_base64 (JPEG_QUALITY) ──▶ bytes / text
    managed API error ── managed_error_handlers ──▶ ManagedProviderError(status)
"""

import base64
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, ValidationInfo
from roboflow_workflows.utils.image_encoding import encode_image_to_jpeg_bytes
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.data.image import ImageData

JPEG_QUALITY = 90
"""V1's e-mail JPEG quality; every image a notification block encodes uses it."""

PARAMETER_PATTERN = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")
"""A ``{{ $parameters.NAME }}`` placeholder; group 2 is the name."""

Scalar = Union[StrictStr, StrictInt, StrictFloat, StrictBool]


def referenced_parameters(message: str, parameters: Dict[str, Any]) -> List[str]:
    """Return the ``message_parameters`` keys the message uses, in first-use order.

    Args:
        message: Template with ``{{ $parameters.NAME }}`` placeholders.
        parameters: Available parameter values by name.

    Returns:
        Names that appear in a placeholder and are keys of ``parameters``.
        Placeholders naming other keys stay in the text unchanged, as in V1.
    """
    names = []
    for _, name in PARAMETER_PATTERN.findall(message):
        if name in parameters and name not in names:
            names.append(name)

    return names


def render_message(message: str, texts: Dict[str, str]) -> str:
    """Replace each placeholder whose name is in ``texts`` by that text.

    The message is scanned once, so a substituted text that itself contains
    a placeholder stays as written.

    Args:
        message: Template with ``{{ $parameters.NAME }}`` placeholders.
        texts: Replacement text by parameter name.

    Returns:
        The rendered message.
    """

    def substitute(match: "re.Match[str]") -> str:
        return texts.get(match.group(2), match.group(0))

    rendered = PARAMETER_PATTERN.sub(substitute, message)

    return rendered


def compile_parameter_operations(
    operations: Dict[str, List[Any]], *, keys: Tuple[str, ...], block_type: str
) -> Dict[str, CompiledOperations]:
    """Compile ``message_parameters_operations`` chains.

    Args:
        operations: UQL operations by parameter name.
        keys: Keys of ``message_parameters``.
        block_type: Block type, for error messages.

    Returns:
        Compiled chains by parameter name.

    Raises:
        ValueError: When an operations key is not a ``message_parameters`` key.
        QueryDefinitionError: On an invalid chain.
    """
    compiled = {}
    for name, chain in operations.items():
        if name not in keys:
            raise ValueError(
                f"{block_type}: message_parameters_operations names `{name}`, "
                f"which is not a key of message_parameters (keys: {list(keys)})"
            )
        compiled[name] = compile_operations(
            chain,
            path=f"$.message_parameters_operations.{name}",
            declared_names=frozenset(),
        )

    return compiled


def validate_parameter_operations(
    value: Dict[str, List[Operation]], info: ValidationInfo, *, block_type: str
) -> Dict[str, List[Operation]]:
    """Field validator body: check operations against literal parameter keys.

    Args:
        value: The ``message_parameters_operations`` value.
        info: Pydantic validation info holding ``message_parameters``.
        block_type: Block type, for error messages.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: When an operations key is not a ``message_parameters`` key.
    """
    if "message_parameters" in info.data:
        compile_parameter_operations(
            value, keys=tuple(info.data["message_parameters"]), block_type=block_type
        )

    return value


class ParameterOperations:
    """Per-step cache of compiled ``message_parameters_operations``.

    Operations and parameter keys are literals, so they compile once per step.

    Args:
        block_type: Block type, for error messages.
    """

    def __init__(self, *, block_type: str) -> None:
        self._block_type = block_type
        self._key: Any = None
        self._compiled: Dict[str, CompiledOperations] = {}

    def apply(
        self,
        parameters: Dict[str, Any],
        operations: Dict[str, List[Any]],
        *,
        names: List[str],
    ) -> Dict[str, Any]:
        """Return the values of ``names`` after their operations.

        Args:
            parameters: ``message_parameters`` values.
            operations: ``message_parameters_operations``.
            names: Parameters to evaluate.

        Returns:
            Values by name; parameters without operations are returned as given.

        Raises:
            QueryRuntimeError: When an operation cannot process a value.
        """
        key = (tuple(parameters), operations)
        if key != self._key:
            self._compiled = compile_parameter_operations(
                operations, keys=tuple(parameters), block_type=self._block_type
            )
            self._key = key

        values = {}
        for name in names:
            chain = self._compiled.get(name)
            value = parameters[name]
            values[name] = value if chain is None else chain.apply(value, parameters={})

        return values


def image_to_jpeg(image: ImageData) -> bytes:
    """Encode an image as JPEG bytes at V1's quality; the host boundary.

    Args:
        image: RGB or single-channel image.

    Returns:
        The JPEG file content.
    """
    pixels = image.tensor_image.detach().cpu().permute(1, 2, 0).numpy()
    if pixels.shape[2] == 3:
        pixels = pixels[:, :, ::-1].copy()
    else:
        pixels = pixels[:, :, 0].copy()
    jpeg = encode_image_to_jpeg_bytes(pixels, jpeg_quality=JPEG_QUALITY)

    return jpeg


def image_to_base64(image: ImageData) -> str:
    """Encode an image as base64 text of a JPEG.

    Args:
        image: RGB or single-channel image.

    Returns:
        ASCII base64 of ``image_to_jpeg(image)``.
    """
    text = base64.b64encode(image_to_jpeg(image)).decode("ascii")

    return text


def as_list(value: Union[None, str, List[str]]) -> Optional[List[str]]:
    """Return a single string as a one-item list; keep lists and ``None``.

    Args:
        value: Address(es) or ``None``.

    Returns:
        A new list, or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]

    listed = list(value)

    return listed


class ManagedProviderError(RuntimeError):
    """The Roboflow API rejected a managed send; carries its status and detail.

    Args:
        status: HTTP status code.
        detail: The API's ``details`` or ``error`` text.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def managed_error_handlers(statuses: Tuple[int, ...]) -> Dict[int, Callable]:
    """Build ``http_errors_handlers`` keeping the API's own error text, as V1 does.

    Args:
        statuses: HTTP status codes to handle.

    Returns:
        Handlers by status; each raises ``ManagedProviderError``.
    """

    def handler_for(status: int) -> Callable[[Exception], None]:
        def handle(http_error: Exception) -> None:
            try:
                body = http_error.response.json()
                detail = body.get("details") or body.get("error") or str(http_error)
            except Exception:
                detail = str(http_error)
            raise ManagedProviderError(status, detail) from http_error

        return handle

    handlers = {status: handler_for(status) for status in statuses}

    return handlers
