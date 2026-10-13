"""Webhook sink block: send workflow values in an HTTP request.

Port of V1 ``roboflow_core/webhook_sink@v1``::

    disabled?  ─▶ error_status=False, V1's disabled message
    cooling?   ─▶ throttling_status=True (one window per step, as in V1)
    payloads ── *_operations (UQL) ──▶ JSON-ready copy ─▶ WebhookRequest
    window starts ─▶ fire_and_forget and sink_effects ─▶ background, else sent now

The request is built in the calling thread: an operation or JSON error is
reported by this call, even with ``fire_and_forget``, and the background
effect holds only a detached copy, never workflow payloads. As in V1, the
cooldown window starts only once the request is built, so a failed operation
does not throttle the next call; a request that is sent but fails, or that
full background effects refuse, does start it.

The transport is the ``webhook_client`` resource; there is no default, so no
request leaves without the host choosing a client. ``HttpWebhookClient`` is the
real one: it keeps V1's SSRF protection (validated, pinned destination; no
proxy; no redirect). Tests and examples pass an in-process fake.

Differences from V1: an unserializable payload, query or header value fails
before sending (V1 failed inside the request, unseen in background mode, or
sent ``str(value)`` as a query value); the cooldown uses a monotonic clock;
``HttpWebhookClient`` has no legacy mode with environment proxies and
redirects. Its status and redirect errors name the URL without credentials,
query string or fragment; transport errors, e.g. a refused connection, keep
the underlying library's text, which can include the full URL, as in V1.
"""

import contextlib
import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    BackgroundEffects,
    Cooldown,
    EffectOutcome,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

BLOCK_TYPE = "v2/webhook_sink@v1"
SENT_MESSAGE = "Notification sent successfully"
FAILED_MESSAGE = "Failed to send webhook notification. Internal error details: {}"
THROTTLED_MESSAGE = "Sink cooldown applies"
PAYLOAD_FIELDS = ("json_payload", "multi_part_encoded_files", "form_data")
THROTTLED = {
    "error_status": False,
    "throttling_status": True,
    "message": THROTTLED_MESSAGE,
}

Scalar = Union[StrictStr, StrictInt, StrictFloat, StrictBool]
SCALAR_KINDS = (STRING_KIND, FLOAT_KIND, BOOLEAN_KIND)
PayloadValue = Union[Scalar, dict, list]


@dataclass(frozen=True)
class WebhookRequest:
    """One prepared request; every value is a detached, JSON-ready copy.

    Args:
        method: ``GET``, ``POST`` or ``PUT``.
        url: Target URL.
        query_parameters: Query string values.
        headers: Header values.
        json_payload: JSON body fields; sent as the JSON body.
        form_data: Form fields.
        files: Multipart file fields.
        timeout: Seconds to wait for the response.
    """

    method: str
    url: str
    query_parameters: Mapping[str, Any]
    headers: Mapping[str, Any]
    json_payload: Mapping[str, Any]
    form_data: Mapping[str, Any]
    files: Mapping[str, Any]
    timeout: float


WebhookClient = Callable[[WebhookRequest], None]
"""Send one request; raise on any failure, including a non-2xx status."""


class HttpWebhookClient:
    """Send webhook requests over HTTP(S) with V1's SSRF protection.

    The destination is resolved once, validated and pinned; proxies and
    redirects are refused. Use ``allow_non_global_addresses=True`` only for
    a trusted private network or a loopback test server.

    Args:
        allow_non_global_addresses: Permit private, loopback and link-local
            destinations.
    """

    def __init__(self, *, allow_non_global_addresses: bool = False) -> None:
        self._allow_non_global_addresses = allow_non_global_addresses

    def __call__(self, request: WebhookRequest) -> None:
        """Send ``request``.

        Args:
            request: The prepared request.

        Raises:
            ValueError: On a URL that is not HTTP(S) with a host.
            requests.HTTPError: On a redirect or an error status.
            URLAddressNotAllowedError: On a forbidden destination.
        """
        import requests
        from roboflow_workflows.utils.url_input import SSRFProtectedHTTPAdapter

        # A backslash in the authority is reinterpreted by urllib3 later.
        if "\\" in urllib.parse.urlsplit(request.url).netloc:
            raise ValueError("Webhook URL authority contains a backslash")

        prepared = requests.Request(
            method=request.method,
            url=request.url,
            params=dict(request.query_parameters),
            headers=dict(request.headers),
            json=dict(request.json_payload),
            files=dict(request.files),
            data=dict(request.form_data),
        ).prepare()
        parsed = urllib.parse.urlsplit(prepared.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Webhook requires an HTTP(S) URL with a valid host")

        adapter = SSRFProtectedHTTPAdapter(
            allow_non_global_addresses=self._allow_non_global_addresses
        )
        with contextlib.closing(adapter):
            response = adapter.send(
                prepared, stream=True, timeout=request.timeout, proxies={}
            )
            with contextlib.closing(response):
                status = response.status_code
        shown_url = without_query(request.url)
        if 300 <= status < 400:
            raise requests.HTTPError(
                f"redirect {status} from {shown_url} is not allowed"
            )
        if status >= 400:
            raise requests.HTTPError(f"HTTP {status} from {shown_url}")


class WebhookSinkBlock(Block):
    """Send workflow values to an HTTP endpoint, at most once per cooldown.

    Example: ``json_payload={"count": "$steps.count.output"}`` with
    ``method="POST"`` posts ``{"count": 3}``. ``json_payload_operations``
    can first turn predictions into plain values, e.g. class names.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Webhook Sink",
        "port_of": "roboflow_core/webhook_sink@v1",
        "section": "notifications",
    }
    outputs = {
        "error_status": ERROR_STATUS_OUTPUT,
        "throttling_status": Output(
            BOOLEAN_KIND,
            description="True when the cooldown skipped this request.",
        ),
        "message": MESSAGE_OUTPUT,
    }

    class Params(BlockParams):
        url: StrictStr | Ref(STRING_KIND) = Field(
            description="URL of the endpoint.",
            examples=["https://example.com/hook"],
        )
        method: Literal["GET", "POST", "PUT"] = Field(
            description="HTTP method.", examples=["POST"]
        )
        query_parameters: Dict[
            str,
            Scalar | List[Scalar] | Ref(*SCALAR_KINDS, allow_none=True),  # noqa: F821
        ] = Field(
            default_factory=dict,
            description="Query string values by name.",
            examples=[{"camera": "$inputs.camera_name"}],
        )
        headers: Dict[
            str, Scalar | Ref(*SCALAR_KINDS, allow_none=True)  # noqa: F821
        ] = Field(
            default_factory=dict,
            description="Request headers by name.",
            examples=[{"X-Site": "north"}],
        )
        json_payload: Dict[str, PayloadValue | Ref(allow_none=True)] = (  # noqa: F821
            Field(
                default_factory=dict,
                description=(
                    "JSON body fields: literals or selectors. Values must be JSON "
                    "data after json_payload_operations."
                ),
                examples=[{"count": "$steps.count.output"}],
            )
        )
        json_payload_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description="UQL operations per json_payload key, applied first.",
            examples=[
                {
                    "predictions": [
                        {
                            "type": "DetectionsPropertyExtract",
                            "property_name": "class_name",
                        }
                    ]
                }
            ],
        )
        multi_part_encoded_files: Dict[
            str, PayloadValue | Ref(allow_none=True)  # noqa: F821
        ] = Field(
            default_factory=dict,
            description=(
                "Multipart file fields; e.g. an image after a ConvertImageToJPEG "
                "operation."
            ),
            examples=[{"image": "$inputs.image"}],
        )
        multi_part_encoded_files_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description="UQL operations per multi_part_encoded_files key.",
            examples=[{"image": [{"type": "ConvertImageToJPEG"}]}],
        )
        form_data: Dict[str, PayloadValue | Ref(allow_none=True)] = Field(  # noqa: F821
            default_factory=dict,
            description="Form fields: literals or selectors.",
            examples=[{"site": "north"}],
        )
        form_data_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description="UQL operations per form_data key.",
            examples=[{}],
        )
        request_timeout: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=2,
            gt=0,
            description="Seconds to wait for the response.",
            examples=[2],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Do not wait for the response when the host provides "
                "background effects; otherwise the request is sent now."
            ),
            examples=[True],
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip sending; outputs say the sink was disabled.",
            examples=[False],
        )
        cooldown_seconds: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=5,
            ge=0,
            description=(
                "Least seconds between two sent requests of this step, over "
                "all inputs. Calls inside the window are throttled."
            ),
            examples=[5],
        )

        # Not a model validator: the engine reruns those per invocation.
        @field_validator(*(f"{name}_operations" for name in PAYLOAD_FIELDS))
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            payload_field = info.field_name.removesuffix("_operations")
            if payload_field in info.data:
                compile_payload_operations(
                    value,
                    field=info.field_name,
                    keys=tuple(info.data[payload_field]),
                )

            return value

    def __init__(
        self,
        *,
        webhook_client: WebhookClient,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
        cooldown_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = webhook_client
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks
        self._cooldown = Cooldown(cooldown_clock)
        self._compiled: Dict[str, Tuple[Any, Dict[str, CompiledOperations]]] = {}

    def run(
        self,
        *,
        url: str,
        method: str,
        query_parameters: Dict[str, Any],
        headers: Dict[str, Any],
        json_payload: Dict[str, Any],
        json_payload_operations: Dict[str, List[Any]],
        multi_part_encoded_files: Dict[str, Any],
        multi_part_encoded_files_operations: Dict[str, List[Any]],
        form_data: Dict[str, Any],
        form_data_operations: Dict[str, List[Any]],
        request_timeout: float,
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: float,
    ) -> Dict[str, Any]:
        """Send one request unless disabled or cooling down.

        Args:
            url: Target URL.
            method: HTTP method.
            query_parameters: Query values.
            headers: Header values.
            json_payload: JSON body fields.
            json_payload_operations: Operations per JSON field.
            multi_part_encoded_files: Multipart file fields.
            multi_part_encoded_files_operations: Operations per file field.
            form_data: Form fields.
            form_data_operations: Operations per form field.
            request_timeout: Response timeout in seconds.
            fire_and_forget: Hand the request to background effects if any.
            disable_sink: Skip this request.
            cooldown_seconds: Least seconds between sent requests.

        Returns:
            ``error_status``, ``throttling_status`` and ``message``.

        Raises:
            QueryRuntimeError: When an operation cannot process a value.
        """
        if self._disable_sinks or disable_sink:
            message = disabled_sink_message(by_policy=self._disable_sinks)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": message,
            }

        if self._cooldown.cooling(cooldown_seconds):
            return dict(THROTTLED)

        parts = {
            "query_parameters": (query_parameters, {}),
            "headers": (headers, {}),
            "json_payload": (json_payload, json_payload_operations),
            "multi_part_encoded_files": (
                multi_part_encoded_files,
                multi_part_encoded_files_operations,
            ),
            "form_data": (form_data, form_data_operations),
        }
        prepared = {}
        for field, (values, operations) in parts.items():
            transformed = self._apply_operations(field, values, operations)
            try:
                prepared[field] = detached_json(transformed)
            except (TypeError, ValueError) as error:
                return {
                    "error_status": True,
                    "throttling_status": False,
                    "message": f"{field} is not JSON data: {error}",
                }

        request = WebhookRequest(
            method=method,
            url=url,
            query_parameters=prepared["query_parameters"],
            headers=prepared["headers"],
            json_payload=prepared["json_payload"],
            form_data=prepared["form_data"],
            files=prepared["multi_part_encoded_files"],
            timeout=float(request_timeout),
        )
        # Another call may have started a window while this one prepared.
        if not self._cooldown.try_start(cooldown_seconds):
            return dict(THROTTLED)

        error_status, message = deliver(
            lambda: send(self._client, request),
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=BLOCK_TYPE,
        )

        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }

    def _apply_operations(
        self, field: str, values: Dict[str, Any], operations: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        if not operations:
            return values

        # Keys and operations are literals, so this compiles once per field.
        key = (tuple(values), operations)
        cached = self._compiled.get(field)
        if cached is None or cached[0] != key:
            compiled = compile_payload_operations(
                operations, field=f"{field}_operations", keys=tuple(values)
            )
            cached = (key, compiled)
            self._compiled[field] = cached
        transformed = dict(values)
        for name, chain in cached[1].items():
            transformed[name] = chain.apply(transformed[name], parameters={})

        return transformed


def compile_payload_operations(
    operations: Dict[str, List[Any]], *, field: str, keys: tuple
) -> Dict[str, CompiledOperations]:
    """Compile per-key operation chains that name no runtime values.

    Args:
        operations: Operations by payload key.
        field: Name of the operations parameter, for errors.
        keys: Keys of the payload the operations apply to.

    Returns:
        Compiled chains by key.

    Raises:
        ValueError: When a key is not a payload key.
        QueryDefinitionError: On an invalid chain.
    """
    compiled = {}
    for name, chain in operations.items():
        if name not in keys:
            raise ValueError(
                f"{BLOCK_TYPE}: {field} names `{name}`, which is not a key of "
                f"{field.removesuffix('_operations')} (keys: {list(keys)})"
            )
        compiled[name] = compile_operations(
            chain, path=f"$.{field}.{name}", declared_names=frozenset()
        )

    return compiled


def detached_json(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of JSON data, sharing nothing with the workflow.

    ``bytes`` values (e.g. an encoded image for a multipart field) are kept
    as they are; they are immutable.

    Args:
        values: Field values.

    Returns:
        The copy.

    Raises:
        TypeError: When a value is not JSON data, naming its type.
    """
    copy = {
        name: value if isinstance(value, bytes) else json.loads(json.dumps(value))
        for name, value in values.items()
    }

    return copy


def send(client: WebhookClient, request: WebhookRequest) -> EffectOutcome:
    """Send ``request`` and return V1's outcome.

    Args:
        client: The transport.
        request: The prepared request.

    Returns:
        ``(False, SENT_MESSAGE)`` or ``(True, FAILED_MESSAGE)`` with the error.
    """
    try:
        client(request)
    except Exception as error:
        outcome = (True, FAILED_MESSAGE.format(error))
        return outcome

    return False, SENT_MESSAGE


def without_query(url: str) -> str:
    """Return ``url`` without credentials, query string and fragment, for messages.

    Args:
        url: The URL.

    Returns:
        ``scheme://host[:port]/path``.
    """
    parts = urllib.parse.urlsplit(url)
    host = parts.netloc.rpartition("@")[2]
    shown = urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))

    return shown
