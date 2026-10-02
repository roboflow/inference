"""Request parsing shared by the hosted authentication middlewares."""

import hmac
import json
import re
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl

from fastapi.responses import JSONResponse
from starlette.routing import Match

from inference_server import configuration
from inference_server.auth import extract_bearer
from inference_server.hosted.assume_identity import enforce_credits_verification

WORKSPACE_ID_HEADER = "X-Workspace-Id"
UNAUTHORIZED_MESSAGE = "Unauthorized api_key"
AUTHENTICATED_METHODS = ("GET", "POST")
STATIC_PREFIXES = ("/static/", "/_next/")

_WORKSPACE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def workspace_id_is_valid(workspace_id: object) -> bool:
    """Return whether an API response contains a usable workspace identity.

    Args:
        workspace_id: Value returned by the platform.

    Returns:
        True when the value is a non-empty string of URL-safe characters.
    """
    is_valid = isinstance(workspace_id, str) and (
        _WORKSPACE_ID_PATTERN.fullmatch(workspace_id) is not None
    )

    return is_valid


class HostedRequest:
    """Read-only view of an ASGI HTTP request that can replay a buffered body."""

    def __init__(self, scope: Dict[str, Any], receive) -> None:
        self.scope = scope
        self._receive = receive
        self._body: Optional[bytes] = None
        self._disconnect: Optional[Dict[str, Any]] = None
        self._replayed = False
        self._json_params: Optional[Dict[str, Any]] = None
        self.method: str = scope.get("method", "")
        self.path: str = scope.get("path", "")
        self.headers: Dict[str, str] = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        self.query_params: Dict[str, str] = dict(
            parse_qsl(
                scope.get("query_string", b"").decode("latin-1"),
                keep_blank_values=True,
            )
        )

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip()

    @property
    def content_length(self) -> int:
        try:
            return int(self.headers.get("content-length", 0))
        except ValueError:
            return 0

    @property
    def has_json_body(self) -> bool:
        return self.content_type == "application/json" and self.content_length > 0

    @property
    def json_params(self) -> Dict[str, Any]:
        return self._json_params or {}

    async def json(self) -> Dict[str, Any]:
        if self._json_params is not None:
            return self._json_params

        body = await self._read_body()
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        self._json_params = parsed if isinstance(parsed, dict) else {}

        return self._json_params

    async def _read_body(self) -> bytes:
        if self._body is not None:
            return self._body

        chunks = []
        while True:
            message = await self._receive()
            if message["type"] != "http.request":
                self._disconnect = message
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        self._body = b"".join(chunks)

        return self._body

    async def receive(self) -> Dict[str, Any]:
        if self._body is not None and not self._replayed:
            self._replayed = True
            if self._disconnect is not None:
                return self._disconnect
            return {"type": "http.request", "body": self._body, "more_body": False}

        message = await self._receive()
        return message


def _is_v2_request(request: HostedRequest) -> bool:
    from inference_server.routers import v2_models, v2_server

    for router in (v2_models.router, v2_server.router):
        for route in router.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return True

    return False


async def resolve_api_key(request: HostedRequest) -> Optional[str]:
    """Resolve the request API key with query > Bearer header > JSON body precedence.

    Args:
        request: Request being authenticated.

    Returns:
        The API key, or None when the request carries none.
    """
    api_key: Any = request.query_params.get("api_key")
    header_allowed = configuration.ALLOW_API_KEY_FROM_HEADERS or _is_v2_request(request)
    if api_key is None and header_allowed:
        api_key = extract_bearer(request.headers.get("authorization", "")) or None
    if api_key is None and request.has_json_body:
        json_params = await request.json()
        api_key = json_params.get("api_key")
    if not isinstance(api_key, str):
        return None

    return api_key


def error_response(
    status_code: int, message: str, workspace_id: Optional[str] = None
) -> JSONResponse:
    """Build the JSON denial body used by the hosted middlewares.

    Args:
        status_code: HTTP status of the denial.
        message: Human-readable reason.
        workspace_id: Workspace the request resolved to, when known.

    Returns:
        Response with ``{"status": ..., "message": ...}`` as body.
    """
    response = JSONResponse(
        status_code=status_code, content={"status": status_code, "message": message}
    )
    if workspace_id_is_valid(workspace_id):
        response.headers[WORKSPACE_ID_HEADER] = workspace_id

    return response


def send_with_workspace_header(send, workspace_id: Optional[str]):
    """Wrap ``send`` so the response start carries ``X-Workspace-Id``.

    Args:
        send: Downstream ASGI send callable.
        workspace_id: Workspace to advertise; skipped when not valid.

    Returns:
        ASGI send callable.
    """
    if not workspace_id_is_valid(workspace_id):
        return send

    header_name = WORKSPACE_ID_HEADER.lower().encode("latin-1")
    header_value = workspace_id.encode("latin-1")

    async def _send(message: Dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            headers = [
                (name, value)
                for name, value in message.get("headers", [])
                if name.lower() != header_name
            ]
            headers.append((header_name, header_value))
            message = {**message, "headers": headers}
        await send(message)

    return _send


_TRUTHY_STRINGS = {"true", "1", "yes", "on"}
_FALSY_STRINGS = {"false", "0", "no", "off"}


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None

    normalized_value = value.strip().lower()
    if normalized_value in _TRUTHY_STRINGS:
        return True
    if normalized_value in _FALSY_STRINGS:
        return False
    return None


def service_secret_is_valid(service_secret: object) -> bool:
    """Return whether a supplied service secret matches the configured one.

    Args:
        service_secret: Value taken from the request.

    Returns:
        True only when ``ROBOFLOW_SERVICE_SECRET`` is set and equal in constant time.
    """
    configured_secret = configuration.ROBOFLOW_SERVICE_SECRET
    if (
        not isinstance(configured_secret, str)
        or not configured_secret
        or not isinstance(service_secret, str)
        or not service_secret
    ):
        return False
    try:
        supplied_secret = service_secret.encode("utf-8")
        expected_secret = configured_secret.encode("utf-8")
    except UnicodeEncodeError:
        return False

    return hmac.compare_digest(supplied_secret, expected_secret)


def is_non_billable_internal_request(request: HostedRequest) -> bool:
    """Return whether the caller asked to skip billing and proved it may.

    ``countinference`` and ``service_secret`` are read from the parsed JSON body
    first and the query string second.

    Args:
        request: Request whose JSON body, if any, has already been parsed.

    Returns:
        True when ``countinference`` is false and the service secret is valid.
    """
    countinference = request.json_params.get(
        "countinference", request.query_params.get("countinference")
    )
    service_secret = request.json_params.get(
        "service_secret", request.query_params.get("service_secret")
    )
    if _coerce_optional_bool(countinference) is not False:
        return False

    return service_secret_is_valid(service_secret)


class BillingIntentMiddleware:
    """Raw ASGI middleware resolving the billing intent of every request.

    Sets the ``enforce_credits_verification`` contextvar to False for an
    authenticated non-billable request (``countinference=false`` plus a valid
    ``service_secret``) so model authorisation omits the credits header. The
    body is only inspected when a service secret is configured, since no
    request can prove non-billable intent otherwise.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not configuration.ROBOFLOW_SERVICE_SECRET:
            await self.app(scope, receive, send)
            return

        request = HostedRequest(scope, receive)
        if request.has_json_body:
            await request.json()
        enforce = not is_non_billable_internal_request(request)

        token = enforce_credits_verification.set(enforce)
        try:
            await self.app(scope, request.receive, send)
        finally:
            enforce_credits_verification.reset(token)
