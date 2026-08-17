from contextvars import ContextVar
from typing import Mapping, Optional

from inference.core.env import ALLOW_API_KEY_FROM_HEADERS

# Request-scoped storage for the API key carried in the `Authorization: Bearer`
# header. Set once per request by the `extract_header_api_key` middleware in
# http_api.py and consumed through `api_key_fallback` at the places where the
# effective API key is materialized onto a request model. Relies on the same
# outer->inner ContextVar propagation through BaseHTTPMiddleware as
# `assume_identity_authorised_workspace_db_id` (see roboflow_api.py).
header_api_key: ContextVar[Optional[str]] = ContextVar("header_api_key", default=None)


def extract_api_key_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    """Read the Roboflow API key from the `Authorization: Bearer` header.

    The header is the LAST-RESORT channel: callers must consult the `api_key`
    query parameter and JSON-body field first. The token value is never
    inspected beyond the standard HTTP auth-scheme split - Roboflow API keys
    are opaque strings.
    """
    if not ALLOW_API_KEY_FROM_HEADERS:
        return None
    # starlette Headers are case-insensitive; the second lookup covers plain
    # dict-like mappings.
    authorization = headers.get("Authorization") or headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() == "bearer" and token:
        return token
    return None


def api_key_fallback(current_value: Optional[str]) -> Optional[str]:
    """Return `current_value`, or the header-carried API key when it is None.

    Applied AFTER the existing query-over-body merges, which keeps the
    precedence order: query parameter > body field > header > env API_KEY.
    """
    if current_value is not None:
        return current_value
    return header_api_key.get()
