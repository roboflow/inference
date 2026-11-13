"""Per-request context forwarded to the Roboflow platform as headers."""

import contextvars
import re
from typing import Dict, Optional

from inference_server import configuration

ASSUME_IDENTITY_ACCESS_TOKEN_HEADER = "x-assume-identity-access-token"
ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER = "x-assume-identity-authorised-workspace"
ENFORCE_CREDITS_VERIFICATION_HEADER = "x-enforce-credits-verification"

_HEADER_IDENTITY_PATTERN = re.compile(r"[\x21-\x7e]+")

assume_identity_authorised_workspace_db_id: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("assume_identity_authorised_workspace_db_id", default=None)
)
enforce_credits_verification: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "enforce_credits_verification", default=True
)


def workspace_db_id_is_valid(workspace_db_id: object) -> bool:
    """Return whether an opaque workspace ID is safe in an HTTP header.

    Args:
        workspace_db_id: Value returned by the platform.

    Returns:
        True when the value is a non-empty printable ASCII string.
    """
    is_valid = isinstance(workspace_db_id, str) and (
        _HEADER_IDENTITY_PATTERN.fullmatch(workspace_db_id) is not None
    )

    return is_valid


def add_assume_identity_headers(headers: Dict[str, str]) -> None:
    """Attach the assume-identity headers for the current request.

    Headers are added only when the service access token is configured and
    the request was authorised for a workspace by the hosted middleware.

    Args:
        headers: Outbound platform request headers, mutated in place.
    """
    if not configuration.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN:
        return

    authorised_workspace = assume_identity_authorised_workspace_db_id.get()
    if not workspace_db_id_is_valid(authorised_workspace):
        return

    headers[ASSUME_IDENTITY_ACCESS_TOKEN_HEADER] = (
        configuration.ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN
    )
    headers[ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER] = authorised_workspace
