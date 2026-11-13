"""Masking of labelled credentials in free-form text.

Shared by error logging and trace export, so it imports nothing from the rest
of ``inference_server``.
"""

import re

_AUTHORIZATION_VALUE = re.compile(
    r"""((?<![A-Za-z0-9])authorization[ \t]*(?:=|:|%3D)[ \t]*)(?!["'])[^\r\n]+""",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"""(?<![A-Za-z0-9])
    (?P<name>api_key|apikey|api-key|service_secret|access_token|token|secret
    |password|signature)
    (?P<name_quote>["']?)
    (?P<separator>[ \t]*(?:=|:|%3D)[ \t]*)
    (?:(?P<quote>["'])[^"']*(?P=quote)|[^\s&,;}"']+)""",
    re.IGNORECASE | re.VERBOSE,
)


def _redact_secret_value(match: re.Match) -> str:
    quote = match.group("quote") or ""
    prefix = match.group("name") + match.group("name_quote") + match.group("separator")

    return f"{prefix}{quote}***{quote}"


def redact_secret_parameters(text: str) -> str:
    """Mask labelled credentials, such as ``api_key=...`` or ``Authorization: ...``.

    URLs keep their host and path; only the credential values are replaced.

    Args:
        text: Text that may carry labelled credentials.

    Returns:
        The text with each labelled credential value replaced by ``***``.
    """
    without_authorization = _AUTHORIZATION_VALUE.sub(r"\1***", text)
    redacted = _SECRET_VALUE.sub(_redact_secret_value, without_authorization)

    return redacted
