import re
from typing import Optional

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")


def _cred_free_netloc(netloc: str) -> str:
    """Drop userinfo from netloc.

    RTSP passwords may contain unencoded ``@`` (e.g. ``user:p@ss@host``), so split
    from the right — the last ``@`` separates credentials from host:port.
    """
    if "@" not in netloc:
        return netloc
    return netloc.rsplit("@", 1)[-1]


def _strip_schemeless_userinfo(ref: str) -> Optional[str]:
    """Strip ``userinfo@host`` refs that omit a URL scheme."""
    path_start = ref.find("/")
    if path_start == -1:
        authority = ref
        path = ""
    else:
        authority = ref[:path_start]
        path = ref[path_start:]

    if "@" not in authority:
        return None

    host_port = authority.rsplit("@", 1)[-1]
    rest = host_port + path
    return rest.split("?", 1)[0].split("#", 1)[0]


def _sanitize_schemed_url(ref: str) -> Optional[str]:
    """Sanitize a URL with scheme using string ops.

    Avoids ``urlparse`` so malformed bracket sequences in passwords cannot trigger
    IPv6 validation errors.
    """
    match = _SCHEME_RE.match(ref)
    if not match:
        return None

    scheme = match.group(1)
    rest = ref[match.end() :]

    path_start = len(rest)
    for delim in ("/", "?", "#"):
        idx = rest.find(delim)
        if idx != -1:
            path_start = min(path_start, idx)

    authority = _cred_free_netloc(rest[:path_start])
    remainder = rest[path_start:]
    path = remainder.split("?", 1)[0].split("#", 1)[0]

    return f"{scheme}://{authority}{path}"


def sanitize_source_reference(ref: str) -> str:
    """Strip credentials and query parameters from URLs for observability surfaces."""
    if not isinstance(ref, str) or not ref:
        return ref

    schemed = _sanitize_schemed_url(ref)
    if schemed is not None:
        return schemed

    schemeless = _strip_schemeless_userinfo(ref)
    if schemeless is not None:
        return schemeless

    return ref
