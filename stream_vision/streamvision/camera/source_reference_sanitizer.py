import re
from typing import Optional
from urllib.parse import urlsplit

UNPARSEABLE_SOURCE = "<unparseable source>"

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_URL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
# ``user[:password]@`` right after ``://``; the password may contain unencoded
# ``@`` (the last one wins) but not ``/``.
_EMBEDDED_CREDENTIALS_RE = re.compile(r"(://)[^/@:\s]+(?::[^/\s]*)?@")
# ``@host:port/`` inside a path: signature of credentials with an unencoded
# '/' spilling past the netloc, where the real authority follows the last '@'.
_PATH_HOSTPORT_AFTER_AT_RE = re.compile(r"@[^/@]*:\d+(?:/|$)")


def redact_credentials_in_text(text: str) -> str:
    """Sanitize every URL-shaped token embedded in free text (error messages,
    reprs); tokens whose credentials cannot be separated safely are replaced
    with ``UNPARSEABLE_SOURCE``."""
    return _URL_TOKEN_RE.sub(lambda match: _sanitize_schemed_url(match.group(0)), text)


def _has_parseable_port(parts) -> bool:
    try:
        parts.port
    except ValueError:
        return False
    return True


def _sanitize_schemed_url(ref: str) -> str:
    if ref[:7].lower() == "file://":
        # local-file URI: '#' and '?' are file-name characters, credentials
        # do not apply
        return ref
    for candidate in (ref, _EMBEDDED_CREDENTIALS_RE.sub(r"\1", ref)):
        try:
            parts = urlsplit(candidate)
        except ValueError:
            continue
        if "@" not in parts.netloc and ":" in parts.netloc:
            # A colon-free netloc cannot be a credential fragment (usernames
            # do not contain '/'), so these checks only apply with a colon.
            if not _has_parseable_port(parts):
                if "@" in candidate:
                    # netloc like 'user:pa' left over from a password
                    # containing '/', '?' or '#'
                    continue
            elif _PATH_HOSTPORT_AFTER_AT_RE.search(parts.path):
                # valid-looking netloc ('user:12') but the path carries an
                # '@host:port' spill — credentials with an unencoded '/'
                continue
        credential_free_netloc = parts.netloc.rsplit("@", 1)[-1].lower()
        return f"{parts.scheme}://{credential_free_netloc}{parts.path}"
    return UNPARSEABLE_SOURCE


def _looks_like_host_with_port(host_port: str) -> bool:
    _, sep, port = host_port.rpartition(":")
    return bool(sep) and port.isdigit()


def _strip_schemeless_userinfo(ref: str) -> Optional[str]:
    """Strip ``user[:password]@host`` from refs that omit a URL scheme."""
    path_start = ref.find("/")
    if path_start == -1:
        authority, path = ref, ""
    else:
        authority, path = ref[:path_start], ref[path_start:]
    if "@" not in authority:
        return None
    userinfo, _, host_port = authority.rpartition("@")
    has_password = ":" in userinfo and "\\" not in userinfo
    if not has_password and not _looks_like_host_with_port(host_port):
        # '@' in a plain file name ('video@2x.mp4', 'C:\clips\cam@1.mp4'),
        # not credentials.
        return None
    return (host_port + path).split("?", 1)[0].split("#", 1)[0]


def sanitize_source_reference(ref: str) -> str:
    """Strip credentials and query parameters from URLs for observability
    surfaces; returns ``UNPARSEABLE_SOURCE`` when credentials cannot be
    separated from the host safely."""
    if _SCHEME_RE.match(ref):
        return _sanitize_schemed_url(ref)
    stripped = _strip_schemeless_userinfo(ref)
    return redact_credentials_in_text(ref if stripped is None else stripped)
