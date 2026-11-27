"""Resolve video references that point to local files."""

import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import unquote

_FILE_URI_PREFIX = "file://"
_LOCAL_AUTHORITIES = {"", "localhost"}
_WINDOWS_DRIVE_PATH = re.compile(r"^/[A-Za-z]:")


def local_file_path(video: Union[str, int]) -> Optional[str]:
    """Return the local file path a video reference points to.

    Scheme-less strings are treated as paths; the caller checks that they exist.

    Args:
        video: Video reference: a camera index, a path, or a URL.

    Returns:
        The decoded path of a local ``file://`` URI, a scheme-less string
        unchanged, or ``None`` for camera indexes, other URL schemes, and
        ``file://`` URIs with a remote host or an undecodable path.
    """
    if not isinstance(video, str):
        return None

    if video[: len(_FILE_URI_PREFIX)].lower() == _FILE_URI_PREFIX:
        decoded_path = _decode_file_uri(video[len(_FILE_URI_PREFIX) :])
        return decoded_path

    if "://" in video:
        return None

    return video


def source_uri(video: Union[str, int]) -> str:
    """Return a URI that GStreamer ``uridecodebin`` can open.

    Args:
        video: Video reference: a camera index, a path, or a URL.

    Returns:
        An absolute ``file://`` URI for local files, else the reference as text.
    """
    file_path = local_file_path(video)
    if file_path is None:
        return str(video)

    file_uri = Path(file_path).resolve().as_uri()
    return file_uri


def _decode_file_uri(authority_and_path: str) -> Optional[str]:
    authority, separator, path = authority_and_path.partition("/")
    if authority.lower() not in _LOCAL_AUTHORITIES:
        return None

    # No query or fragment parsing: '#' and '?' are file-name characters.
    try:
        decoded_path = unquote(separator + path, errors="strict")
    except UnicodeDecodeError:
        # Surrogates from invalid UTF-8 crash cv2.VideoCapture.
        return None

    if _WINDOWS_DRIVE_PATH.match(decoded_path):
        decoded_path = decoded_path[1:]
    return decoded_path
