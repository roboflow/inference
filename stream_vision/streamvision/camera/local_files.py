"""Resolve video references that point to local files."""

from typing import Optional, Union
from urllib.parse import unquote, urlparse


def local_file_path(video: Union[str, int]) -> Optional[str]:
    """Return the local file path a video reference points to.

    Scheme-less strings are treated as paths; the caller checks that they exist.

    Args:
        video: Video reference: a camera index, a path, or a URL.

    Returns:
        The decoded path of a ``file://`` URI, a scheme-less string unchanged,
        or ``None`` for camera indexes and other URL schemes.
    """
    if not isinstance(video, str):
        return None
    if video.startswith("file://"):
        decoded_path = unquote(urlparse(video).path)
        return decoded_path
    if "://" in video:
        return None
    return video
