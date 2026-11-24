"""Resolve video references that point to local files."""

from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse
from urllib.request import url2pathname


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
        # url2pathname also maps "/C:/clip.mp4" to a Windows drive path.
        decoded_path = url2pathname(urlparse(video).path)
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
