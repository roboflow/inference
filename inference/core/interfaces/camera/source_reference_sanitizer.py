from typing import Callable, List, Union
from urllib.parse import urlparse, urlunparse

from inference.core.interfaces.camera.entities import VideoFrameProducer

SourceReferenceInput = Union[
    str, int, List[Union[str, int]], Callable[[], VideoFrameProducer]
]


def sanitize_source_reference(ref: str) -> str:
    """Strip credentials and query parameters from URLs for observability surfaces."""
    parsed = urlparse(ref)
    if parsed.scheme and parsed.hostname:
        netloc = parsed.hostname + (f":{parsed.port}" if parsed.port else "")
        sanitized = parsed._replace(netloc=netloc, query="", fragment="")
        return urlunparse(sanitized)
    return ref


def sanitize_source_reference_for_log(ref: str) -> str:
    return sanitize_source_reference(ref)


def classify_source_reference(ref: SourceReferenceInput) -> str:
    if callable(ref):
        return type(ref).__name__
    if isinstance(ref, list):
        return ",".join(classify_source_reference(item) for item in ref)
    return sanitize_source_reference(str(ref))
