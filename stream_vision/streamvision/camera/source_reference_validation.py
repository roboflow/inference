"""Keep request-supplied media references out of raw GStreamer launch syntax."""

import re
from typing import List, Union
from urllib.parse import urlsplit

VideoReference = Union[str, int, List[Union[str, int]]]
SUPPORTED_VIDEO_URL_SCHEMES = {
    "http",
    "https",
    "rtsp",
    "rtsps",
    "rtspt",
    "rtspst",
    "rtmp",
    "rtmps",
    "udp",
    "srt",
    "rtp",
    "tcp",
    "file",
    "csi",
}


def validate_video_references(
    value: VideoReference, *, allow_unsafe: bool
) -> VideoReference:
    if allow_unsafe:
        return value
    for reference in value if isinstance(value, list) else [value]:
        if isinstance(reference, int):
            continue
        if not reference or any(ord(character) < 32 for character in reference):
            raise ValueError("Invalid video reference")
        if "://" in reference:
            parsed = urlsplit(reference)
            if (
                parsed.scheme not in SUPPORTED_VIDEO_URL_SCHEMES
                or any(
                    character.isspace() or character in "\"'" for character in reference
                )
                or (parsed.scheme != "file" and not parsed.hostname)
                or (parsed.scheme == "csi" and not reference[len("csi://") :].isdigit())
            ):
                raise ValueError(
                    "Video references must use a supported, encoded video URL"
                )
            continue
        explicit_path = reference.startswith(("/", "./", "../", ".\\", "..\\")) or bool(
            re.match(r"^[A-Za-z]:[\\/]", reference)
        )
        if not explicit_path or any(
            character in reference for character in ("!", "=", '"', "'")
        ):
            raise ValueError(
                "Stream requests require a camera index, supported URL, or an absolute/explicit "
                "relative file path (for example ./video.mp4). Bare identifiers and raw "
                "GStreamer syntax are disabled. Trusted administrators may explicitly set "
                "ALLOW_UNSAFE_GSTREAMER_PIPELINES=True."
            )
    return value
