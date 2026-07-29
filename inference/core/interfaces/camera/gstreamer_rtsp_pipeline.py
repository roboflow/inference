"""GStreamer pipeline builder for RTSP(S) ingest on Jetson (ENT-1544 B2)."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Iterable, Tuple
from urllib.parse import unquote

from inference.core.interfaces.camera.rtsp_tls import (
    is_rtsp_url,
    rtsp_tls_validation_flags_gstreamer_suffix,
)

RTSP_PROTOCOLS_ENV_VAR = "ROBOFLOW_RTSP_PROTOCOLS"
RTSP_LATENCY_ENV_VAR = "ROBOFLOW_RTSP_LATENCY_MS"
_DEFAULT_RTSP_PROTOCOLS = "tcp"
_DEFAULT_RTSP_LATENCY_MS = 50
_BOUNDED_QUEUE_OPTIONS = "max-size-buffers=64 max-size-bytes=0 max-size-time=50000000"
_REQUIRED_ELEMENTS = ("rtspsrc", "parsebin", "videoconvert")


def split_rtsp_credentials(url: str) -> tuple[str, str | None, str | None]:
    """Return cred-free URL plus optional username/password.

    Uses right-split on ``@`` so passwords containing ``@`` are handled correctly.
    """
    if not isinstance(url, str) or "://" not in url:
        return url, None, None

    scheme, rest = url.split("://", 1)
    path_start = rest.find("/")
    if path_start == -1:
        authority = rest
        path_and_query = ""
    else:
        authority = rest[:path_start]
        path_and_query = rest[path_start:]

    username: str | None = None
    password: str | None = None
    hostport = authority
    if "@" in authority:
        userinfo, hostport = authority.rsplit("@", 1)
        if ":" in userinfo:
            username, password = userinfo.split(":", 1)
        else:
            username = userinfo
        username = unquote(username)
        password = unquote(password) if password is not None else None

    clean_url = f"{scheme}://{hostport}{path_and_query}"
    return clean_url, username, password


def quote_gstreamer_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _rtsp_protocols() -> str:
    return os.getenv(RTSP_PROTOCOLS_ENV_VAR, _DEFAULT_RTSP_PROTOCOLS).strip() or (
        _DEFAULT_RTSP_PROTOCOLS
    )


def _rtsp_latency_ms() -> int:
    raw = os.getenv(RTSP_LATENCY_ENV_VAR)
    if raw is None:
        return _DEFAULT_RTSP_LATENCY_MS
    try:
        latency = int(raw)
    except ValueError:
        return _DEFAULT_RTSP_LATENCY_MS
    return latency if latency >= 0 else _DEFAULT_RTSP_LATENCY_MS


def build_gstreamer_rtsp_pipeline(url: str) -> str:
    """Build an OpenCV CAP_GSTREAMER pipeline for RTSP(S) with TLS env support."""
    if not is_rtsp_url(url):
        raise ValueError(f"Expected an rtsp:// or rtsps:// URL, got {url!r}")

    clean_url, username, password = split_rtsp_credentials(url)
    tls_suffix = rtsp_tls_validation_flags_gstreamer_suffix()
    source = (
        f'rtspsrc location="{quote_gstreamer_value(clean_url)}" '
        f"protocols={_rtsp_protocols()} latency={_rtsp_latency_ms()}"
        " drop-on-latency=true"
    )
    if username:
        source += f' user-id="{quote_gstreamer_value(username)}"'
    if password:
        source += f' user-pw="{quote_gstreamer_value(password)}"'
    source += (
        f"{tls_suffix} ! application/x-rtp,media=video ! "
        f"queue {_BOUNDED_QUEUE_OPTIONS} ! "
        "parsebin ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    return source


def _load_gstreamer_library():
    library_name = ctypes.util.find_library("gstreamer-1.0")
    if not library_name:
        library_name = "libgstreamer-1.0.so.0"
    gst = ctypes.CDLL(library_name)
    gst.gst_init_check.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    gst.gst_init_check.restype = ctypes.c_int
    gst.gst_element_factory_find.argtypes = [ctypes.c_char_p]
    gst.gst_element_factory_find.restype = ctypes.c_void_p
    gst.gst_object_unref.argtypes = [ctypes.c_void_p]
    gst.gst_object_unref.restype = None
    return gst


def probe_gstreamer_elements(elements: Iterable[str]) -> Tuple[bool, str]:
    try:
        gst = _load_gstreamer_library()
    except OSError as error:
        return False, f"could not load GStreamer: {error}"

    if gst.gst_init_check(None, None, None) == 0:
        return False, "GStreamer failed to initialise"

    for element_name in elements:
        factory = gst.gst_element_factory_find(element_name.encode("ascii"))
        if not factory:
            return False, f"missing GStreamer element: {element_name}"
        gst.gst_object_unref(factory)

    return True, ""


def gstreamer_rtsp_capture_available() -> bool:
    available, _ = probe_gstreamer_elements(_REQUIRED_ELEMENTS)
    return available
