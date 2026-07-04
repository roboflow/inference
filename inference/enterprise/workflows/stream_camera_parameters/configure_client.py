import os
from typing import Any, Dict, Optional

import requests

from inference.enterprise.workflows.stream_camera_parameters.entities import (
    ApplyCameraParametersResult,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def get_configure_base_url() -> str:
    return os.getenv("STREAM_CAMERA_PARAMETERS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def configure_usb_camera(
    video_reference: str,
    v4l2_properties: Dict[str, Any],
    *,
    base_url: Optional[str] = None,
) -> ApplyCameraParametersResult:
    if not v4l2_properties:
        return ApplyCameraParametersResult(
            success=False,
            message="v4l2_properties must be non-empty",
        )

    base = (base_url or get_configure_base_url()).rstrip("/")
    url = f"{base}/cameras/configure"
    payload = {
        "video_reference": str(video_reference),
        "usb_camera_settings": {"v4l2_camera_properties": v4l2_properties},
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        body = response.json() if response.content else {}
    except Exception as exc:
        return ApplyCameraParametersResult(success=False, message=str(exc))

    if not isinstance(body, dict):
        body = {"success": False, "message": str(body)}

    details = body.get("details") or {}
    applied = list(details.get("applied") or [])
    failed = list(details.get("failed") or [])
    success = bool(body.get("success", response.ok))

    return ApplyCameraParametersResult(
        success=success,
        applied=applied,
        failed=failed,
        message=body.get("message"),
    )
