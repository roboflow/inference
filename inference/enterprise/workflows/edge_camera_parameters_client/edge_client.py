import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from inference.enterprise.workflows.edge_camera_parameters_client.entities import (
    ApplyCameraParametersResult,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def get_edge_base_url() -> str:
    return (
        os.getenv("EDGE_CAMERA_PARAMETERS_BASE_URL")
        or os.getenv("STREAM_CAMERA_PARAMETERS_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def list_pipeline_ids(base_url: Optional[str] = None) -> List[str]:
    url = f"{base_url or get_edge_base_url()}/inference_pipelines/list"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()

    pipelines: List[Any] = []
    if isinstance(data, dict):
        nested = data.get("data")
        if isinstance(nested, dict) and isinstance(nested.get("pipelines"), list):
            pipelines = nested["pipelines"]
        elif isinstance(data.get("pipelines"), list):
            pipelines = data["pipelines"]
    elif isinstance(data, list):
        pipelines = data

    pipeline_ids: List[str] = []
    for entry in pipelines:
        if isinstance(entry, dict) and entry.get("pipeline_id"):
            pipeline_ids.append(str(entry["pipeline_id"]))
    return pipeline_ids


def _encode_pipeline_id_for_path(pipeline_id: str) -> str:
    if "%" in pipeline_id:
        return pipeline_id
    return quote(pipeline_id, safe="")


def post_camera_parameters(
    pipeline_id: str,
    parameters: Dict[str, Any],
    *,
    persist: bool = False,
    only_if_changed: bool = True,
    base_url: Optional[str] = None,
) -> ApplyCameraParametersResult:
    base = base_url or get_edge_base_url()
    encoded_id = _encode_pipeline_id_for_path(pipeline_id)
    url = f"{base}/streams/{encoded_id}/camera-parameters"
    payload = {
        "parameters": parameters,
        "persist": persist,
        "only_if_changed": only_if_changed,
    }
    response = requests.post(url, json=payload, timeout=10)
    try:
        body = response.json()
    except ValueError:
        body = {"success": False, "message": response.text}
    if not isinstance(body, dict):
        body = {"success": False, "message": str(body)}
    if response.status_code == 405:
        body.setdefault("message", "HTTP 405 Method Not Allowed")
    return ApplyCameraParametersResult.from_dict(body, http_ok=response.ok)
