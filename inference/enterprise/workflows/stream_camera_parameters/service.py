from typing import Any, Dict, Optional
from urllib.parse import quote

from inference.core.logger import logger
from inference.enterprise.workflows.stream_camera_parameters import edge_client
from inference.enterprise.workflows.stream_camera_parameters.entities import (
    ApplyCameraParametersResult,
)


class StreamCameraParametersError(Exception):
    pass


def resolve_pipeline_id(stream_name: Optional[str]) -> str:
    if stream_name:
        return quote(stream_name, safe="")
    pipeline_ids = edge_client.list_pipeline_ids()
    if len(pipeline_ids) == 1:
        return pipeline_ids[0]
    if not pipeline_ids:
        raise StreamCameraParametersError(
            "No active inference pipelines; provide stream_name"
        )
    raise StreamCameraParametersError(
        "Multiple active pipelines; stream_name is required"
    )


def apply_camera_register_parameters(
    parameters: Dict[str, Any],
    *,
    stream_name: Optional[str] = None,
    persist: bool = False,
    only_if_changed: bool = True,
) -> ApplyCameraParametersResult:
    if not isinstance(parameters, dict) or not parameters:
        return ApplyCameraParametersResult(
            success=False,
            message="parameters must be a non-empty object",
        )

    try:
        pipeline_id = resolve_pipeline_id(stream_name)
    except StreamCameraParametersError as exc:
        return ApplyCameraParametersResult(success=False, message=str(exc))

    try:
        return edge_client.post_camera_parameters(
            pipeline_id,
            parameters,
            persist=persist,
            only_if_changed=only_if_changed,
        )
    except Exception as exc:
        logger.exception("Failed to apply stream camera parameters")
        return ApplyCameraParametersResult(success=False, message=str(exc))
