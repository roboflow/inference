from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.roboflow_api import get_roboflow_workspace


def _set_workspace_id_on_webrtc_request(
    webrtc_request: WebRTCWorkerRequest,
    workspace_id: str,
) -> None:
    webrtc_request.workspace_id = workspace_id
    webrtc_request.workflow_configuration.workspace_name = workspace_id


def resolve_workspace_id_for_webrtc_request(
    webrtc_request: WebRTCWorkerRequest,
) -> str:
    workspace_id = get_roboflow_workspace(api_key=webrtc_request.api_key)
    _set_workspace_id_on_webrtc_request(
        webrtc_request=webrtc_request,
        workspace_id=workspace_id,
    )
    return workspace_id


def reuse_resolved_workspace_id_for_webrtc_request(
    webrtc_request: WebRTCWorkerRequest,
) -> str:
    workspace_id = webrtc_request.workspace_id
    if not workspace_id:
        workspace_id = resolve_workspace_id_for_webrtc_request(webrtc_request)
    else:
        _set_workspace_id_on_webrtc_request(
            webrtc_request=webrtc_request,
            workspace_id=workspace_id,
        )
    return workspace_id
