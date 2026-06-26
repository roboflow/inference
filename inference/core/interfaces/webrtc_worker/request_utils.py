from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.roboflow_api import get_roboflow_workspace


def resolve_workspace_id_for_webrtc_request(
    webrtc_request: WebRTCWorkerRequest,
) -> str:
    workspace_id = (
        webrtc_request.workspace_id
        or webrtc_request.workflow_configuration.workspace_name
    )
    if not workspace_id:
        workspace_id = get_roboflow_workspace(api_key=webrtc_request.api_key)
    webrtc_request.workspace_id = workspace_id
    if not webrtc_request.workflow_configuration.workspace_name:
        webrtc_request.workflow_configuration.workspace_name = workspace_id
    return workspace_id
