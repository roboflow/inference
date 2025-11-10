from __future__ import annotations

from typing import Optional

from inference_sdk.http.client import InferenceHTTPClient
from inference_sdk.webrtc.config import WebcamConfig
from inference_sdk.webrtc.session import WebRTCSession


class WebRTCClient:
    """Namespaced WebRTC API bound to an InferenceHTTPClient instance."""

    def __init__(self, http_client: InferenceHTTPClient) -> None:
        self._client = http_client

    def use_webcam(
        self,
        *,
        image_input_name: str,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        workflow_specification: Optional[dict] = None,
        config: Optional[WebcamConfig] = None,
    ) -> WebRTCSession:
        """Open a WebRTC webcam session.

        Provide either (workspace_name + workflow_id) or workflow_specification.
        image_input_name is required.
        """
        if not image_input_name:
            raise ValueError("image_input_name is required")
        spec_mode = workflow_specification is not None
        id_mode = workspace_name is not None and workflow_id is not None
        if not (spec_mode ^ id_mode):
            raise ValueError(
                "Provide exactly one of: (workspace_name + workflow_id) or workflow_specification"
            )

        cfg = config or WebcamConfig()
        # Access base url and api key from client (private fields)
        api_url = getattr(self._client, f"_{self._client.__class__.__name__}__api_url")
        api_key = getattr(self._client, f"_{self._client.__class__.__name__}__api_key")

        return WebRTCSession(
            api_url=api_url,
            api_key=api_key,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_specification=workflow_specification,
            image_input_name=image_input_name,
            resolution=cfg.resolution,
            webrtc_realtime_processing=cfg.webrtc_realtime_processing,
            webrtc_turn_config=cfg.webrtc_turn_config,
            stream_output=cfg.stream_output,
            data_output=cfg.data_output,
            declared_fps=cfg.declared_fps,
            workflows_parameters=cfg.workflows_parameters,
        )
