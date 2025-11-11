"""WebRTC client for the Inference SDK."""

from __future__ import annotations

from typing import Optional, Union

from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.session import WebRTCSession
from inference_sdk.webrtc.sources import StreamSource


class WebRTCClient:
    """Namespaced WebRTC API bound to an InferenceHTTPClient instance.

    Provides a unified streaming interface for different video sources
    (webcam, RTSP, video files, manual frames).
    """

    def __init__(self, api_url: str, api_key: Optional[str]) -> None:
        """Initialize WebRTC client.

        Args:
            api_url: Base URL for the inference API
            api_key: API key for authentication (optional)
        """
        self._api_url = api_url
        self._api_key = api_key

    def stream(
        self,
        source: StreamSource,
        *,
        workflow: Union[str, dict],
        image_input: str = "image",
        workspace: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ) -> WebRTCSession:
        """Create a WebRTC streaming session.

        Args:
            source: Stream source (WebcamSource, RTSPSource, VideoFileSource, or ManualSource)
            workflow: Either a workflow ID (str) or workflow specification (dict)
            image_input: Name of the image input in the workflow
            workspace: Workspace name (required if workflow is an ID string)
            config: Stream configuration (output routing, FPS, TURN server, etc.)

        Returns:
            WebRTCSession context manager

        Raises:
            ValueError: If workflow/workspace parameters are invalid

        Examples:
            # Webcam streaming
            from inference_sdk.webrtc import WebcamSource

            with client.webrtc.stream(
                source=WebcamSource(resolution=(1920, 1080)),
                workflow="object-detection",
                workspace="my-workspace"
            ) as session:
                for frame in session.video():
                    cv2.imshow("Frame", frame)

            # RTSP streaming
            from inference_sdk.webrtc import RTSPSource

            with client.webrtc.stream(
                source=RTSPSource("rtsp://camera.local/stream"),
                workflow=workflow_spec_dict
            ) as session:
                @session.on_data("predictions")
                def handle_predictions(data):
                    print("Got predictions:", data)

                session.wait()

            # Manual frame sending
            from inference_sdk.webrtc import ManualSource

            manual = ManualSource()
            with client.webrtc.stream(source=manual, ...) as session:
                for frame in my_frames:
                    manual.send(frame)
        """
        # Validate workflow configuration
        workflow_config = self._parse_workflow_config(workflow, workspace)

        # Use default config if not provided
        if config is None:
            config = StreamConfig()

        # Create session
        return WebRTCSession(
            api_url=self._api_url,
            api_key=self._api_key,
            source=source,
            image_input_name=image_input,
            workflow_config=workflow_config,
            stream_config=config,
        )

    def _parse_workflow_config(
        self, workflow: Union[str, dict], workspace: Optional[str]
    ) -> dict:
        """Parse workflow configuration from inputs.

        Args:
            workflow: Either workflow ID (str) or specification (dict)
            workspace: Workspace name (required for ID mode)

        Returns:
            Dictionary with workflow configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if isinstance(workflow, str):
            # Workflow ID mode - requires workspace
            if not workspace:
                raise ValueError(
                    "workspace parameter required when workflow is an ID string"
                )
            return {"workflow_id": workflow, "workspace_name": workspace}
        elif isinstance(workflow, dict):
            # Workflow specification mode
            return {"workflow_specification": workflow}
        else:
            raise ValueError(
                f"workflow must be a string (ID) or dict (specification), got {type(workflow)}"
            )
