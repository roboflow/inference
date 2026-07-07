"""WebRTC client for the Inference SDK."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

from inference_sdk.http.errors import InvalidParameterError
from inference_sdk.utils.decorators import experimental
from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.session import WebRTCSession
from inference_sdk.webrtc.sources import StreamSource


class WebRTCClient:
    """Namespaced WebRTC API bound to an InferenceHTTPClient instance.

    Provides a unified streaming interface for different video sources
    (webcam, RTSP, video files, manual frames).
    """

    @experimental(
        info="WebRTC SDK is experimental and under active development. "
        "API may change in future releases. Please report issues at "
        "https://github.com/roboflow/inference/issues"
    )
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
        workflow: Optional[Union[str, dict]] = None,
        model_id: Optional[str] = None,
        image_input: str = "image",
        workspace: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ) -> WebRTCSession:
        """Create a WebRTC streaming session.

        Exactly one of ``workflow`` or ``model_id`` must be provided.

        When ``model_id`` is given, a minimal object-detection workflow is built
        under the hood and the session delivers the raw serialized predictions
        dict alongside each frame instead of raw metadata (see the model_id
        example below).

        Args:
            source: Stream source (WebcamSource, RTSPSource, VideoFileSource, or ManualSource)
            workflow: Either a workflow ID (str) or workflow specification (dict).
                Mutually exclusive with ``model_id``.
            model_id: Roboflow model ID (e.g. "rfdetr-nano"). When provided, a
                Workflow wrapping this object-detection model is built
                automatically and ``on_frame`` handlers receive
                ``(frame, data)`` where ``data`` is the raw serialized
                predictions dict exactly as received from the server. The dict
                is inference-response-shaped, so convert it in your handler with
                ``sv.Detections.from_inference(data)``. Mutually exclusive with
                ``workflow``.
            image_input: Name of the image input in the workflow
            workspace: Workspace name (required if workflow is an ID string)
            config: Stream configuration (output routing, FPS, TURN server, etc.)

        Returns:
            WebRTCSession context manager

        Raises:
            InvalidParameterError: If workflow/model_id/workspace parameters are invalid

        Examples:
            # Pattern 1: Using run() with decorators (recommended, auto-cleanup)
            from inference_sdk.webrtc import WebcamSource

            session = client.webrtc.stream(
                source=WebcamSource(resolution=(1920, 1080)),
                workflow="object-detection",
                workspace="my-workspace"
            )

            @session.on_frame
            def process_frame(frame, metadata):
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    session.close()

            session.run()  # Auto-closes on exception or stream end

            # Pattern 2: Using video() iterator (requires context manager or explicit close)
            from inference_sdk.webrtc import RTSPSource

            # Option A: With context manager (recommended)
            with client.webrtc.stream(
                source=RTSPSource("rtsp://camera.local/stream"),
                workflow=workflow_spec_dict
            ) as session:
                for frame, metadata in session.video():
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            # Auto-cleanup on exit

            # Option B: Manual cleanup (not recommended)
            session = client.webrtc.stream(source=RTSPSource("rtsp://..."), ...)
            for frame, metadata in session.video():
                process(frame)
            session.close()  # Must call close() explicitly!

            # Pattern 3: model_id mode - handlers receive the raw predictions dict
            import supervision as sv
            from inference_sdk.webrtc import VideoFileSource

            session = client.webrtc.stream(
                source=VideoFileSource("cars.mp4"),
                model_id="rfdetr-nano",
            )

            @session.on_frame
            def show(frame, data):   # data = raw serialized predictions dict
                detections = sv.Detections.from_inference(data)
                annotated = sv.BoxAnnotator().annotate(frame.copy(), detections)
                cv2.imshow("Frame", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    session.close()

            session.run()
        """
        # Validate that exactly one of workflow / model_id is provided
        if (workflow is None) == (model_id is None):
            raise InvalidParameterError(
                "Exactly one of 'workflow' or 'model_id' must be provided "
                "(got both or neither)."
            )

        model_mode = model_id is not None

        if model_mode:
            workflow_config = {
                "workflow_specification": self._build_model_workflow(model_id)
            }
            config = self._apply_model_id_defaults(config)
        else:
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
            model_mode=model_mode,
            predictions_output="predictions",
        )

    @staticmethod
    def _build_model_workflow(model_id: str) -> dict:
        """Build a minimal object-detection workflow spec wrapping a model ID.

        Args:
            model_id: Roboflow model ID (e.g. "rfdetr-nano")

        Returns:
            Workflow specification dict producing "predictions" and "image" outputs
        """
        return {
            "version": "1.0",
            "inputs": [{"type": "InferenceImage", "name": "image"}],
            "steps": [
                {
                    "type": "roboflow_core/roboflow_object_detection_model@v2",
                    "name": "model",
                    "images": "$inputs.image",
                    "model_id": model_id,
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "predictions",
                    "coordinates_system": "own",
                    "selector": "$steps.model.predictions",
                },
                {
                    "type": "JsonField",
                    "name": "image",
                    "selector": "$inputs.image",
                },
            ],
        }

    @staticmethod
    def _apply_model_id_defaults(config: Optional[StreamConfig]) -> StreamConfig:
        """Fill default stream/data outputs for model_id mode.

        Defaults route the "image" output through the video path and the
        "predictions" output through the data channel. When the user supplies a
        config, only empty ``stream_output`` / ``data_output`` fields are filled;
        all other settings are preserved. The user's config is never mutated.

        Args:
            config: User-provided stream configuration, or None

        Returns:
            StreamConfig with model_id defaults applied
        """
        if config is None:
            return StreamConfig(stream_output=["image"], data_output=["predictions"])
        return replace(
            config,
            stream_output=config.stream_output or ["image"],
            data_output=config.data_output or ["predictions"],
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
            InvalidParameterError: If configuration is invalid
        """
        if isinstance(workflow, str):
            # Workflow ID mode - requires workspace
            if not workspace:
                raise InvalidParameterError(
                    "workspace parameter required when workflow is an ID string"
                )
            return {"workflow_id": workflow, "workspace_name": workspace}
        elif isinstance(workflow, dict):
            # Workflow specification mode
            return {"workflow_specification": workflow}
        else:
            raise InvalidParameterError(
                f"workflow must be a string (ID) or dict (specification), got {type(workflow)}"
            )
