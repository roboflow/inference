"""WebRTC client for the Inference SDK."""

from __future__ import annotations

from typing import Optional, Union

from inference_sdk.http.errors import InvalidParameterError
from inference_sdk.utils.decorators import experimental
from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.model_workflows import (
    apply_model_id_defaults,
    build_model_workflow,
    resolve_task_type,
)
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
        task_type: Optional[str] = None,
        image_input: str = "image",
        workspace: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ) -> WebRTCSession:
        """Create a WebRTC streaming session.

        Exactly one of ``workflow`` or ``model_id`` must be provided.

        When ``model_id`` is given, a minimal single-model workflow is built
        under the hood and the session delivers the raw serialized predictions
        dict alongside each frame instead of raw metadata (see the model_id
        example below). The model's task type (object-detection,
        instance-segmentation, classification, multi-label-classification,
        keypoint-detection, semantic-segmentation) is resolved automatically
        via a Roboflow API lookup and the matching Workflow model block is
        selected. Pass ``task_type`` explicitly to skip that lookup (see
        below). VLMs are not supported in model_id mode (no generic model
        block exists for them) - stream them by passing a full ``workflow``
        instead.

        Args:
            source: Stream source (WebcamSource, RTSPSource, VideoFileSource, or ManualSource)
            workflow: Either a workflow ID (str) or workflow specification (dict).
                Mutually exclusive with ``model_id``.
            model_id: Roboflow model ID or alias (e.g. "rfdetr-nano"). When
                provided, a Workflow wrapping this model is built
                automatically and ``on_frame`` handlers receive
                ``(frame, data)`` where ``data`` is the raw serialized
                predictions dict exactly as received from the server, or None
                when predictions are unavailable for the frame - check before
                use. The dict's shape follows the model's task type; for
                detection-family models it is inference-response-shaped, so
                ``sv.Detections.from_inference(data)`` /
                ``sv.KeyPoints.from_inference(data)`` work directly.
                Mutually exclusive with ``workflow``.
            task_type: Optional model task type. Only valid together with
                ``model_id``. When omitted (the default), the task type is
                resolved automatically from the Roboflow API. Pass it explicitly
                to skip the network lookup (useful for air-gapped / self-hosted
                deployments). Must be one of: "object-detection",
                "instance-segmentation", "classification",
                "multi-label-classification", "keypoint-detection",
                "semantic-segmentation".
            image_input: Name of the image input in the workflow
            workspace: Workspace name (required if workflow is an ID string)
            config: Stream configuration (output routing, FPS, TURN server, etc.)

        Returns:
            WebRTCSession context manager

        Raises:
            InvalidParameterError: If workflow/model_id/workspace/task_type parameters are invalid

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
                if data is None:     # predictions unavailable for this frame
                    return
                detections = sv.Detections.from_inference(data)
                annotated = sv.BoxAnnotator().annotate(frame.copy(), detections)
                cv2.imshow("Frame", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    session.close()

            session.run()

            # Other task types work the same way: only the shape of `data`
            # changes (e.g. classification predictions instead of boxes). The
            # task type is auto-resolved; pass task_type="classification"
            # (or another supported type) to skip the lookup on air-gapped /
            # self-hosted servers. For VLM models, pass a full workflow=
            # instead of model_id=.
        """
        # Validate that exactly one of workflow / model_id is provided
        if (workflow is None) == (model_id is None):
            raise InvalidParameterError(
                "Exactly one of 'workflow' or 'model_id' must be provided "
                "(got both or neither)."
            )

        model_mode = model_id is not None

        if not model_mode and task_type is not None:
            raise InvalidParameterError(
                "'task_type' is only valid together with 'model_id', "
                "not with 'workflow'."
            )

        if model_mode:
            resolved_task_type = resolve_task_type(
                model_id, task_type, api_key=self._api_key
            )
            workflow_config = {
                "workflow_specification": build_model_workflow(
                    model_id, resolved_task_type
                )
            }
            config = apply_model_id_defaults(config)
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
