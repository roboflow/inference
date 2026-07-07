"""WebRTC client for the Inference SDK."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

import requests

from inference_sdk.config import RF_API_BASE_URL
from inference_sdk.http.errors import InvalidParameterError
from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias
from inference_sdk.utils.decorators import experimental
from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.session import WebRTCSession
from inference_sdk.webrtc.sources import StreamSource

# Map project task type (as returned by the Roboflow /ort endpoint, and matching
# the model-registry keys in inference/models/utils.py) to the Workflow block
# `type` literal that wraps a Roboflow model of that task type, plus the name of
# the block output holding the primary predictions. Every block below exposes a
# "predictions" output (verified against each block manifest's describe_outputs),
# so the selector is uniform; the mapping keeps the predictions-output name
# explicit so the generated workflow can adapt if a future block renames it.
TASK_TYPE_TO_BLOCK = {
    "object-detection": {
        "block_type": "roboflow_core/roboflow_object_detection_model@v2",
        "predictions_output": "predictions",
    },
    "instance-segmentation": {
        "block_type": "roboflow_core/roboflow_instance_segmentation_model@v2",
        "predictions_output": "predictions",
    },
    "classification": {
        "block_type": "roboflow_core/roboflow_classification_model@v2",
        "predictions_output": "predictions",
    },
    "multi-label-classification": {
        "block_type": "roboflow_core/roboflow_multi_label_classification_model@v2",
        "predictions_output": "predictions",
    },
    "keypoint-detection": {
        "block_type": "roboflow_core/roboflow_keypoint_detection_model@v2",
        "predictions_output": "predictions",
    },
    "semantic-segmentation": {
        "block_type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "predictions_output": "predictions",
    },
}

# Timeout (seconds) for the Roboflow /ort task-type lookup.
_TASK_TYPE_LOOKUP_TIMEOUT = 10


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
        classification, instance/semantic-segmentation, keypoint-detection,
        multi-label-classification) is resolved automatically via a Roboflow API
        lookup and the matching Workflow model block is selected. Pass
        ``task_type`` explicitly to skip that lookup (see below).

        Args:
            source: Stream source (WebcamSource, RTSPSource, VideoFileSource, or ManualSource)
            workflow: Either a workflow ID (str) or workflow specification (dict).
                Mutually exclusive with ``model_id``.
            model_id: Roboflow model ID (e.g. "rfdetr-nano"). Works for any
                supported task type, not just detection. When provided, a
                Workflow wrapping this model is built automatically and
                ``on_frame`` handlers receive ``(frame, data)`` where ``data`` is
                the raw serialized predictions dict exactly as received from the
                server. The dict is inference-response-shaped; convert it in your
                handler with the appropriate ``supervision`` helper for the model
                type (e.g. ``sv.Detections.from_inference(data)`` for
                detection/segmentation). Mutually exclusive with ``workflow``.
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
                detections = sv.Detections.from_inference(data)
                annotated = sv.BoxAnnotator().annotate(frame.copy(), detections)
                cv2.imshow("Frame", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    session.close()

            session.run()

            # Classification / segmentation / keypoint models work the same way:
            # only the shape of `data` changes to match the model's task type
            # (e.g. classification predictions instead of boxes). The task type
            # is auto-resolved; pass task_type="classification" to skip the
            # lookup on air-gapped / self-hosted servers.
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
            resolved_task_type = self._resolve_task_type(model_id, task_type)
            workflow_config = {
                "workflow_specification": self._build_model_workflow(
                    model_id, resolved_task_type
                )
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

    def _resolve_task_type(
        self, model_id: str, task_type: Optional[str]
    ) -> str:
        """Resolve a model's task type, either from ``task_type`` or the API.

        When ``task_type`` is provided it is validated against
        ``TASK_TYPE_TO_BLOCK`` and returned as-is (no network call). When None,
        the model alias is resolved, the model ID is split into dataset/version,
        and the Roboflow ``/ort`` endpoint is queried for ``ort.type``.

        Args:
            model_id: Roboflow model ID or alias (e.g. "rfdetr-nano").
            task_type: Explicit task type, or None to auto-resolve.

        Returns:
            A task type string that is a key of ``TASK_TYPE_TO_BLOCK``.

        Raises:
            InvalidParameterError: If ``task_type`` is unsupported, the model ID
                has no version after de-aliasing, or the resolved task type is
                unsupported.
            RuntimeError: If the API lookup fails (network / parse error).
        """
        if task_type is not None:
            if task_type not in TASK_TYPE_TO_BLOCK:
                raise InvalidParameterError(
                    f"Unsupported task_type '{task_type}'. Supported task types: "
                    f"{sorted(TASK_TYPE_TO_BLOCK)}."
                )
            return task_type

        resolved_model_id = resolve_roboflow_model_alias(model_id)
        parts = resolved_model_id.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise InvalidParameterError(
                f"Could not derive a dataset/version pair from model_id "
                f"'{model_id}' (resolved to '{resolved_model_id}'). Model IDs "
                "must be of the form 'dataset/version'. If the task type cannot "
                "be resolved this way, pass task_type= explicitly (one of "
                f"{sorted(TASK_TYPE_TO_BLOCK)})."
            )
        dataset_id, version_id = parts

        url = f"{RF_API_BASE_URL}/ort/{dataset_id}/{version_id}"
        try:
            response = requests.get(
                url,
                params={
                    "api_key": self._api_key,
                    "nocache": "true",
                    "device": "sdk",
                    "dynamic": "true",
                },
                timeout=_TASK_TYPE_LOOKUP_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            resolved = payload["ort"]["type"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve task type for model_id '{model_id}' "
                f"(resolved to '{resolved_model_id}') via the Roboflow API: "
                f"{e.__class__.__name__}: {e}. You can bypass this lookup by "
                "passing task_type= explicitly (one of "
                f"{sorted(TASK_TYPE_TO_BLOCK)})."
            ) from e

        if resolved not in TASK_TYPE_TO_BLOCK:
            raise InvalidParameterError(
                f"Roboflow API reported task type '{resolved}' for model_id "
                f"'{model_id}', which is not supported for model_id streaming. "
                f"Supported task types: {sorted(TASK_TYPE_TO_BLOCK)}."
            )
        return resolved

    @staticmethod
    def _build_model_workflow(model_id: str, task_type: str) -> dict:
        """Build a minimal single-model workflow spec wrapping a model ID.

        The Workflow model block is chosen from ``TASK_TYPE_TO_BLOCK`` based on
        ``task_type``. The block's primary predictions output is always exposed
        under the "predictions" JsonField name (the selector points at whatever
        the block calls its predictions output), so the session's raw-dict
        contract is task-agnostic.

        Args:
            model_id: Roboflow model ID (e.g. "rfdetr-nano")
            task_type: Model task type; must be a key of ``TASK_TYPE_TO_BLOCK``.

        Returns:
            Workflow specification dict producing "predictions" and "image" outputs

        Raises:
            InvalidParameterError: If ``task_type`` is not supported.
        """
        block = TASK_TYPE_TO_BLOCK.get(task_type)
        if block is None:
            raise InvalidParameterError(
                f"Unsupported task_type '{task_type}'. Supported task types: "
                f"{sorted(TASK_TYPE_TO_BLOCK)}."
            )
        predictions_output = block["predictions_output"]
        return {
            "version": "1.0",
            "inputs": [{"type": "InferenceImage", "name": "image"}],
            "steps": [
                {
                    "type": block["block_type"],
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
                    "selector": f"$steps.model.{predictions_output}",
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
