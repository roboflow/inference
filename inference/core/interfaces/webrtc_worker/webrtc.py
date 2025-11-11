import asyncio
import base64
import datetime
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay, PlayerStreamTrack
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av import logging as av_logging
from pydantic import ValidationError

from inference.core import logger
from inference.core.env import (
    WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
)
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    WebRTCConfigurationError,
)
from inference.core.interfaces.camera.entities import VideoFrame as InferenceVideoFrame
from inference.core.interfaces.camera.entities import VideoFrameProducer
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCData,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCOutput,
    WebRTCOutputMode,
    WebRTCVideoMetadata,
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.utils import process_frame
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.usage_tracking.collector import usage_collector

logging.getLogger("aiortc").setLevel(logging.WARNING)


def serialize_workflow_output(
    output_data: Any, is_explicit_request: bool
) -> Tuple[Any, Optional[str]]:
    """Serialize a workflow output value recursively.

    Args:
        output_data: The workflow output value to serialize
        is_explicit_request: True if field was explicitly requested in data_output

    Returns (serialized_value, error_message)
    - serialized_value: The value ready for JSON serialization, or None if
      skipped/failed
    - error_message: Error string if serialization failed, None otherwise

    Image serialization rules:
    - Images are NEVER serialized UNLESS explicitly requested in data_output list
    - If explicit: serialize to base64 JPEG (quality 85)
    - If implicit (data_output=None): skip images

    Handles nested structures recursively (dicts, lists) to ensure all complex
    types are properly serialized.
    """
    try:
        # Handle WorkflowImageData (convert to base64 only if explicit)
        if isinstance(output_data, WorkflowImageData):
            if not is_explicit_request:
                # Skip images when listing all outputs (data_output=None)
                return None, None  # Skip without error

            # Explicitly requested - serialize to base64 JPEG
            try:
                np_image = output_data.numpy_image
                # Encode as JPEG with quality 85 (good quality, much smaller than PNG)
                success, buffer = cv2.imencode(
                    ".jpg", np_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if success:
                    base64_image = base64.b64encode(buffer).decode("utf-8")
                    return f"data:image/jpeg;base64,{base64_image}", None
                else:
                    return None, "Failed to encode image as JPEG"
            except Exception as e:
                return None, f"Failed to serialize image: {str(e)}"

        # Handle sv.Detections (use existing serializer)
        elif isinstance(output_data, sv.Detections):
            try:
                parsed_detections = serialise_sv_detections(output_data)
                return parsed_detections, None
            except Exception as e:
                return None, f"Failed to serialize detections: {str(e)}"

        # Handle dict (serialize recursively)
        elif isinstance(output_data, dict):
            return _serialize_collection(
                output_data.items(), is_explicit_request, as_dict=True
            )

        # Handle list (serialize recursively)
        elif isinstance(output_data, list):
            return _serialize_collection(
                enumerate(output_data), is_explicit_request, as_dict=False
            )

        # Handle primitives (str, int, float, bool)
        elif isinstance(output_data, (str, int, float, bool, type(None))):
            return output_data, None

        # Handle numpy types
        elif isinstance(output_data, (np.integer, np.floating)):
            return output_data.item(), None

        # Handle numpy arrays
        elif isinstance(output_data, np.ndarray):
            try:
                return output_data.tolist(), None
            except Exception as e:
                return None, f"Failed to serialize numpy array: {str(e)}"

        # Unknown type - convert to string as fallback
        else:
            return str(output_data), None

    except Exception as e:
        return None, f"Unexpected error serializing output: {str(e)}"


def _serialize_collection(
    items, is_explicit_request: bool, as_dict: bool
) -> Tuple[Any, Optional[str]]:
    """Helper to serialize dict or list collections recursively.

    Args:
        items: Iterator of (key, value) pairs for dict or (index, value) for list
        is_explicit_request: Whether the parent field was explicitly requested
        as_dict: True to return dict, False to return list

    Returns (serialized_collection, error_message)

    Note: If serialization fails for some fields, those fields are replaced with
    error placeholders and the collection is still returned with valid fields.
    The error message lists which fields failed.

    Error placeholder format:
    - For dicts: {"__serialization_error__": "error message"} (key identifies field)
    - For lists: {"__serialization_error__": "error message", "__field__": "index"}
    """
    result = {} if as_dict else []
    errors = []

    for key_or_idx, value in items:
        serialized_value, error = serialize_workflow_output(value, is_explicit_request)

        if error:
            # Store error info and add placeholder
            errors.append(f"{key_or_idx}: {error}")

            if as_dict:
                # For dict: key already identifies the field
                result[key_or_idx] = {"__serialization_error__": error}
            else:
                # For list: include index in placeholder
                result.append(
                    {"__serialization_error__": error, "__field__": str(key_or_idx)}
                )
        elif serialized_value is not None:
            if as_dict:
                result[key_or_idx] = serialized_value
            else:
                result.append(serialized_value)
        # else: skip None values (e.g., images when not explicit)

    # Return result with placeholders + error message listing failed fields
    if errors:
        error_message = f"Partial serialization - errors in: {'; '.join(errors)}"
        return result, error_message
    return result, None


class RTCPeerConnectionWithLoop(RTCPeerConnection):
    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop


class VideoFrameProcessor:
    """Base class for processing video frames through workflow.

    Can be used independently for data-only processing (no video track output)
    or as a base for VideoTransformTrackWithLoop when video output is needed.
    """

    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        workflow_configuration: WorkflowConfiguration,
        api_key: str,
        data_output: Optional[List[str]] = None,
        stream_output: Optional[str] = None,
        output_mode: WebRTCOutputMode = WebRTCOutputMode.BOTH,
        declared_fps: float = 30,
        termination_date: Optional[datetime.datetime] = None,
        terminate_event: Optional[asyncio.Event] = None,
        use_data_channel_frames: bool = False,
    ):
        self._loop = asyncio_loop
        self._termination_date = termination_date
        self._terminate_event = terminate_event
        self.track: Optional[RemoteStreamTrack] = None
        self._track_active: bool = False
        self._av_logging_set: bool = False
        self._received_frames = 0
        self._declared_fps = declared_fps
        self._stop_processing = False
        self.use_data_channel_frames = use_data_channel_frames
        self._data_frame_queue: "asyncio.Queue[Optional[VideoFrame]]" = asyncio.Queue()

        self.output_mode = output_mode
        self.stream_output = stream_output
        self.data_channel: Optional[RTCDataChannel] = None

        # Normalize data_output to avoid edge cases
        if data_output is None:
            self.data_output = None
        elif isinstance(data_output, list):
            self.data_output = [f for f in data_output if f]
        else:
            raise WebRTCConfigurationError(
                f"data_output must be list or None, got {type(data_output).__name__}"
            )

        # Validate that workflow is specified either by specification or workspace/workflow_id
        has_specification = workflow_configuration.workflow_specification is not None
        has_workspace_and_id = (
            workflow_configuration.workspace_name is not None
            and workflow_configuration.workflow_id is not None
        )

        if not has_specification and not has_workspace_and_id:
            raise WebRTCConfigurationError(
                "Either 'workflow_specification' or both 'workspace_name' and 'workflow_id' must be provided"
            )

        # Fetch workflow_specification from API if not provided directly
        if not has_specification and has_workspace_and_id:
            try:
                logger.info(
                    f"Fetching workflow specification for workspace={workflow_configuration.workspace_name}, "
                    f"workflow_id={workflow_configuration.workflow_id}"
                )
                workflow_configuration.workflow_specification = (
                    get_workflow_specification(
                        api_key=api_key,
                        workspace_id=workflow_configuration.workspace_name,
                        workflow_id=workflow_configuration.workflow_id,
                    )
                )
                # Clear workspace_name and workflow_id after fetch to avoid conflicts
                # InferencePipeline requires these to be mutually exclusive with workflow_specification
                workflow_configuration.workspace_name = None
                workflow_configuration.workflow_id = None
            except Exception as e:
                raise WebRTCConfigurationError(
                    f"Failed to fetch workflow specification from API: {str(e)}"
                )

        # Validate data_output and stream_output against workflow specification
        if workflow_configuration.workflow_specification is not None:
            workflow_outputs = workflow_configuration.workflow_specification.get(
                "outputs", []
            )
            available_output_names = [o.get("name") for o in workflow_outputs]

            # Validate data_output fields
            if self.data_output is not None and len(self.data_output) > 0:
                invalid_fields = [
                    field
                    for field in self.data_output
                    if field not in available_output_names
                ]

                if invalid_fields:
                    raise WebRTCConfigurationError(
                        f"Invalid data_output fields: {invalid_fields}. "
                        f"Available workflow outputs: {available_output_names}"
                    )

            # Validate stream_output field (if explicitly specified and not empty)
            if self.stream_output and self.stream_output not in available_output_names:
                raise WebRTCConfigurationError(
                    f"Invalid stream_output field: '{self.stream_output}'. "
                    f"Available workflow outputs: {available_output_names}"
                )

        self._inference_pipeline = InferencePipeline.init_with_workflow(
            video_reference=VideoFrameProducer,
            workflow_specification=workflow_configuration.workflow_specification,
            workspace_name=workflow_configuration.workspace_name,
            workflow_id=workflow_configuration.workflow_id,
            api_key=api_key,
            image_input_name=workflow_configuration.image_input_name,
            workflows_parameters=workflow_configuration.workflows_parameters,
            workflows_thread_pool_workers=workflow_configuration.workflows_thread_pool_workers,
            cancel_thread_pool_tasks_on_exit=workflow_configuration.cancel_thread_pool_tasks_on_exit,
            video_metadata_input_name=workflow_configuration.video_metadata_input_name,
        )

    def set_track(self, track: RemoteStreamTrack):
        if not self.track:
            self.track = track

    def close(self):
        self._track_active = False
        self._stop_processing = True

    def _check_termination(self):
        """Check if we should terminate based on timeout"""
        if (
            self._termination_date
            and self._termination_date < datetime.datetime.now()
            and self._terminate_event
            and not self._terminate_event.is_set()
        ):
            logger.info("Timeout reached, terminating inference pipeline")
            self._terminate_event.set()
            return True
        return False

    def _process_frame_data_only(
        self, frame: VideoFrame, frame_id: int
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Process frame through workflow without rendering visuals.

        Returns (workflow_output, errors)
        """
        np_image = frame.to_ndarray(format="bgr24")
        workflow_output = {}
        errors = []

        try:
            video_frame = InferenceVideoFrame(
                image=np_image,
                frame_id=frame_id,
                frame_timestamp=datetime.datetime.now(),
                comes_from_video_file=False,
                fps=self._declared_fps,
                measured_fps=self._declared_fps,
            )
            workflow_output = self._inference_pipeline._on_video_frame([video_frame])[0]
        except Exception as e:
            logger.exception("Error in workflow processing")
            errors.append(str(e))

        return workflow_output, errors

    async def _send_data_output(
        self,
        workflow_output: Dict[str, Any],
        frame_timestamp: datetime.datetime,
        frame: VideoFrame,
        errors: List[str],
    ):
        """Send data via data channel based on data_output configuration.

        - data_output = None: Send all workflow outputs
        - data_output = []: Don't send any data (only metadata)
        - data_output = ["field1", "field2"]: Send only specified fields
        """
        if not self.data_channel or self.data_channel.readyState != "open":
            return

        video_metadata = WebRTCVideoMetadata(
            frame_id=self._received_frames,
            received_at=frame_timestamp.isoformat(),
            pts=frame.pts,
            time_base=frame.time_base,
            declared_fps=self._declared_fps,
        )

        webrtc_output = WebRTCOutput(
            serialized_output_data=None,
            video_metadata=video_metadata,
            errors=list(errors),  # Copy errors list
        )

        # Determine which fields to send
        if self.data_output is None:
            # Send ALL workflow outputs
            fields_to_send = list(workflow_output.keys())
        elif len(self.data_output) == 0:
            self.data_channel.send(json.dumps(webrtc_output.model_dump()))
            return
        else:
            fields_to_send = self.data_output

        # Serialize each field
        serialized_outputs = {}

        # Determine if this is an explicit request (fields listed) or implicit (all fields)
        is_all_outputs = self.data_output is None

        for field_name in fields_to_send:
            if field_name not in workflow_output:
                webrtc_output.errors.append(
                    f"Requested output '{field_name}' not found in workflow outputs"
                )
                continue

            output_data = workflow_output[field_name]

            # Determine if this field was explicitly requested
            if is_all_outputs:
                # data_output=None means listing all, so not explicit for individual fields
                is_explicit_request = False
            else:
                # Field is in the data_output list, so it's explicit
                is_explicit_request = True

            serialized_value, error = serialize_workflow_output(
                output_data=output_data,
                is_explicit_request=is_explicit_request,
            )

            if error:
                # Add error to errors list and include placeholder in output
                webrtc_output.errors.append(f"{field_name}: {error}")
                serialized_outputs[field_name] = {"__serialization_error__": error}
            elif serialized_value is not None:
                serialized_outputs[field_name] = serialized_value
            # else: serialized_value is None and no error = field was skipped (e.g., image in video track)

        # Only set serialized_output_data if we have data to send
        if serialized_outputs:
            webrtc_output.serialized_output_data = serialized_outputs

        self.data_channel.send(json.dumps(webrtc_output.model_dump()))

    async def _handle_data_channel_frame(self, message: str) -> None:
        """Handle incoming frame from upstream_frames data channel."""
        try:
            payload = json.loads(message)
            if payload.get("type") != "frame":
                logger.warning(f"Unknown message type: {payload.get('type')}")
                return
            
            frame_id = payload.get("frame_id", 0)
            image_b64 = payload.get("image", "")
            
            if frame_id % 100 == 1:
                logger.info(f"Received frame {frame_id} via data channel")
            
            # Decode base64 → JPEG bytes → numpy array
            image_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if np_image is None:
                logger.error(f"Failed to decode frame {frame_id}")
                return
            
            # Convert to VideoFrame and queue
            video_frame = VideoFrame.from_ndarray(np_image, format="bgr24")
            await self._data_frame_queue.put((frame_id, video_frame))
            
            if frame_id % 100 == 1:
                logger.info(f"Queued frame {frame_id}, queue size: {self._data_frame_queue.qsize()}")
            
        except Exception as e:
            logger.error(f"Error handling data channel frame: {e}", exc_info=True)

    async def process_frames_data_only(self):
        """Process frames for data extraction only, without video track output.

        This is used when output_mode is DATA_ONLY and no video track is needed.
        """
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        logger.info(f"Starting data-only frame processing (use_data_channel_frames={self.use_data_channel_frames})")

        try:
            while not self._stop_processing:
                if self._check_termination():
                    break

                # Get frame from appropriate source
                if self.use_data_channel_frames:
                    # Wait for frame from data channel queue
                    item = await self._data_frame_queue.get()
                    if item is None:
                        logger.info("Received stop signal from data channel")
                        break
                    frame_id, frame = item
                    self._received_frames = frame_id
                else:
                    # Get frame from media track (existing behavior)
                    if not self.track or self.track.readyState == "ended":
                        break
                    
                    # Drain queue if using PlayerStreamTrack (RTSP)
                    if isinstance(self.track, PlayerStreamTrack):
                        while self.track._queue.qsize() > 30:
                            self.track._queue.get_nowait()

                    frame = await self.track.recv()
                    self._received_frames += 1

                frame_timestamp = datetime.datetime.now()

                # Process workflow without rendering
                loop = asyncio.get_running_loop()
                workflow_output, errors = await loop.run_in_executor(
                    None,
                    self._process_frame_data_only,
                    frame,
                    self._received_frames,
                )

                # Send data via data channel
                await self._send_data_output(
                    workflow_output, frame_timestamp, frame, errors
                )

        except asyncio.CancelledError:
            logger.info("Data-only processing cancelled")
        except MediaStreamError:
            logger.info("Stream ended in data-only processing")
        except Exception as exc:
            logger.error("Error in data-only processing: %s", exc)


class VideoTransformTrackWithLoop(VideoStreamTrack, VideoFrameProcessor):
    """Video track that processes frames through workflow and sends video back.

    Inherits from both VideoStreamTrack (for WebRTC video track functionality)
    and VideoFrameProcessor (for workflow processing logic).
    """

    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        workflow_configuration: WorkflowConfiguration,
        api_key: str,
        data_output: Optional[List[str]] = None,
        stream_output: Optional[str] = None,
        output_mode: WebRTCOutputMode = WebRTCOutputMode.BOTH,
        declared_fps: float = 30,
        termination_date: Optional[datetime.datetime] = None,
        terminate_event: Optional[asyncio.Event] = None,
        use_data_channel_frames: bool = False,
        *args,
        **kwargs,
    ):
        VideoStreamTrack.__init__(self, *args, **kwargs)
        VideoFrameProcessor.__init__(
            self,
            asyncio_loop=asyncio_loop,
            workflow_configuration=workflow_configuration,
            api_key=api_key,
            data_output=data_output,
            stream_output=stream_output,
            output_mode=output_mode,
            declared_fps=declared_fps,
            termination_date=termination_date,
            terminate_event=terminate_event,
            use_data_channel_frames=use_data_channel_frames,
        )

    async def recv(self):
        """Called by WebRTC to get the next frame to send.

        This method processes frames through the workflow and returns
        the processed video frame for transmission.
        """
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        # Check if we should terminate
        if self._check_termination():
            raise MediaStreamError("Processing terminated due to timeout")

        # Drain queue if using PlayerStreamTrack (RTSP)
        if isinstance(self.track, PlayerStreamTrack):
            while self.track._queue.qsize() > 30:
                self.track._queue.get_nowait()

        frame: VideoFrame = await self.track.recv()
        self._received_frames += 1
        frame_timestamp = datetime.datetime.now()

        # Process frame through workflow WITH rendering (for video output)
        loop = asyncio.get_running_loop()
        workflow_output, new_frame, errors, detected_output = (
            await loop.run_in_executor(
                None,
                process_frame,
                frame,
                self._received_frames,
                self._inference_pipeline,
                self.stream_output,
            )
        )

        # Update stream_output if it was auto-detected (only when None)
        if self.stream_output is None and detected_output is not None:
            self.stream_output = detected_output
            logger.info(f"Auto-detected and set stream_output to: {detected_output}")

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        # Send data via data channel if needed (BOTH or DATA_ONLY modes)
        if self.output_mode in [WebRTCOutputMode.BOTH, WebRTCOutputMode.DATA_ONLY]:
            await self._send_data_output(
                workflow_output, frame_timestamp, frame, errors
            )

        return new_frame


async def _wait_ice_complete(peer_connection: RTCPeerConnectionWithLoop, timeout=2.0):
    if peer_connection.iceGatheringState == "complete":
        return
    fut = asyncio.get_running_loop().create_future()

    @peer_connection.on("icegatheringstatechange")
    def _():
        if not fut.done() and peer_connection.iceGatheringState == "complete":
            fut.set_result(True)

    try:
        await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError:
        pass


async def init_rtc_peer_connection_with_loop(
    webrtc_request: WebRTCWorkerRequest,
    send_answer: Callable[[WebRTCWorkerResult], None],
    asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> RTCPeerConnectionWithLoop:
    termination_date = None
    terminate_event = asyncio.Event()

    if WEBRTC_MODAL_FUNCTION_TIME_LIMIT is not None:
        try:
            time_limit_seconds = int(WEBRTC_MODAL_FUNCTION_TIME_LIMIT)
            termination_date = datetime.datetime.now() + datetime.timedelta(
                seconds=time_limit_seconds - 1
            )
            logger.info("Setting termination date to %s", termination_date)
        except (TypeError, ValueError):
            pass
    output_mode = webrtc_request.output_mode
    stream_output = None

    # Normalize stream_output
    if webrtc_request.stream_output:
        filtered = [s for s in webrtc_request.stream_output if s]
        stream_output = filtered[0] if filtered else None

    # Handle data_output as list
    # - None or not provided: send all outputs
    # - []: send nothing
    # - ["field1", "field2"]: send only those fields
    data_output = (
        webrtc_request.data_output if webrtc_request.data_output is not None else None
    )

    # Determine if we should send video back based on output mode
    should_send_video = output_mode in [
        WebRTCOutputMode.VIDEO_ONLY,
        WebRTCOutputMode.BOTH,
    ]

    try:
        # For DATA_ONLY mode, we use VideoFrameProcessor directly (no video track)
        # For other modes, we use VideoTransformTrackWithLoop (includes video track)
        if should_send_video:
            video_processor = VideoTransformTrackWithLoop(
                asyncio_loop=asyncio_loop,
                workflow_configuration=webrtc_request.workflow_configuration,
                api_key=webrtc_request.api_key,
                data_output=data_output,
                stream_output=stream_output,
                output_mode=output_mode,
                declared_fps=webrtc_request.declared_fps,
                termination_date=termination_date,
                terminate_event=terminate_event,
                use_data_channel_frames=webrtc_request.use_data_channel_frames,
            )
        else:
            # DATA_ONLY or OFF mode - use base VideoFrameProcessor
            video_processor = VideoFrameProcessor(
                asyncio_loop=asyncio_loop,
                workflow_configuration=webrtc_request.workflow_configuration,
                api_key=webrtc_request.api_key,
                data_output=data_output,
                stream_output=stream_output,
                output_mode=output_mode,
                declared_fps=webrtc_request.declared_fps,
                termination_date=termination_date,
                terminate_event=terminate_event,
                use_data_channel_frames=webrtc_request.use_data_channel_frames,
            )
    except (
        ValidationError,
        MissingApiKeyError,
        KeyError,
        NotImplementedError,
    ) as error:
        send_answer(
            WebRTCWorkerResult(
                exception_type=error.__class__.__name__,
                error_message="Could not decode InferencePipeline initialisation command payload.",
            )
        )
        return
    except WebRTCConfigurationError as error:
        send_answer(
            WebRTCWorkerResult(
                exception_type=error.__class__.__name__,
                error_message=str(error),
            )
        )
        return
    except RoboflowAPINotAuthorizedError:
        send_answer(
            WebRTCWorkerResult(
                exception_type=RoboflowAPINotAuthorizedError.__name__,
                error_message="Invalid API key used or API key is missing. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key",
            )
        )
        return
    except RoboflowAPINotNotFoundError:
        send_answer(
            WebRTCWorkerResult(
                exception_type=RoboflowAPINotNotFoundError.__name__,
                error_message="Requested Roboflow resources (models / workflows etc.) not available or wrong API key used.",
            )
        )
        return
    except WorkflowSyntaxError as error:
        send_answer(
            WebRTCWorkerResult(
                exception_type=WorkflowSyntaxError.__name__,
                error_message=str(error),
                error_context=str(error.context),
                inner_error=str(error.inner_error),
            )
        )
        return
    except Exception as error:
        send_answer(
            WebRTCWorkerResult(
                exception_type=error.__class__.__name__,
                error_message=str(error),
            )
        )
        return

    if webrtc_request.webrtc_turn_config:
        turn_server = RTCIceServer(
            urls=[webrtc_request.webrtc_turn_config.urls],
            username=webrtc_request.webrtc_turn_config.username,
            credential=webrtc_request.webrtc_turn_config.credential,
        )
        peer_connection = RTCPeerConnectionWithLoop(
            configuration=RTCConfiguration(iceServers=[turn_server]),
            asyncio_loop=asyncio_loop,
        )
    else:
        peer_connection = RTCPeerConnectionWithLoop(
            asyncio_loop=asyncio_loop,
        )

    relay = MediaRelay()

    player: Optional[MediaPlayer] = None
    if webrtc_request.rtsp_url:
        if webrtc_request.rtsp_url == WEBRTC_MODAL_RTSP_PLACEHOLDER:
            webrtc_request.rtsp_url = WEBRTC_MODAL_RTSP_PLACEHOLDER_URL
        logger.info("Processing RTSP URL: %s", webrtc_request.rtsp_url)
        player = MediaPlayer(
            webrtc_request.rtsp_url,
            format="rtsp",
            options={
                "rtsp_transport": "tcp",
                "rtsp_flags": "prefer_tcp",
                "stimeout": "2000000",  # 2s socket timeout
            },
        )
        video_processor.set_track(track=player.video)

        # Only add video track if we should send video back
        if should_send_video:
            peer_connection.addTrack(video_processor)
        else:
            # For DATA_ONLY, start data-only processing task
            logger.info("Starting data-only processing for RTSP stream")
            asyncio.create_task(video_processor.process_frames_data_only())

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("Track received from client")
        relayed_track = relay.subscribe(
            track,
            buffered=False if webrtc_request.webrtc_realtime_processing else True,
        )
        video_processor.set_track(track=relayed_track)

        # Only add video track back if we should send video
        if should_send_video:
            logger.info(f"Output mode: {output_mode} - Adding video track to send back")
            peer_connection.addTrack(video_processor)
        else:
            # For DATA_ONLY, start data-only processing task
            logger.info(
                f"Output mode: {output_mode} - Starting data-only processing (no video track)"
            )
            asyncio.create_task(video_processor.process_frames_data_only())

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            if video_processor.track:
                logger.info("Stopping video processor track")
                video_processor.track.stop()
            video_processor.close()
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
            terminate_event.set()
        logger.info("'connectionstatechange' event handler finished")

    @peer_connection.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info("Data channel '%s' received", channel.label)

        # Handle upstream frames channel (client sending frames to server)
        if channel.label == "upstream_frames":
            logger.info("Upstream frames channel established, starting data-only processing")
            
            @channel.on("message")
            def on_frame_message(message):
                asyncio.create_task(video_processor._handle_data_channel_frame(message))
            
            # Start processing immediately since we won't get a media track
            if webrtc_request.use_data_channel_frames and not should_send_video:
                asyncio.create_task(video_processor.process_frames_data_only())
            
            return

        # Handle inference control channel (bidirectional communication)
        @channel.on("message")
        def on_message(message):
            logger.info("Data channel message received: %s", message)
            try:
                message_data = WebRTCData(**json.loads(message))
            except json.JSONDecodeError:
                logger.error("Failed to decode webrtc data payload: %s", message)
                return

            # Handle output changes (which workflow output to send)
            if message_data.stream_output is not None:
                video_processor.stream_output = message_data.stream_output or None

            if message_data.data_output is not None:
                video_processor.data_output = message_data.data_output

        video_processor.data_channel = channel

    await peer_connection.setRemoteDescription(
        RTCSessionDescription(
            sdp=webrtc_request.webrtc_offer.sdp, type=webrtc_request.webrtc_offer.type
        )
    )
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)

    logger.debug(f"WebRTC connection status: {peer_connection.connectionState}")

    await _wait_ice_complete(peer_connection, timeout=2.0)

    send_answer(
        WebRTCWorkerResult(
            answer={
                "type": peer_connection.localDescription.type,
                "sdp": peer_connection.localDescription.sdp,
            },
        )
    )

    await terminate_event.wait()
    if player:
        logger.info("Stopping player")
        player.video.stop()
    if peer_connection.connectionState != "closed":
        logger.info("Closing WebRTC connection")
        await peer_connection.close()
    if video_processor.track:
        logger.info("Stopping video processor track")
        video_processor.track.stop()
    video_processor.close()
    await usage_collector.async_push_usage_payloads()
    logger.info("WebRTC peer connection closed")
