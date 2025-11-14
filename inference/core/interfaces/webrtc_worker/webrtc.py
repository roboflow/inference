import asyncio
import datetime
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
)
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    WebRTCConfigurationError,
)
from inference.core.interfaces.camera.entities import VideoFrameProducer
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCData,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.entities import (
    DataOutputMode,
    StreamOutputMode,
    WebRTCOutput,
    WebRTCVideoMetadata,
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.utils import (
    detect_image_output,
    process_frame,
)
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.core_steps.common.serializers import (
    serialize_wildcard_kind,
)
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.usage_tracking.collector import usage_collector

logging.getLogger("aiortc").setLevel(logging.WARNING)


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
        has_video_track: bool = True,
        declared_fps: float = 30,
        termination_date: Optional[datetime.datetime] = None,
        terminate_event: Optional[asyncio.Event] = None,
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

        self.has_video_track = has_video_track
        self.stream_output = stream_output
        self.data_channel: Optional[RTCDataChannel] = None

        if data_output is None:
            self.data_output = None
            self._data_mode = DataOutputMode.NONE
        elif isinstance(data_output, list):
            self.data_output = [f for f in data_output if f]
            if self.data_output == ["*"]:
                self._data_mode = DataOutputMode.ALL
            elif len(self.data_output) == 0:
                self._data_mode = DataOutputMode.NONE
            else:
                self._data_mode = DataOutputMode.SPECIFIC
        else:
            raise WebRTCConfigurationError(
                f"data_output must be list or None, got {type(data_output).__name__}"
            )

        self._ensure_workflow_specification(workflow_configuration, api_key)
        self._validate_output_fields(workflow_configuration)

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

    async def _send_data_output(
        self,
        workflow_output: Dict[str, Any],
        frame_timestamp: datetime.datetime,
        frame: VideoFrame,
        errors: List[str],
    ):
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
            errors=errors.copy(),
        )

        if self._data_mode == DataOutputMode.NONE:
            self.data_channel.send(json.dumps(webrtc_output.model_dump()))
            return

        if self._data_mode == DataOutputMode.ALL:
            fields_to_send = list(workflow_output.keys())
        else:
            fields_to_send = self.data_output

        serialized_outputs = {}

        for field_name in fields_to_send:
            if field_name not in workflow_output:
                webrtc_output.errors.append(
                    f"Requested output '{field_name}' not found in workflow outputs"
                )
                continue

            output_data = workflow_output[field_name]

            if self._data_mode == DataOutputMode.ALL and isinstance(
                output_data, WorkflowImageData
            ):
                continue

            try:
                serialized_value = serialize_wildcard_kind(output_data)
                serialized_outputs[field_name] = serialized_value
            except Exception as e:
                webrtc_output.errors.append(f"{field_name}: {e}")
                serialized_outputs[field_name] = {"__serialization_error__": str(e)}

        # Only set serialized_output_data if we have data to send
        if serialized_outputs:
            webrtc_output.serialized_output_data = serialized_outputs

        self.data_channel.send(json.dumps(webrtc_output.model_dump(mode="json")))

    async def process_frames_data_only(self):
        """Process frames for data extraction only, without video track output.

        This is used when stream_output=[] and no video track is needed.
        """
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        logger.info("Starting data-only frame processing")

        try:
            while (
                self.track
                and self.track.readyState != "ended"
                and not self._stop_processing
            ):
                if self._check_termination():
                    break

                # Drain queue if using PlayerStreamTrack (RTSP)
                if isinstance(self.track, PlayerStreamTrack):
                    while self.track._queue.qsize() > 30:
                        self.track._queue.get_nowait()

                frame: VideoFrame = await self.track.recv()
                self._received_frames += 1
                frame_timestamp = datetime.datetime.now()

                workflow_output, _, errors = await self._process_frame_async(
                    frame=frame,
                    frame_id=self._received_frames,
                    render_output=False,
                    include_errors_on_frame=False,
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

    @staticmethod
    def _ensure_workflow_specification(
        workflow_configuration: WorkflowConfiguration, api_key: str
    ) -> None:
        has_specification = workflow_configuration.workflow_specification is not None
        has_workspace_and_workflow_id = (
            workflow_configuration.workspace_name is not None
            and workflow_configuration.workflow_id is not None
        )

        if not has_specification and not has_workspace_and_workflow_id:
            raise WebRTCConfigurationError(
                "Either 'workflow_specification' or both 'workspace_name' and 'workflow_id' must be provided"
            )

        if not has_specification and has_workspace_and_workflow_id:
            try:
                workflow_configuration.workflow_specification = (
                    get_workflow_specification(
                        api_key=api_key,
                        workspace_id=workflow_configuration.workspace_name,
                        workflow_id=workflow_configuration.workflow_id,
                    )
                )
                workflow_configuration.workspace_name = None
                workflow_configuration.workflow_id = None
            except Exception as e:
                raise WebRTCConfigurationError(
                    f"Failed to fetch workflow specification from API: {str(e)}"
                )

    def _validate_output_fields(
        self, workflow_configuration: WorkflowConfiguration
    ) -> None:
        if workflow_configuration.workflow_specification is None:
            return

        workflow_outputs = workflow_configuration.workflow_specification.get(
            "outputs", []
        )
        available_output_names = [o.get("name") for o in workflow_outputs]

        if self._data_mode == DataOutputMode.SPECIFIC:
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

        if self.stream_output and self.stream_output not in available_output_names:
            raise WebRTCConfigurationError(
                f"Invalid stream_output field: '{self.stream_output}'. "
                f"Available workflow outputs: {available_output_names}"
            )

    async def _process_frame_async(
        self,
        frame: VideoFrame,
        frame_id: int,
        stream_output: Optional[str] = None,
        render_output: bool = True,
        include_errors_on_frame: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[VideoFrame], List[str]]:
        """Async wrapper for process_frame using executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            process_frame,
            frame,
            frame_id,
            self._inference_pipeline,
            stream_output,
            render_output,
            include_errors_on_frame,
        )


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
        has_video_track: bool = True,
        declared_fps: float = 30,
        termination_date: Optional[datetime.datetime] = None,
        terminate_event: Optional[asyncio.Event] = None,
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
            has_video_track=has_video_track,
            declared_fps=declared_fps,
            termination_date=termination_date,
            terminate_event=terminate_event,
        )

    async def _auto_detect_stream_output(
        self, frame: VideoFrame, frame_id: int
    ) -> None:
        workflow_output_for_detect, _, _ = await self._process_frame_async(
            frame=frame,
            frame_id=frame_id,
            render_output=False,
            include_errors_on_frame=False,
        )
        detected_output = detect_image_output(workflow_output_for_detect)
        if detected_output:
            self.stream_output = detected_output
            logger.info(f"Auto-detected stream_output: {detected_output}")
        else:
            logger.warning("No image output detected, will use fallback")
            self.stream_output = ""

    async def recv(self):
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

        if self.stream_output is None and self._received_frames == 1:
            await self._auto_detect_stream_output(frame, self._received_frames)

        workflow_output, new_frame, errors = await self._process_frame_async(
            frame=frame,
            frame_id=self._received_frames,
            stream_output=self.stream_output,
            render_output=True,
            include_errors_on_frame=True,
        )

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        await self._send_data_output(workflow_output, frame_timestamp, frame, errors)

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

    if webrtc_request.processing_timeout is not None:
        try:
            time_limit_seconds = int(webrtc_request.processing_timeout)
            termination_date = datetime.datetime.now() + datetime.timedelta(
                seconds=time_limit_seconds - 1
            )
            logger.info("Setting termination date to %s", termination_date)
        except (TypeError, ValueError):
            pass
    if webrtc_request.stream_output is None:
        stream_mode = StreamOutputMode.AUTO_DETECT
        stream_field = None
    elif len(webrtc_request.stream_output) == 0:
        stream_mode = StreamOutputMode.NO_VIDEO
        stream_field = None
    else:
        filtered = [s for s in webrtc_request.stream_output if s]
        if filtered:
            stream_mode = StreamOutputMode.SPECIFIC_FIELD
            stream_field = filtered[0]
        else:
            stream_mode = StreamOutputMode.NO_VIDEO
            stream_field = None

    if webrtc_request.data_output is None or len(webrtc_request.data_output) == 0:
        data_fields = None
    elif webrtc_request.data_output == ["*"]:
        data_fields = ["*"]
    else:
        data_fields = webrtc_request.data_output

    try:
        should_send_video = stream_mode != StreamOutputMode.NO_VIDEO

        if should_send_video:
            video_processor = VideoTransformTrackWithLoop(
                asyncio_loop=asyncio_loop,
                workflow_configuration=webrtc_request.workflow_configuration,
                api_key=webrtc_request.api_key,
                data_output=data_fields,
                stream_output=stream_field,
                has_video_track=True,
                declared_fps=webrtc_request.declared_fps,
                termination_date=termination_date,
                terminate_event=terminate_event,
            )
        else:
            # No video track - use base VideoFrameProcessor
            video_processor = VideoFrameProcessor(
                asyncio_loop=asyncio_loop,
                workflow_configuration=webrtc_request.workflow_configuration,
                api_key=webrtc_request.api_key,
                data_output=data_fields,
                stream_output=None,
                has_video_track=False,
                declared_fps=webrtc_request.declared_fps,
                termination_date=termination_date,
                terminate_event=terminate_event,
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

    if webrtc_request.webrtc_config is not None:
        ice_servers = []
        for ice_server in webrtc_request.webrtc_config.iceServers:
            if not ice_server.username:
                continue
            ice_servers.append(
                RTCIceServer(
                    urls=ice_server.urls,
                    username=ice_server.username,
                    credential=ice_server.credential,
                )
            )
    else:
        ice_servers = None
    peer_connection = RTCPeerConnectionWithLoop(
        configuration=RTCConfiguration(iceServers=ice_servers) if ice_servers else None,
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
            logger.info("Adding video track to send back")
            peer_connection.addTrack(video_processor)
        else:
            # No video track - start data-only processing task
            logger.info("Starting data-only processing (no video track)")
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

        @channel.on("message")
        def on_message(message):
            logger.info("Data channel message received: %s", message)
            try:
                message_data = WebRTCData(**json.loads(message))
            except json.JSONDecodeError:
                logger.error("Failed to decode webrtc data payload: %s", message)
                return

            # Handle stream_output changes
            if message_data.stream_output is not None:
                if not video_processor.has_video_track:
                    logger.warning(
                        "Cannot change stream_output: video track was not initialized. "
                        "stream_output must be set at initialization to enable video."
                    )
                else:
                    if len(message_data.stream_output) == 0:
                        video_processor.stream_output = None
                    else:
                        filtered = [s for s in message_data.stream_output if s]
                        video_processor.stream_output = (
                            filtered[0] if filtered else None
                        )

            # Handle data_output changes (always allowed)
            if message_data.data_output is not None:
                video_processor.data_output = message_data.data_output
                if (
                    message_data.data_output is None
                    or len(message_data.data_output) == 0
                ):
                    video_processor._data_mode = DataOutputMode.NONE
                elif message_data.data_output == ["*"]:
                    video_processor._data_mode = DataOutputMode.ALL
                else:
                    video_processor._data_mode = DataOutputMode.SPECIFIC

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
