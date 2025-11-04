import asyncio
import datetime
import json
import logging
from typing import Callable, Optional

import supervision as sv
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av import logging as av_logging
from pydantic import ValidationError

from inference.core import logger
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.interfaces.camera.entities import VideoFrameProducer
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCData,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCOutput,
    WebRTCVideoMetadata,
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.utils import process_frame
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

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


class VideoTransformTrackWithLoop(VideoStreamTrack):
    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        workflow_configuration: WorkflowConfiguration,
        api_key: str,
        data_output: Optional[str] = None,
        stream_output: Optional[str] = None,
        declared_fps: float = 30,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop

        self.track: Optional[RemoteStreamTrack] = None
        self._track_active: bool = False

        self._av_logging_set: bool = False

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
        self.data_output = data_output
        self.stream_output = stream_output
        self.data_channel: Optional[RTCDataChannel] = None
        self._received_frames = 0
        self._declared_fps = declared_fps

    def set_track(
        self,
        track: RemoteStreamTrack,
    ):
        if not self.track:
            self.track = track

    def close(self):
        self._track_active = False

    async def recv(self):
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        frame: VideoFrame = await self.track.recv()

        self._received_frames += 1
        frame_timestamp = datetime.datetime.now()
        loop = asyncio.get_running_loop()
        workflow_output, new_frame, errors = await loop.run_in_executor(
            None,
            process_frame,
            frame,
            self._received_frames,
            self._inference_pipeline,
            self.stream_output,
        )

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        if self.data_channel and self.data_channel.readyState == "open":
            video_metadata = WebRTCVideoMetadata(
                frame_id=self._received_frames,
                received_at=frame_timestamp.isoformat(),
                pts=new_frame.pts,
                time_base=new_frame.time_base,
                declared_fps=self._declared_fps,
            )
            webrtc_output = WebRTCOutput(
                output_name=None,
                serialized_output_data=None,
                video_metadata=video_metadata,
                errors=errors,
            )
            if self.data_output and self.data_output in workflow_output:
                workflow_output = workflow_output[self.data_output]
                serialized_data = None
                if isinstance(workflow_output, WorkflowImageData):
                    webrtc_output.errors.append(
                        f"Selected data output '{self.data_output}' contains image, please use video output instead"
                    )
                elif isinstance(workflow_output, sv.Detections):
                    try:
                        parsed_detections = serialise_sv_detections(workflow_output)
                        serialized_data = json.dumps(parsed_detections)
                    except Exception:
                        webrtc_output.errors.append(
                            f"Failed to serialise output: {self.data_output}"
                        )
                elif isinstance(workflow_output, dict):
                    try:
                        serialized_data = json.dumps(workflow_output)
                    except Exception:
                        webrtc_output.errors.append(
                            f"Failed to serialise output: {self.data_output}"
                        )
                else:
                    serialized_data = str(workflow_output)
                if serialized_data is not None:
                    webrtc_output.output_name = self.data_output
                    webrtc_output.serialized_output_data = serialized_data
                    self.data_channel.send(json.dumps(webrtc_output.model_dump()))

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
    stream_output = None
    if webrtc_request.stream_output:
        # TODO: UI sends None as stream_output for wildcard outputs
        stream_output = webrtc_request.stream_output[0] or ""
    data_output = None
    if webrtc_request.data_output:
        data_output = webrtc_request.data_output[0]

    try:
        video_transform_track = VideoTransformTrackWithLoop(
            asyncio_loop=asyncio_loop,
            workflow_configuration=webrtc_request.workflow_configuration,
            api_key=webrtc_request.api_key,
            data_output=data_output,
            stream_output=stream_output,
            declared_fps=webrtc_request.declared_fps,
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

    closed = asyncio.Event()
    relay = MediaRelay()

    player: Optional[MediaPlayer] = None
    if webrtc_request.rtsp_url:
        logger.info("Processing RTSP URL: %s", webrtc_request.rtsp_url)
        player = MediaPlayer(
            webrtc_request.rtsp_url,
            options={
                "rtsp_transport": "tcp",  # avoid UDP loss/reorder
                "stimeout": "2000000",  # 2s socket timeout (microseconds)
                # "rw_timeout": "2000000",  # (optional) I/O timeout, if supported
                # "max_delay": "0",         # (optional) may reduce latency on some sources
                "rtsp_flags": "prefer_tcp",  # alternative to rtsp_transport=tcp
                # Avoid 'fflags=nobuffer' unless your encoder has NO B-frames
                # "fflags": "nobuffer",
                # "flags": "low_delay",
            },
        )
        video_transform_track.set_track(
            track=relay.subscribe(
                player.video,
                buffered=False if webrtc_request.webrtc_realtime_processing else True,
            )
        )
        peer_connection.addTrack(video_transform_track)

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("track received")
        video_transform_track.set_track(
            track=relay.subscribe(
                track,
                buffered=False if webrtc_request.webrtc_realtime_processing else True,
            )
        )
        peer_connection.addTrack(video_transform_track)

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            if video_transform_track.track:
                logger.info("Stopping video transform track")
                video_transform_track.track.stop()
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
            closed.set()
        logger.info("'connectionstatechange' event handler finished")

    @peer_connection.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info("Data channel '%s' received", channel.label)

        @channel.on("message")
        def on_message(message):
            logger.info("Data channel message received: %s", message)
            try:
                message = WebRTCData(**json.loads(message))
            except json.JSONDecodeError:
                logger.error("Failed to decode webrtc data payload: %s", message)
                return
            if message.stream_output is not None:
                video_transform_track.stream_output = message.stream_output or None
            if message.data_output is not None:
                video_transform_track.data_output = message.data_output or None

        video_transform_track.data_channel = channel

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

    await closed.wait()
    if player:
        logger.info("Stopping player")
        player.video.stop()
    if peer_connection.connectionState != "closed":
        logger.info("Closing WebRTC connection")
        await peer_connection.close()
    if video_transform_track.track:
        logger.info("Stopping video transform track")
        video_transform_track.track.stop()
    logger.info("WebRTC peer connection closed")
