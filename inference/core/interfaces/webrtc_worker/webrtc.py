import asyncio
import base64
import datetime
import fractions
import av
import json
import logging
import queue
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import supervision as sv

from aioice import ice
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay, PlayerStreamTrack
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av import logging as av_logging
from pydantic import ValidationError

from inference.core import logger
from inference.core.env import (
    WEBRTC_DATA_CHANNEL_ACK_WINDOW,
    WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY,
    WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT,
    WEBRTC_MODAL_PUBLIC_STUN_SERVERS,
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
    WEBRTC_MODAL_SHUTDOWN_RESERVE,
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
    VideoFileUploadState,
    WebRTCOutput,
    WebRTCVideoMetadata,
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.serializers import serialize_for_webrtc
from inference.core.interfaces.webrtc_worker.sources.file import (
    ThreadedVideoFileTrack,
    VideoFileUploadHandler,
)
from inference.core.interfaces.webrtc_worker.utils import (
    detect_image_output,
    get_cv2_rotation_code,
    get_video_rotation,
    parse_video_file_chunk,
    process_frame,
    rotate_video_frame,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.errors import WorkflowError, WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.usage_tracking.collector import usage_collector

logging.getLogger("aiortc").setLevel(logging.WARNING)

# WebRTC data channel chunking configuration
CHUNK_SIZE = 48 * 1024  # 48KB - safe for all WebRTC implementations


def create_chunked_binary_message(
    frame_id: int, chunk_index: int, total_chunks: int, payload: bytes
) -> bytes:
    """Create a binary message with standard 12-byte header.

    Format: [frame_id: 4][chunk_index: 4][total_chunks: 4][payload: N]
    All integers are uint32 little-endian.
    """
    header = struct.pack("<III", frame_id, chunk_index, total_chunks)
    return header + payload


async def wait_for_buffer_drain(
    data_channel: RTCDataChannel,
    timeout: float = 30.0,
    heartbeat_callback: Optional[Callable[[], None]] = None,
    low_threshold: Optional[int] = None,
) -> bool:
    """Wait for data channel buffer to drain below threshold, with timeout.

    We use a low threshold (1/4 of limit) instead of just below the limit to avoid
    hysteresis - constantly triggering this wait after sending just a few chunks.

    And we wait WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY to avoid starving the
    event loop.
    """
    if low_threshold is None:
        low_threshold = WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT // 4

    start_time = asyncio.get_event_loop().time()

    while data_channel.bufferedAmount > low_threshold:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            logger.error("[BUFFER_DRAIN] Timeout after %.1fs", timeout)
            return False
        if data_channel.readyState != "open":
            logger.error("[BUFFER_DRAIN] Channel closed: %s", data_channel.readyState)
            return False
        if heartbeat_callback:
            heartbeat_callback()
        await asyncio.sleep(WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY)

    return True


async def send_chunked_data(
    data_channel: RTCDataChannel,
    frame_id: int,
    payload_bytes: bytes,
    chunk_size: int = CHUNK_SIZE,
    heartbeat_callback: Optional[Callable[[], None]] = None,
    buffer_timeout: float = 120.0,
) -> bool:
    """Send payload via data channel with chunking and backpressure.

    We chunk large payloads because WebRTC data channels have message size limits.
    We apply backpressure (wait for buffer to drain) to avoid overwhelming the
    network and causing ICE connection failures.

    Heads up: buffer_timeout needs to be higher than WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY!
    Otherwise we will timeout ourselves.
    """
    if data_channel.readyState != "open":
        return False

    payload_size = len(payload_bytes)
    total_chunks = (payload_size + chunk_size - 1) // chunk_size
    view = memoryview(payload_bytes)
    high_threshold = WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT

    for chunk_index in range(total_chunks):
        if data_channel.readyState != "open":
            logger.error(
                "[SEND_CHUNKED] Channel closed at chunk %d/%d",
                chunk_index,
                total_chunks,
            )
            return False

        start = chunk_index * chunk_size
        end = min(start + chunk_size, payload_size)
        chunk_data = view[start:end]

        message = create_chunked_binary_message(
            frame_id, chunk_index, total_chunks, chunk_data
        )

        if data_channel.bufferedAmount > high_threshold:
            if not await wait_for_buffer_drain(
                data_channel, buffer_timeout, heartbeat_callback
            ):
                logger.error(
                    "[SEND_CHUNKED] Buffer drain failed at chunk %d/%d",
                    chunk_index,
                    total_chunks,
                )
                return False

        data_channel.send(message)

        if heartbeat_callback:
            heartbeat_callback()
        await asyncio.sleep(0.001)

    return True


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
        model_manager: Optional[ModelManager] = None,
        data_output: Optional[List[str]] = None,
        stream_output: Optional[str] = None,
        has_video_track: bool = True,
        declared_fps: float = 30,
        termination_date: Optional[datetime.datetime] = None,
        terminate_event: Optional[asyncio.Event] = None,
        heartbeat_callback: Optional[Callable[[], None]] = None,
        realtime_processing: bool = True,
        is_preview: bool = False,
    ):
        self._loop = asyncio_loop
        self._termination_date = termination_date
        self._terminate_event = terminate_event
        self.track: Optional[MediaStreamTrack] = None
        self._track_active: bool = False
        self._av_logging_set: bool = False
        self._received_frames = 0
        self._declared_fps = declared_fps
        self._stop_processing = False
        self.heartbeat_callback = heartbeat_callback

        self.has_video_track = has_video_track
        self.stream_output = stream_output
        self.data_channel: Optional[RTCDataChannel] = None

        # Video file upload support
        self.video_upload_handler: Optional[VideoFileUploadHandler] = None
        self._track_ready_event: asyncio.Event = asyncio.Event()
        self.realtime_processing = realtime_processing
        self._rotation_code: Optional[int] = None

        # Optional receiver-paced flow control (enabled only after first ACK is received)
        self._ack_last: int = 0
        # If ack=1 and window=4, server may produce/send up to frame 5.
        # Configurable via WEBRTC_DATACHANNEL_ACK_WINDOW env var.
        self._ack_window: int = WEBRTC_DATA_CHANNEL_ACK_WINDOW
        self._ack_event: asyncio.Event = asyncio.Event()

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
            model_manager=model_manager,
            _is_preview=is_preview,
        )

    def set_track(self, track: MediaStreamTrack, rotation_code: Optional[int] = None):
        if not self.track:
            self.track = track
            self._rotation_code = rotation_code
            self._track_ready_event.set()

    async def close(self):
        self._track_active = False
        self._stop_processing = True
        # Clean up video upload handler if present
        if self.video_upload_handler is not None:
            await self.video_upload_handler.cleanup()

    def record_ack(self, ack: int) -> None:
        """Record cumulative ACK from the client.

        ACK semantics: client has fully handled all frames <= ack.
        Backwards compatible: pacing is disabled until we receive the first ACK.
        """
        try:
            ack_int = int(ack)
        except (TypeError, ValueError):
            logger.warning("Invalid ACK value: %s", ack)
            return
        if ack_int < 0:
            logger.warning("Invalid ACK value: %s", ack)
            return
        if ack_int > self._ack_last:
            if ack_int % 100 == 1:
                logger.info("ACK received: %s", ack_int)
            self._ack_last = ack_int
            self._ack_event.set()

    async def _wait_for_ack_window(self, next_frame_id: int) -> None:
        """Block frame production when too far ahead of client ACKs."""
        if self.realtime_processing or self._ack_last == 0:
            return

        wait_counter = 0
        while not self._stop_processing and next_frame_id > (
            self._ack_last + self._ack_window
        ):
            if self._check_termination():
                return
            if self.heartbeat_callback:
                self.heartbeat_callback()

            self._ack_event.clear()
            try:
                await asyncio.wait_for(self._ack_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                wait_counter += 1
                if wait_counter % 5 == 1:
                    logger.info(
                        "Waiting for ACK window (next=%d, ack_last=%d, window=%d)",
                        next_frame_id,
                        self._ack_last,
                        self._ack_window,
                    )

    def _check_termination(self):
        """Check if we should terminate based on timeout"""
        if (
            self._termination_date
            and self._termination_date < datetime.datetime.now()
            or self._terminate_event
            and self._terminate_event.is_set()
        ):
            logger.info("Timeout reached, terminating inference pipeline")
            self._terminate_event.set()
            return True
        return False

    @staticmethod
    def serialize_outputs_sync(
        fields_to_send: List[str],
        workflow_output: Dict[str, Any],
        data_output_mode: DataOutputMode,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Serialize workflow outputs for WebRTC transmission."""
        serialized = {}
        serialization_errors = []

        for field_name in fields_to_send:
            if field_name not in workflow_output:
                serialization_errors.append(f"Output '{field_name}' not found")
                continue

            output_data = workflow_output[field_name]

            if data_output_mode == DataOutputMode.ALL and isinstance(
                output_data, WorkflowImageData
            ):
                continue

            try:
                serialized[field_name] = serialize_for_webrtc(output_data)
            except Exception as e:
                serialization_errors.append(f"{field_name}: {e}")
                serialized[field_name] = {"__serialization_error__": str(e)}
                logger.error("[SERIALIZE] Error: %s - %s", field_name, e)

        return serialized, serialization_errors

    async def _send_data_output(
        self,
        workflow_output: Dict[str, Any],
        frame_timestamp: datetime.datetime,
        frame: VideoFrame,
        errors: List[str],
    ):
        frame_id = self._received_frames

        if not self.data_channel or self.data_channel.readyState != "open":
            return

        video_metadata = WebRTCVideoMetadata(
            frame_id=frame_id,
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
            json_bytes = await asyncio.to_thread(
                lambda: json.dumps(webrtc_output.model_dump()).encode("utf-8")
            )
            await send_chunked_data(
                self.data_channel,
                frame_id,
                json_bytes,
                heartbeat_callback=self.heartbeat_callback,
            )
            return

        if self._data_mode == DataOutputMode.ALL:
            fields_to_send = list(workflow_output.keys())
        else:
            fields_to_send = self.data_output

        serialized_outputs, serialization_errors = await asyncio.to_thread(
            VideoFrameProcessor.serialize_outputs_sync,
            fields_to_send,
            workflow_output,
            self._data_mode,
        )

        webrtc_output.errors.extend(serialization_errors)
        if serialized_outputs:
            webrtc_output.serialized_output_data = serialized_outputs

        def compress_json():
            import gzip

            json_bytes = json.dumps(webrtc_output.model_dump(mode="json")).encode(
                "utf-8"
            )
            return gzip.compress(json_bytes, compresslevel=6)

        compressed_bytes = await asyncio.to_thread(compress_json)

        success = await send_chunked_data(
            self.data_channel,
            frame_id,
            compressed_bytes,
            heartbeat_callback=self.heartbeat_callback,
        )
        if not success:
            logger.error("[SEND_OUTPUT] Frame %d failed", frame_id)

    async def _send_processing_complete(self):
        """Send final message indicating processing is complete."""
        if not self.data_channel or self.data_channel.readyState != "open":
            return

        completion_output = WebRTCOutput(
            processing_complete=True,
            video_metadata=WebRTCVideoMetadata(
                frame_id=self._received_frames,
                received_at=datetime.datetime.now().isoformat(),
            ),
        )
        json_bytes = json.dumps(completion_output.model_dump()).encode("utf-8")
        await send_chunked_data(
            self.data_channel, self._received_frames + 1, json_bytes
        )

    async def process_frames_data_only(self):
        """Process frames for data extraction only, without video track output."""
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        try:
            while not self._stop_processing:
                await self._wait_for_ack_window(next_frame_id=self._received_frames + 1)
                if self._check_termination():
                    break
                if self.heartbeat_callback:
                    self.heartbeat_callback()
                if not self.track or self.track.readyState == "ended":
                    break

                # Drain queue for realtime RTSP
                if (
                    isinstance(self.track, PlayerStreamTrack)
                    and self.realtime_processing
                ):
                    while self.track._queue.qsize() > 30:
                        self.track._queue.get_nowait()

                frame = await self.track.recv()
                self._received_frames += 1
                frame_timestamp = datetime.datetime.now()

                workflow_output, _, errors = await self._process_frame_async(
                    frame=frame,
                    frame_id=self._received_frames,
                    render_output=False,
                    include_errors_on_frame=False,
                )

                await self._send_data_output(
                    workflow_output, frame_timestamp, frame, errors
                )

        except asyncio.CancelledError:
            raise
        except MediaStreamError:
            pass  # Expected when video ends
        except Exception as exc:
            logger.error(
                "[DATA_ONLY] Error at frame %d: %s", self._received_frames, exc
            )
        finally:
            await self._send_processing_complete()

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

        if self._rotation_code is not None:
            frame = rotate_video_frame(frame, self._rotation_code)

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
        model_manager: Optional[ModelManager] = None,
        data_output: Optional[List[str]] = None,
        stream_output: Optional[str] = None,
        has_video_track: bool = True,
        declared_fps: float = 30,
        termination_date: Optional[datetime.datetime] = None,
        terminate_event: Optional[asyncio.Event] = None,
        heartbeat_callback: Optional[Callable[[], None]] = None,
        realtime_processing: bool = True,
        is_preview: bool = False,
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
            model_manager=model_manager,
            heartbeat_callback=heartbeat_callback,
            realtime_processing=realtime_processing,
            is_preview=is_preview,
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
        recv_start = time.time()

        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        if self.heartbeat_callback:
            self.heartbeat_callback()

        # Check if we should terminate
        if self._check_termination():
            logger.warning("[RECV] Termination triggered, stopping")
            raise MediaStreamError("Processing terminated due to timeout")

        # Wait for track to be ready (video file upload case)
        if self.track is None:
            logger.info("[RECV] Track is None, waiting for track_ready_event...")
            await self._track_ready_event.wait()
            if self.track is None:
                logger.error("[RECV] Track still None after wait!")
                raise MediaStreamError("Track not available after wait")

        # Optional ACK pacing: block producing the next frame if we're too far ahead.
        await self._wait_for_ack_window(next_frame_id=self._received_frames + 1)

        # Drain queue if using PlayerStreamTrack (RTSP/video file)
        if isinstance(self.track, PlayerStreamTrack) and self.realtime_processing:
            queue_size = self.track._queue.qsize()
            if queue_size > 30:
                drained = 0
                while self.track._queue.qsize() > 30:
                    self.track._queue.get_nowait()
                    drained += 1
                logger.info(
                    "[RECV] Drained %d frames from queue (was %d)", drained, queue_size
                )

        frame: VideoFrame = await self.track.recv()

        self._received_frames += 1
        frame_id = self._received_frames
        frame_timestamp = datetime.datetime.now()

        if self.stream_output is None and frame_id == 1:
            await self._auto_detect_stream_output(frame, frame_id)

        workflow_output, new_frame, errors = await self._process_frame_async(
            frame=frame,
            frame_id=frame_id,
            stream_output=self.stream_output,
            render_output=True,
            include_errors_on_frame=True,
        )

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        await self._send_data_output(workflow_output, frame_timestamp, frame, errors)

        if errors:
            logger.warning("[RECV] Frame %d errors: %s", frame_id, errors)

        return new_frame


async def _wait_ice_complete(peer_connection: RTCPeerConnectionWithLoop, timeout=2.0):
    if peer_connection.iceGatheringState == "complete":
        logger.info("ICE gathering state already complete")
        return
    fut = asyncio.get_running_loop().create_future()

    @peer_connection.on("icegatheringstatechange")
    def _():
        logger.info(
            "ICE gathering state changed to %s", peer_connection.iceGatheringState
        )
        if not fut.done() and peer_connection.iceGatheringState == "complete":
            fut.set_result(True)

    try:
        logger.info("Waiting for ICE gathering to complete...")
        await asyncio.wait_for(fut, timeout)
        logger.info("ICE gathering completed")
    except asyncio.TimeoutError:
        logger.info("ICE gathering did not complete in %s seconds", timeout)
        pass


async def init_rtc_peer_connection_with_loop(
    webrtc_request: WebRTCWorkerRequest,
    send_answer: Callable[[WebRTCWorkerResult], None],
    asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
    model_manager: Optional[ModelManager] = None,
    shutdown_reserve: int = WEBRTC_MODAL_SHUTDOWN_RESERVE,
    heartbeat_callback: Optional[Callable[[], None]] = None,
) -> RTCPeerConnectionWithLoop:
    logger.warning(
        "=" * 60 + "\n"
        "[WEBRTC_SESSION] STARTING NEW SESSION\n"
        "  stream_output=%s\n"
        "  data_output=%s\n"
        "  timeout=%s\n" + "=" * 60,
        webrtc_request.stream_output,
        webrtc_request.data_output,
        webrtc_request.processing_timeout,
    )
    logger.info("Initializing RTC peer connection with loop")
    # ice._mdns is instantiated on the module level, it has a lock that is bound to the event loop
    # avoid RuntimeError: asyncio.locks.Lock is bound to a different event loop
    if hasattr(ice, "_mdns"):
        if hasattr(ice._mdns, "lock"):
            logger.info("Removing lock from aioice.ice._mdns")
            delattr(ice._mdns, "lock")
    else:
        logger.warning(
            "aioice.ice implementation was changed, _mdns attribute is not available"
        )

    termination_date = None
    terminate_event = asyncio.Event()

    if webrtc_request.processing_timeout is not None:
        try:
            time_limit_seconds = int(webrtc_request.processing_timeout)
            datetime_now = webrtc_request.processing_session_started
            if datetime_now is None:
                datetime_now = datetime.datetime.now()
            termination_date = datetime_now + datetime.timedelta(
                seconds=time_limit_seconds - shutdown_reserve
            )
            logger.info(
                "Setting termination date to %s (%s seconds from %s)",
                termination_date.isoformat(),
                time_limit_seconds,
                datetime_now.isoformat(),
            )
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
                model_manager=model_manager,
                api_key=webrtc_request.api_key,
                data_output=data_fields,
                stream_output=stream_field,
                has_video_track=True,
                declared_fps=webrtc_request.declared_fps,
                termination_date=termination_date,
                terminate_event=terminate_event,
                heartbeat_callback=heartbeat_callback,
                realtime_processing=webrtc_request.webrtc_realtime_processing,
                is_preview=webrtc_request.is_preview,
            )
        else:
            # No video track - use base VideoFrameProcessor
            video_processor = VideoFrameProcessor(
                asyncio_loop=asyncio_loop,
                workflow_configuration=webrtc_request.workflow_configuration,
                model_manager=model_manager,
                api_key=webrtc_request.api_key,
                data_output=data_fields,
                stream_output=None,
                has_video_track=False,
                declared_fps=webrtc_request.declared_fps,
                termination_date=termination_date,
                terminate_event=terminate_event,
                heartbeat_callback=heartbeat_callback,
                realtime_processing=webrtc_request.webrtc_realtime_processing,
                is_preview=webrtc_request.is_preview,
            )
    except (
        ValidationError,
        MissingApiKeyError,
        KeyError,
        NotImplementedError,
    ) as error:
        # heartbeat to indicate caller error
        heartbeat_callback()
        send_answer(
            WebRTCWorkerResult(
                exception_type=error.__class__.__name__,
                error_message="Could not decode InferencePipeline initialisation command payload.",
            )
        )
        return
    except WebRTCConfigurationError as error:
        # heartbeat to indicate caller error
        heartbeat_callback()
        send_answer(
            WebRTCWorkerResult(
                exception_type=error.__class__.__name__,
                error_message=str(error),
            )
        )
        return
    except RoboflowAPINotAuthorizedError:
        # heartbeat to indicate caller error
        heartbeat_callback()
        send_answer(
            WebRTCWorkerResult(
                exception_type=RoboflowAPINotAuthorizedError.__name__,
                error_message="Invalid API key used or API key is missing. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key",
            )
        )
        return
    except RoboflowAPINotNotFoundError:
        # heartbeat to indicate caller error
        heartbeat_callback()
        send_answer(
            WebRTCWorkerResult(
                exception_type=RoboflowAPINotNotFoundError.__name__,
                error_message="Requested Roboflow resources (models / workflows etc.) not available or wrong API key used.",
            )
        )
        return
    except WorkflowSyntaxError as error:
        # heartbeat to indicate caller error
        heartbeat_callback()
        send_answer(
            WebRTCWorkerResult(
                exception_type=WorkflowSyntaxError.__name__,
                error_message=str(error),
                error_context=str(error.context),
                inner_error=str(error.inner_error),
            )
        )
        return
    except WorkflowError as error:
        # heartbeat to indicate caller error
        heartbeat_callback()
        send_answer(
            WebRTCWorkerResult(
                exception_type=WorkflowError.__name__,
                error_message=str(error),
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
            ice_servers.append(
                RTCIceServer(
                    urls=ice_server.urls,
                    username=ice_server.username,
                    credential=ice_server.credential,
                )
            )
        # Always add public stun servers (if specified)
        if WEBRTC_MODAL_PUBLIC_STUN_SERVERS:
            for stun_server in WEBRTC_MODAL_PUBLIC_STUN_SERVERS.split(","):
                try:
                    ice_servers.append(RTCIceServer(urls=stun_server.strip()))
                except Exception as e:
                    logger.warning(
                        "Failed to add public stun server '%s': %s", stun_server, e
                    )
    else:
        ice_servers = None
    peer_connection = RTCPeerConnectionWithLoop(
        configuration=RTCConfiguration(iceServers=ice_servers) if ice_servers else None,
        asyncio_loop=asyncio_loop,
    )

    relay = MediaRelay()

    # Add video track early for SDP negotiation when stream_output is requested
    # The track source will be set later by the appropriate handler (RTSP, on_track, video_upload)
    if should_send_video:
        logger.info("Adding video track early for SDP negotiation")
        peer_connection.addTrack(video_processor)

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

        # For DATA_ONLY mode, start data-only processing task
        if not should_send_video:
            logger.info("Starting data-only processing for RTSP stream")
            asyncio.create_task(video_processor.process_frames_data_only())

    elif webrtc_request.mjpeg_url:
        logger.info("Processing MJPEG URL: %s", webrtc_request.mjpeg_url)
        player = MediaPlayer(webrtc_request.mjpeg_url)
        video_processor.set_track(track=player.video)

        if not should_send_video:
            logger.info("Starting data-only processing for MJPEG stream")
            asyncio.create_task(video_processor.process_frames_data_only())

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("Track received from client")
        relayed_track = relay.subscribe(
            track,
            buffered=False if webrtc_request.webrtc_realtime_processing else True,
        )
        video_processor.set_track(track=relayed_track)

        # For DATA_ONLY mode, start data-only processing task
        if not should_send_video:
            logger.info("Starting data-only processing (no video track)")
            asyncio.create_task(video_processor.process_frames_data_only())

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        state = peer_connection.connectionState
        ice_state = peer_connection.iceConnectionState
        logger.warning(
            "[CONNECTION_STATE] Changed to: %s (ICE state: %s, "
            "frames_received: %d, data_channel: %s)",
            state,
            ice_state,
            video_processor._received_frames,
            (
                video_processor.data_channel.readyState
                if video_processor.data_channel
                else "N/A"
            ),
        )
        if state in {"failed", "closed"}:
            logger.error(
                "[CONNECTION_STATE] FATAL: Connection %s! ICE=%s, "
                "frames_processed=%d. Cleaning up...",
                state,
                ice_state,
                video_processor._received_frames,
            )
            if video_processor.track:
                logger.info("[CONNECTION_STATE] Stopping video processor track")
                video_processor.track.stop()
            await video_processor.close()
            logger.info("[CONNECTION_STATE] Stopping WebRTC peer")
            await peer_connection.close()
            terminate_event.set()

    # Monitor ICE connection state - consent expires after ~30s without STUN refresh
    @peer_connection.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        state = peer_connection.iceConnectionState
        conn_state = peer_connection.connectionState
        logger.warning(
            "[ICE_STATE] Changed to: %s (connection state: %s, "
            "frames_received: %d, data_channel: %s)",
            state,
            conn_state,
            video_processor._received_frames,
            (
                video_processor.data_channel.readyState
                if video_processor.data_channel
                else "N/A"
            ),
        )

        if state == "failed":
            logger.error(
                "[ICE_STATE] FAILED! This typically means STUN consent expired. "
                "Causes: (1) Event loop starvation preventing aioice from sending "
                "STUN packets, (2) Network issues, (3) NAT/firewall blocking. "
                "Check logs for [BUFFER_DRAIN] timeouts or missing asyncio.sleep(0) yields."
            )
            # The connectionstatechange handler will clean up
        elif state == "disconnected":
            logger.warning(
                "[ICE_STATE] DISCONNECTED - may recover automatically. "
                "If this persists for >30s, will transition to 'failed'."
            )
        elif state == "checking":
            logger.info("[ICE_STATE] Checking connectivity candidates...")
        elif state == "connected":
            logger.info("[ICE_STATE] Successfully connected via ICE")

    def process_video_upload_message(
        message: bytes, video_processor: VideoTransformTrackWithLoop
    ):
        chunk_index, total_chunks, data = parse_video_file_chunk(message)
        video_processor.video_upload_handler.handle_chunk(
            chunk_index, total_chunks, data
        )

        video_path = video_processor.video_upload_handler.try_start_processing()
        return video_path

    @peer_connection.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info("Data channel '%s' received", channel.label)
        # Handle video file upload channel
        if channel.label == "video_upload":
            logger.info("Video upload channel established")

            video_processor.video_upload_handler = VideoFileUploadHandler()

            @channel.on("message")
            async def on_upload_message(message):
                # Keep watchdog alive during upload and keepalive pings
                if video_processor.heartbeat_callback:
                    video_processor.heartbeat_callback()

                # Ignore keepalive pings (1-byte messages)
                if len(message) <= 1:
                    channel.send(message)
                    return
                loop = asyncio.get_running_loop()
                video_path = await loop.run_in_executor(
                    None, process_video_upload_message, message, video_processor
                )
                if video_path:
                    logger.info(
                        "Video upload complete, processing: realtime=%s, path=%s",
                        webrtc_request.webrtc_realtime_processing,
                        video_path,
                    )

                    rotation = get_video_rotation(video_path)
                    rotation_code = get_cv2_rotation_code(rotation)
                    if rotation_code is not None:
                        logger.info("Video has %d° rotation, will correct", rotation)

                    if webrtc_request.webrtc_realtime_processing:
                        # We are dealing with a live video stream,
                        player = MediaPlayer(video_path, loop=False)
                        player._throttle_playback = True
                        video_processor.set_track(
                            track=player.video, rotation_code=rotation_code
                        )
                    else:
                        # we are dealing with a video file,
                        track = ThreadedVideoFileTrack(video_path)
                        video_processor.set_track(
                            track=track, rotation_code=rotation_code
                        )

                    if not should_send_video:
                        logger.info("Starting data-only processing for video file")
                        asyncio.create_task(video_processor.process_frames_data_only())

            return

        # Handle inference control channel (bidirectional communication)
        @channel.on("message")
        def on_message(message):
            try:
                message_data = WebRTCData(**json.loads(message))
            except json.JSONDecodeError:
                logger.error("Failed to decode webrtc data payload: %s", message)
                return
            # Optional ACK-based flow control (enabled only after first ACK is received)
            if message_data.ack is not None:
                video_processor.record_ack(message_data.ack)

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

    await _wait_ice_complete(peer_connection, timeout=2.0)

    logger.info(
        "Initialized RTC peer connection with loop (status: %s), sending answer",
        peer_connection.connectionState,
    )

    send_answer(
        WebRTCWorkerResult(
            answer={
                "type": peer_connection.localDescription.type,
                "sdp": peer_connection.localDescription.sdp,
            },
        )
    )

    logger.info("Answer sent, waiting for termination event")
    await terminate_event.wait()
    logger.info("Termination event received, closing WebRTC connection")
    if player:
        logger.info("Stopping player")
        player.video.stop()
    if peer_connection.connectionState != "closed":
        logger.info("Closing WebRTC connection")
        await peer_connection.close()
    if video_processor.track:
        logger.info("Stopping video processor track")
        video_processor.track.stop()
    await video_processor.close()
    await usage_collector.async_push_usage_payloads()
    logger.info("WebRTC peer connection closed")
