import asyncio
import datetime
import json
import logging
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from inference.core.interfaces.webrtc_worker.utils import (
    detect_image_output,
    parse_video_file_chunk,
    process_frame,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.core_steps.common.serializers import (
    serialize_wildcard_kind,
)
from inference.core.workflows.errors import WorkflowError, WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.usage_tracking.collector import usage_collector
from inference.core.interfaces.webrtc_worker.video_encoders import (
    register_custom_vp8_encoder, set_sender_bitrates
)

logging.getLogger("aiortc").setLevel(logging.WARNING)

# WebRTC data channel chunking configuration
CHUNK_SIZE = 48 * 1024  # 48KB - safe for all WebRTC implementations


# Register custom VP8 encoder for faster ramp-up.
register_custom_vp8_encoder()


def create_chunked_binary_message(
    frame_id: int, chunk_index: int, total_chunks: int, payload: bytes
) -> bytes:
    """Create a binary message with standard 12-byte header.

    Format: [frame_id: 4][chunk_index: 4][total_chunks: 4][payload: N]
    All integers are uint32 little-endian.
    """
    header = struct.pack("<III", frame_id, chunk_index, total_chunks)
    return header + payload


class OnDemandVideoTrack(MediaStreamTrack):
    """Lazy video track that decodes frames on-demand without pre-buffering.

    Unlike MediaPlayer which spawns a background thread to decode ALL frames
    into an unbounded queue, this class decodes one frame per recv() call.
    This keeps memory usage constant (~50-100MB) regardless of video length.

    Use this for video file processing when realtime_processing=False.
    For throttled playback (realtime_processing=True), use MediaPlayer instead.
    """

    kind = "video"

    def __init__(self, filepath: str):
        super().__init__()
        import av

        self._container = av.open(filepath)
        self._stream = self._container.streams.video[0]
        self._iterator = self._container.decode(self._stream)

    async def recv(self) -> VideoFrame:
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, lambda: next(self._iterator, None))
        if frame is None:
            self.stop()
            raise MediaStreamError("End of video file")
        return frame

    def stop(self):
        super().stop()
        if self._container:
            self._container.close()
            self._container = None


class VideoFileUploadHandler:
    """Handles video file uploads via data channel.

    Protocol: [chunk_index:u32][total_chunks:u32][payload]
    Auto-completes when all chunks received.
    """

    def __init__(self):
        self._chunks: Dict[int, bytes] = {}
        self._total_chunks: Optional[int] = None
        self._temp_file_path: Optional[str] = None
        self._state = VideoFileUploadState.IDLE
        self.upload_complete_event = asyncio.Event()

    @property
    def temp_file_path(self) -> Optional[str]:
        return self._temp_file_path

    def handle_chunk(self, chunk_index: int, total_chunks: int, data: bytes) -> None:
        """Handle a chunk. Auto-completes when all chunks received."""
        if self._total_chunks is None:
            self._total_chunks = total_chunks
            self._state = VideoFileUploadState.UPLOADING
            logger.info(f"Starting video upload: {total_chunks} chunks")

        self._chunks[chunk_index] = data

        if chunk_index % 100 == 0:
            logger.info(
                "Upload progress: %s/%s chunks", len(self._chunks), total_chunks
            )

        # Auto-complete when all chunks received
        # TODO: Handle the file writing without keeping all chunks in memory
        if len(self._chunks) == total_chunks:
            self._write_to_temp_file()
            self._state = VideoFileUploadState.COMPLETE
            self.upload_complete_event.set()

    def _write_to_temp_file(self) -> None:
        """Reassemble chunks and write to temp file."""
        import tempfile

        total_size = 0
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4", delete=False) as f:
            for i in range(self._total_chunks):
                chunk_data = self._chunks[i]
                f.write(chunk_data)
                total_size += len(chunk_data)
            self._temp_file_path = f.name

        logger.info(
            "Video upload complete: {total_size} bytes -> %s", self._temp_file_path
        )
        self._chunks.clear()  # Free memory

    def try_start_processing(self) -> Optional[str]:
        """Atomically check if upload is complete and transition to PROCESSING.

        Returns video path if processing should start, None otherwise.
        This ensures process_video_file() is only triggered once.
        """
        if self._state == VideoFileUploadState.COMPLETE:
            self._state = VideoFileUploadState.PROCESSING
            return self._temp_file_path
        return None

    async def cleanup(self) -> None:
        """Clean up temp file."""
        if self._temp_file_path:
            import os

            path_to_delete = self._temp_file_path
            self._temp_file_path = None
            try:
                await asyncio.to_thread(os.unlink, path_to_delete)
            except Exception:
                pass


async def send_chunked_data(
    data_channel: RTCDataChannel,
    frame_id: int,
    payload_bytes: bytes,
    chunk_size: int = CHUNK_SIZE,
    heartbeat_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Send payload via data channel with rate limiting.

    Automatically chunks large payloads and rate limits to prevent
    SCTP buffer overflow.

    Args:
        data_channel: RTCDataChannel to send on
        frame_id: Frame identifier
        payload_bytes: Data to send (JPEG, JSON UTF-8, etc.)
        chunk_size: Maximum chunk size (default 48KB)
    """
    if data_channel.readyState != "open":
        logger.warning(f"Cannot send response for frame {frame_id}, channel not open")
        return

    sleep_count = 0

    async def wait_for_buffer_drain() -> None:
        nonlocal sleep_count
        while data_channel.bufferedAmount > WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT:
            sleep_count += 1
            if sleep_count % 10 == 0:
                logger.debug(
                    "Waiting for data channel buffer to drain. Data channel buffer size: %s",
                    data_channel.bufferedAmount,
                )
            if heartbeat_callback:
                heartbeat_callback()
            await asyncio.sleep(WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY)

    await wait_for_buffer_drain()

    total_chunks = (
        len(payload_bytes) + chunk_size - 1
    ) // chunk_size  # Ceiling division

    if frame_id % 100 == 1:
        logger.info(
            f"Sending response for frame {frame_id}: {total_chunks} chunk(s), {len(payload_bytes)} bytes"
        )

    view = memoryview(payload_bytes)
    for chunk_index in range(total_chunks):
        if data_channel.readyState != "open":
            logger.warning("Channel closed while sending frame %s", frame_id)
            return
        await wait_for_buffer_drain()

        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(payload_bytes))
        chunk_data = view[start:end]

        message = create_chunked_binary_message(
            frame_id, chunk_index, total_chunks, chunk_data
        )
        data_channel.send(message)
        await asyncio.sleep(0)


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
        )

    def set_track(self, track: MediaStreamTrack):
        if not self.track:
            self.track = track
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
        """Block frame production when too far ahead of client ACKs.

        Allows up to (_ack_window) frames in flight beyond the last ACK.
        Only active for non-realtime processing (video file uploads).
        """
        if self.realtime_processing:
            return
        if self._ack_last == 0:
            return
        wait_counter = 0
        while not self._stop_processing and next_frame_id > (
            self._ack_last + self._ack_window
        ):
            if self._check_termination():
                return
            if self.heartbeat_callback:
                self.heartbeat_callback()
            # Wait briefly for an ACK; timeout keeps heartbeats flowing.
            self._ack_event.clear()
            try:
                await asyncio.wait_for(self._ack_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                wait_counter += 1
                if wait_counter % 5 == 1:
                    logger.info(
                        "Timeout waiting for ACK window (next_frame_id=%s, ack_last=%s, ack_window=%s)",
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
        """Serialize workflow outputs in a thread to avoid blocking the event loop."""
        serialized = {}
        serialization_errors = []

        for field_name in fields_to_send:
            if field_name not in workflow_output:
                serialization_errors.append(
                    f"Requested output '{field_name}' not found in workflow outputs"
                )
                continue

            output_data = workflow_output[field_name]

            if data_output_mode == DataOutputMode.ALL and isinstance(
                output_data, WorkflowImageData
            ):
                continue

            try:
                serialized_value = serialize_wildcard_kind(output_data)
                serialized[field_name] = serialized_value
            except Exception as e:
                serialization_errors.append(f"{field_name}: {e}")
                serialized[field_name] = {"__serialization_error__": str(e)}

        return serialized, serialization_errors

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
            # Even empty responses use binary protocol
            json_bytes = await asyncio.to_thread(
                lambda: json.dumps(webrtc_output.model_dump()).encode("utf-8")
            )
            await send_chunked_data(
                self.data_channel,
                self._received_frames,
                json_bytes,
                heartbeat_callback=self.heartbeat_callback,
            )
            return

        if self._data_mode == DataOutputMode.ALL:
            fields_to_send = list(workflow_output.keys())
        else:
            fields_to_send = self.data_output

        # Offload CPU-intensive serialization (especially image base64 encoding) to thread
        serialized_outputs, serialization_errors = await asyncio.to_thread(
            VideoFrameProcessor.serialize_outputs_sync,
            fields_to_send,
            workflow_output,
            self._data_mode,
        )
        webrtc_output.errors.extend(serialization_errors)

        # Set serialized outputs
        if serialized_outputs:
            webrtc_output.serialized_output_data = serialized_outputs

        # Send using binary chunked protocol
        json_bytes = await asyncio.to_thread(
            lambda: json.dumps(webrtc_output.model_dump(mode="json")).encode("utf-8")
        )
        await send_chunked_data(
            self.data_channel,
            self._received_frames,
            json_bytes,
            heartbeat_callback=self.heartbeat_callback,
        )

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
        logger.info(
            "Sent processing_complete signal after %s frames", self._received_frames
        )

    async def process_frames_data_only(self):
        """Process frames for data extraction only, without video track output.

        This is used when stream_output=[] and no video track is needed.
        """
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        logger.info(
            "Starting data-only frame processing. This mode is used when stream_output=[] and no video track is needed."
        )

        try:
            while not self._stop_processing:
                await self._wait_for_ack_window(next_frame_id=self._received_frames + 1)
                if self._check_termination():
                    break
                if self.heartbeat_callback:
                    self.heartbeat_callback()

                # Get frame from media track (existing behavior)
                if not self.track or self.track.readyState == "ended":
                    break

                # Drain queue if using PlayerStreamTrack (RTSP)
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

                # Send data via data channel (await for backpressure)
                await self._send_data_output(
                    workflow_output, frame_timestamp, frame, errors
                )

        except asyncio.CancelledError as exc:
            logger.info("Data-only processing cancelled: %s", exc)
        except MediaStreamError as exc:
            logger.info("Stream ended in data-only processing: %s", exc)
        except Exception as exc:
            logger.error("Error in data-only processing: %s", exc)
        finally:
            # Send completion signal to client
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

        if self.heartbeat_callback:
            self.heartbeat_callback()

        # Check if we should terminate
        if self._check_termination():
            raise MediaStreamError("Processing terminated due to timeout")

        # Wait for track to be ready (video file upload case)
        if self.track is None:
            logger.info("Waiting for track to be ready...")
            await self._track_ready_event.wait()
            if self.track is None:
                raise MediaStreamError("Track not available after wait")

        # Optional ACK pacing: block producing the next frame if we're too far ahead.
        await self._wait_for_ack_window(next_frame_id=self._received_frames + 1)

        # Drain queue if using PlayerStreamTrack (RTSP/video file)
        if isinstance(self.track, PlayerStreamTrack) and self.realtime_processing:
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
            )
            # IMPORTANT: raise GoogCC floor/ceiling to speed ramp-up


        try:
            await set_sender_bitrates(
                video_sender,
                min_bps=1_500_000,   # 1.5 Mbps floor
                max_bps=6_000_000,   # 6 Mbps ceiling
            )
            logger.info("Applied sender bitrate bounds (min=%s, max=%s)", 1_500_000, 6_000_000)
        except Exception as e:
            logger.warning("Failed to set sender bitrates: %s", e)
            


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
        logger.info("on_connectionstatechange: %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            if video_processor.track:
                logger.info("Stopping video processor track")
                video_processor.track.stop()
            await video_processor.close()
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
            terminate_event.set()

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
                    if webrtc_request.webrtc_realtime_processing:
                        # Throttled playback - use MediaPlayer
                        player = MediaPlayer(video_path, loop=False)
                        player._throttle_playback = True
                        video_processor.set_track(track=player.video)
                    else:
                        # Fast processing - use OnDemandVideoTrack (no pre-buffering)
                        track = OnDemandVideoTrack(video_path)
                        video_processor.set_track(track=track)

                    if not should_send_video:
                        # For DATA_ONLY, start data-only processing task
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
