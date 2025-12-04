import asyncio
import datetime
import fractions
import json
import logging
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
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
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av import logging as av_logging
from pydantic import ValidationError

from inference.core import logger
from inference.core.env import (
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
from inference.core.workflows.errors import WorkflowSyntaxError
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


def parse_chunked_binary_message(message: bytes) -> Tuple[int, int, int, bytes]:
    """Parse a binary message with standard 12-byte header.

    Returns: (frame_id, chunk_index, total_chunks, payload)
    """
    if len(message) < 12:
        raise ValueError(f"Message too short: {len(message)} bytes (expected >= 12)")

    frame_id, chunk_index, total_chunks = struct.unpack("<III", message[0:12])
    payload = message[12:]
    return frame_id, chunk_index, total_chunks, payload


class ChunkReassembler:
    """Helper to reassemble chunked binary messages."""

    def __init__(self):
        self._chunks: Dict[
            int, Dict[int, bytes]
        ] = {}  # {frame_id: {chunk_index: data}}
        self._total: Dict[int, int] = {}  # {frame_id: total_chunks}

    def add_chunk(
        self, frame_id: int, chunk_index: int, total_chunks: int, chunk_data: bytes
    ) -> Optional[bytes]:
        """Add a chunk and return complete payload if all chunks received.

        Returns:
            Complete reassembled payload bytes if all chunks received, None otherwise.
        """
        # Initialize buffers for new frame
        if frame_id not in self._chunks:
            self._chunks[frame_id] = {}
            self._total[frame_id] = total_chunks

        # Store chunk
        self._chunks[frame_id][chunk_index] = chunk_data

        # Check if all chunks received
        if len(self._chunks[frame_id]) >= total_chunks:
            # Reassemble in order
            complete_payload = b"".join(
                self._chunks[frame_id][i] for i in range(total_chunks)
            )

            # Clean up
            del self._chunks[frame_id]
            del self._total[frame_id]

            return complete_payload

        return None


class VideoFileUploadHandler:
    """Handles video file uploads via data channel.

    Protocol: [chunk_index:u32][total_chunks:u32][payload]
    Auto-completes when all chunks received.
    """

    def __init__(self):
        import tempfile

        self._chunks: Dict[int, bytes] = {}
        self._total_chunks: Optional[int] = None
        self._temp_file_path: Optional[str] = None
        self._state = VideoFileUploadState.IDLE
        self.upload_complete_event = asyncio.Event()

    @property
    def temp_file_path(self) -> Optional[str]:
        return self._temp_file_path

    def handle_chunk(
        self, chunk_index: int, total_chunks: int, data: bytes
    ) -> None:
        """Handle a chunk. Auto-completes when all chunks received."""
        if self._total_chunks is None:
            self._total_chunks = total_chunks
            self._state = VideoFileUploadState.UPLOADING
            logger.info(f"Starting video upload: {total_chunks} chunks")

        self._chunks[chunk_index] = data

        if chunk_index % 100 == 0:
            logger.info(f"Upload progress: {len(self._chunks)}/{total_chunks} chunks")

        # Auto-complete when all chunks received
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
            f"Video upload complete: {total_size} bytes -> {self._temp_file_path}"
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

    def cleanup(self) -> None:
        """Clean up temp file."""
        if self._temp_file_path:
            import os

            try:
                os.unlink(self._temp_file_path)
            except Exception:
                pass
            self._temp_file_path = None


def send_chunked_data(
    data_channel: RTCDataChannel,
    frame_id: int,
    payload_bytes: bytes,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Send payload via data channel, automatically chunking if needed.

    Args:
        data_channel: RTCDataChannel to send on
        frame_id: Frame identifier
        payload_bytes: Data to send (JPEG, JSON UTF-8, etc.)
        chunk_size: Maximum chunk size (default 48KB)
    """
    if data_channel.readyState != "open":
        logger.warning(f"Cannot send response for frame {frame_id}, channel not open")
        return

    total_chunks = (
        len(payload_bytes) + chunk_size - 1
    ) // chunk_size  # Ceiling division

    if frame_id % 100 == 1:
        logger.info(
            f"Sending response for frame {frame_id}: {total_chunks} chunk(s), {len(payload_bytes)} bytes"
        )

    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(payload_bytes))
        chunk_data = payload_bytes[start:end]

        message = create_chunked_binary_message(
            frame_id, chunk_index, total_chunks, chunk_data
        )
        data_channel.send(message)


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
        use_data_channel_frames: bool = False,
        heartbeat_callback: Optional[Callable[[], None]] = None,
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
        self.heartbeat_callback = heartbeat_callback
        self.use_data_channel_frames = use_data_channel_frames
        self._data_frame_queue: "asyncio.Queue[Optional[VideoFrame]]" = asyncio.Queue()
        self._chunk_reassembler = (
            ChunkReassembler()
        )  # For reassembling inbound frame chunks

        self.has_video_track = has_video_track
        self.stream_output = stream_output
        self.data_channel: Optional[RTCDataChannel] = None

        # Video file upload support
        self.video_upload_handler: Optional[VideoFileUploadHandler] = None
        self._video_file_mode = False

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

    def set_track(self, track: RemoteStreamTrack):
        if not self.track:
            self.track = track

    def close(self):
        self._track_active = False
        self._stop_processing = True
        # Clean up video upload handler if present
        if self.video_upload_handler is not None:
            self.video_upload_handler.cleanup()
        # Clean up video capture if present
        if hasattr(self, "_video_capture") and self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

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
            json_bytes = json.dumps(webrtc_output.model_dump()).encode("utf-8")
            send_chunked_data(self.data_channel, self._received_frames, json_bytes)
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

        # Set serialized outputs
        if serialized_outputs:
            webrtc_output.serialized_output_data = serialized_outputs

        # Send using binary chunked protocol
        json_bytes = json.dumps(webrtc_output.model_dump(mode="json")).encode("utf-8")
        send_chunked_data(self.data_channel, self._received_frames, json_bytes)

    async def _handle_data_channel_frame(self, message: bytes) -> None:
        """Handle incoming binary frame chunk from upstream_frames data channel.

        Uses standard binary protocol with 12-byte header + JPEG chunk payload.
        """
        try:
            # Parse message
            frame_id, chunk_index, total_chunks, jpeg_chunk = (
                parse_chunked_binary_message(message)
            )

            # Add chunk and check if complete
            jpeg_bytes = self._chunk_reassembler.add_chunk(
                frame_id, chunk_index, total_chunks, jpeg_chunk
            )

            if jpeg_bytes is None:
                # Still waiting for more chunks
                return

            # All chunks received - decode and queue frame
            if frame_id % 100 == 1:
                logger.info(
                    f"Received frame {frame_id}: {total_chunks} chunk(s), {len(jpeg_bytes)} bytes JPEG"
                )

            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if np_image is None:
                logger.error(f"Failed to decode JPEG for frame {frame_id}")
                return

            video_frame = VideoFrame.from_ndarray(np_image, format="bgr24")
            await self._data_frame_queue.put((frame_id, video_frame))

            if frame_id % 100 == 1:
                logger.info(f"Queued frame {frame_id}")

        except Exception as e:
            logger.error(f"Error handling frame chunk: {e}", exc_info=True)

    async def process_frames_data_only(self):
        """Process frames for data extraction only, without video track output.

        This is used when stream_output=[] and no video track is needed.
        """
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        logger.info(
            f"Starting data-only frame processing (use_data_channel_frames={self.use_data_channel_frames})"
        )

        try:
            while not self._stop_processing:
                if self._check_termination():
                    break
                if self.heartbeat_callback:
                    self.heartbeat_callback()

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

    async def process_video_file(self, video_path: str) -> None:
        """Process a video file through the workflow.

        Reads frames from the video file using cv2.VideoCapture and sends
        processed results via data channel.

        Args:
            video_path: Path to the video file to process
        """
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        logger.info(f"Starting video file processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video info: fps={fps}, total_frames={total_frames}")

            frame_id = 0
            while not self._stop_processing:
                if self._check_termination():
                    break

                if self.heartbeat_callback:
                    self.heartbeat_callback()

                ret, np_frame = cap.read()
                if not ret:
                    logger.info("Reached end of video file")
                    break

                frame_id += 1
                self._received_frames = frame_id
                frame_timestamp = datetime.datetime.now()

                # Convert numpy frame to VideoFrame
                video_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")

                workflow_output, _, errors = await self._process_frame_async(
                    frame=video_frame,
                    frame_id=frame_id,
                    render_output=False,
                    include_errors_on_frame=False,
                )

                # Send data via data channel
                await self._send_data_output(
                    workflow_output, frame_timestamp, video_frame, errors
                )

                if frame_id % 100 == 0:
                    logger.info(f"Processed frame {frame_id}/{total_frames}")

            logger.info(f"Video file processing complete: {frame_id} frames processed")

            # Send completion signal via data channel
            if self.data_channel and self.data_channel.readyState == "open":
                completion_message = WebRTCOutput(
                    serialized_output_data=None,
                    video_metadata=WebRTCVideoMetadata(
                        frame_id=frame_id,
                        received_at=datetime.datetime.now().isoformat(),
                    ),
                    errors=[],
                    processing_complete=True,
                )
                json_bytes = json.dumps(
                    completion_message.model_dump(mode="json")
                ).encode("utf-8")
                send_chunked_data(self.data_channel, frame_id + 1, json_bytes)
                logger.info("Sent processing_complete signal to client")

        except Exception as exc:
            logger.error(f"Error processing video file: {exc}", exc_info=True)
        finally:
            cap.release()

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
        use_data_channel_frames: bool = False,
        heartbeat_callback: Optional[Callable[[], None]] = None,
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
            use_data_channel_frames=use_data_channel_frames,
            model_manager=model_manager,
            heartbeat_callback=heartbeat_callback,
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

        # Video file upload mode: wait for upload, then read from file
        if self._video_file_mode:
            return await self._recv_from_video_file()

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

    async def _recv_from_video_file(self) -> VideoFrame:
        """Read and process frames from uploaded video file.

        Waits for upload completion, then reads frames sequentially.
        Returns processed frames for video track output.
        """
        # Wait for upload to complete (only on first call)
        if not hasattr(self, "_video_capture") or self._video_capture is None:
            if self.video_upload_handler is None:
                raise MediaStreamError("Video upload handler not initialized")

            logger.info("Waiting for video file upload to complete...")
            await self.video_upload_handler.upload_complete_event.wait()

            video_path = self.video_upload_handler.temp_file_path
            if not video_path:
                raise MediaStreamError("No video file path after upload complete")

            logger.info(f"Opening uploaded video file: {video_path}")
            self._video_capture = cv2.VideoCapture(video_path)
            if not self._video_capture.isOpened():
                raise MediaStreamError(f"Failed to open video file: {video_path}")

            self._video_fps = self._video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            self._video_total_frames = int(
                self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            self._video_pts = 0
            logger.info(
                f"Video opened: fps={self._video_fps}, total_frames={self._video_total_frames}"
            )

        # Read next frame
        ret, np_frame = self._video_capture.read()
        if not ret:
            logger.info("Reached end of video file")
            self._video_capture.release()
            self._video_capture = None
            # Clean up temp file
            if self.video_upload_handler:
                self.video_upload_handler.cleanup()
            raise MediaStreamError("End of video file")

        self._received_frames += 1
        frame_timestamp = datetime.datetime.now()

        # Convert numpy frame to VideoFrame
        frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
        frame.pts = self._video_pts
        frame.time_base = fractions.Fraction(1, int(self._video_fps))
        self._video_pts += 1

        # Auto-detect stream output on first frame
        if self.stream_output is None and self._received_frames == 1:
            await self._auto_detect_stream_output(frame, self._received_frames)

        # Process frame
        workflow_output, new_frame, errors = await self._process_frame_async(
            frame=frame,
            frame_id=self._received_frames,
            stream_output=self.stream_output,
            render_output=True,
            include_errors_on_frame=True,
        )

        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        # Send data output
        await self._send_data_output(workflow_output, frame_timestamp, frame, errors)

        if self._received_frames % 100 == 0:
            logger.info(
                f"Video file processing: {self._received_frames}/{self._video_total_frames}"
            )

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
                use_data_channel_frames=webrtc_request.use_data_channel_frames,
                heartbeat_callback=heartbeat_callback,
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
                use_data_channel_frames=webrtc_request.use_data_channel_frames,
                heartbeat_callback=heartbeat_callback,
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
        logger.info("on_connectionstatechange: %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            if video_processor.track:
                logger.info("Stopping video processor track")
                video_processor.track.stop()
            video_processor.close()
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
            terminate_event.set()

    @peer_connection.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info("Data channel '%s' received", channel.label)

        # Handle upstream frames channel (client sending frames to server)
        if channel.label == "upstream_frames":
            logger.info(
                "Upstream frames channel established, starting data-only processing"
            )

            @channel.on("message")
            def on_frame_message(message):
                asyncio.create_task(video_processor._handle_data_channel_frame(message))

            # Start processing immediately since we won't get a media track
            if webrtc_request.use_data_channel_frames and not should_send_video:
                asyncio.create_task(video_processor.process_frames_data_only())

            return

        # Handle video file upload channel
        if channel.label == "video_upload":
            logger.info("Video upload channel established")

            video_processor.video_upload_handler = VideoFileUploadHandler()

            if should_send_video:
                # Video track output: add track now, recv() will wait for upload
                video_processor._video_file_mode = True
                peer_connection.addTrack(video_processor)
            # else: data-only mode, will call process_video_file() when done

            @channel.on("message")
            def on_upload_message(message):
                # Keep watchdog alive during upload and keepalive pings
                if video_processor.heartbeat_callback:
                    video_processor.heartbeat_callback()

                # Ignore keepalive pings (1-byte messages)
                if len(message) <= 1:
                    return

                chunk_index, total_chunks, data = parse_video_file_chunk(message)
                video_processor.video_upload_handler.handle_chunk(
                    chunk_index, total_chunks, data
                )

                # For data-only: start processing when upload completes
                if not should_send_video:
                    video_path = video_processor.video_upload_handler.try_start_processing()
                    if video_path:
                        asyncio.create_task(video_processor.process_video_file(video_path))

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
    video_processor.close()
    await usage_collector.async_push_usage_payloads()
    logger.info("WebRTC peer connection closed")
