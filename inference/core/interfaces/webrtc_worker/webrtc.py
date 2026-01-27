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
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
    serialize_timestamp,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.errors import WorkflowError, WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.usage_tracking.collector import usage_collector

logging.getLogger("aiortc").setLevel(logging.WARNING)

# WebRTC data channel chunking configuration
CHUNK_SIZE = 48 * 1024  # 48KB - safe for all WebRTC implementations

# WebRTC image compression quality - lower = smaller file size
# quality=10 reduces ~1MB raw to ~50KB, quality=50 produces ~150-200KB
WEBRTC_JPEG_QUALITY = 70

# Keepalive frame interval in seconds - send black frame to keep video track open
WEBRTC_KEEPALIVE_INTERVAL = 1.0

# Default keepalive frame dimensions
KEEPALIVE_FRAME_WIDTH = 640
KEEPALIVE_FRAME_HEIGHT = 480


def create_keepalive_frame(
    width: int = KEEPALIVE_FRAME_WIDTH,
    height: int = KEEPALIVE_FRAME_HEIGHT,
    pts: int = 0,
    time_base: fractions.Fraction = fractions.Fraction(1, 30),
) -> VideoFrame:
    """Create a black keepalive frame to keep the video track open.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        pts: Presentation timestamp
        time_base: Time base for the frame
        
    Returns:
        A black VideoFrame
    """
    import numpy as np
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame = VideoFrame.from_ndarray(black_frame, format="bgr24")
    frame.pts = pts
    frame.time_base = time_base
    return frame


def serialise_image_for_webrtc(image: WorkflowImageData) -> Dict[str, Any]:
    """Serialize image with low JPEG quality for efficient WebRTC transmission."""
    jpeg_bytes = encode_image_to_jpeg_bytes(image.numpy_image, jpeg_quality=WEBRTC_JPEG_QUALITY)
    return {
        "type": "base64",
        "value": base64.b64encode(jpeg_bytes).decode("ascii"),
        "video_metadata": image.video_metadata.dict() if image.video_metadata else None,
    }


def _recompress_base64_image(base64_str: str) -> str:
    """Decode base64 image and re-encode with low JPEG quality.
    
    Handles images that were pre-serialized with high quality (95) and
    re-compresses them to ~50KB using quality=10 for efficient WebRTC transmission.
    """
    try:
        import cv2
        import numpy as np
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)
        # Decode image bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return base64_str  # Not a valid image, return as-is
        # Re-encode with low quality
        jpeg_bytes = encode_image_to_jpeg_bytes(img, jpeg_quality=WEBRTC_JPEG_QUALITY)
        return base64.b64encode(jpeg_bytes).decode("ascii")
    except Exception:
        # If anything fails, return original
        return base64_str


def serialize_for_webrtc(value: Any) -> Any:
    """Recursively serialize, compressing images with low JPEG quality.
    
    Handles:
    - WorkflowImageData objects
    - Pre-serialized image dicts ({"type": "base64", "value": "..."}) with high-quality images
    - Raw numpy arrays (images)
    """
    import numpy as np
    
    if isinstance(value, WorkflowImageData):
        return serialise_image_for_webrtc(value)
    if isinstance(value, np.ndarray):
        # Raw image array - compress with low quality
        if len(value.shape) >= 2:  # Looks like an image
            jpeg_bytes = encode_image_to_jpeg_bytes(value, jpeg_quality=WEBRTC_JPEG_QUALITY)
            return {
                "type": "base64",
                "value": base64.b64encode(jpeg_bytes).decode("ascii"),
            }
        return value.tolist()  # Not an image, convert to list
    if isinstance(value, dict):
        # Check if this is a pre-serialized image dict with base64 data
        # These come from workflow blocks that already serialized their output
        if value.get("type") == "base64" and isinstance(value.get("value"), str):
            # Re-compress the base64 image with low quality for WebRTC
            recompressed = _recompress_base64_image(value["value"])
            return {**value, "value": recompressed}
        return {k: serialize_for_webrtc(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_for_webrtc(v) for v in value]
    if isinstance(value, sv.Detections):
        return serialise_sv_detections(value)
    if isinstance(value, datetime.datetime):
        return serialize_timestamp(value)
    return value


def create_chunked_binary_message(
    frame_id: int, chunk_index: int, total_chunks: int, payload: bytes
) -> bytes:
    """Create a binary message with standard 12-byte header.

    Format: [frame_id: 4][chunk_index: 4][total_chunks: 4][payload: N]
    All integers are uint32 little-endian.
    """
    header = struct.pack("<III", frame_id, chunk_index, total_chunks)
    return header + payload


def _decode_worker(filepath: str, frame_queue, stop_event):
    """Decode video frames in a thread and put them on the queue."""

    try:
        container = av.open(filepath)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            if stop_event.is_set():
                break
            frame_queue.put(
                frame,
                timeout=30,
            )

        container.close()
    except Exception as e:
        frame_queue.put({"error": str(e)})
    finally:
        frame_queue.put(None)


class ThreadedVideoTrack(MediaStreamTrack):
    """Video track that decodes frames from a queue."""

    kind = "video"

    def __init__(self, filepath: str, queue_size: int = 10):
        super().__init__()
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._decode_thread = threading.Thread(
            target=_decode_worker,
            args=(filepath, self._queue, self._stop_event),
            daemon=True,
        )
        self._decode_thread.start()

    async def recv(self) -> VideoFrame:
        while True:
            try:
                data = self._queue.get_nowait()
                break
            except queue.Empty:
                # we use a non-blocking get + sleep to avoid blocking the
                # event loop.
                # The queue is typically pre-filled by the decoder thread,
                # so this sleep rarely triggers during normal operation.
                await asyncio.sleep(0.01)

        if data is None:
            self.stop()
            raise MediaStreamError("End of video file")
        if isinstance(data, dict):
            self.stop()
            raise MediaStreamError(data.get("error", "Unknown decode error"))

        return data

    def stop(self):
        super().stop()
        self._stop_event.set()


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


async def wait_for_buffer_drain(
    data_channel: RTCDataChannel,
    timeout: float = 30.0,
    heartbeat_callback: Optional[Callable[[], None]] = None,
    low_threshold: Optional[int] = None,
) -> bool:
    """Wait for data channel buffer to drain below threshold, with timeout.
    
    Uses a low threshold (default: high_limit / 4) to implement proper backpressure.
    We wait until buffer drops significantly before resuming to prevent oscillation.
    
    The asyncio.sleep() calls while waiting yield to the event loop, which is
    critical for ICE health - aioice needs event loop time to send STUN packets.
    
    Args:
        data_channel: The RTCDataChannel to monitor
        timeout: Maximum time to wait in seconds
        heartbeat_callback: Optional callback for server watchdog (e.g., Modal timeout)
        low_threshold: Buffer level to wait for (default: BUFFER_SIZE_LIMIT / 4)
        
    Returns:
        True if buffer drained, False if timeout or channel closed
    """
    if low_threshold is None:
        low_threshold = WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT // 4
    
    start_time = asyncio.get_event_loop().time()
    while data_channel.bufferedAmount > low_threshold:
        if asyncio.get_event_loop().time() - start_time > timeout:
            logger.warning(
                "Buffer drain timeout after %.1fs, buffered: %d bytes",
                timeout, data_channel.bufferedAmount
            )
            return False
        if data_channel.readyState != "open":
            logger.warning("Channel closed while waiting for buffer drain")
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
    buffer_timeout: float = 30.0,
) -> bool:
    """Send payload via data channel with chunking and backpressure.
    
    Uses proper backpressure with high/low watermarks to prevent buffer overflow.
    
    CRITICAL: Yields to event loop every N chunks via asyncio.sleep(0).
    This allows aioice to send STUN Binding Indications to refresh ICE consent.
    Without yielding, consent expires after ~30s causing "Consent to send expired".
    
    Args:
        data_channel: The RTCDataChannel to send on
        frame_id: Frame identifier for chunked message headers
        payload_bytes: Full payload to send
        chunk_size: Size of each chunk (default 48KB)
        heartbeat_callback: Optional callback for server watchdog (e.g., Modal timeout)
        buffer_timeout: Max time to wait for buffer drain
        
    Returns:
        True if all chunks sent, False if channel closed or timeout
    """
    if data_channel.readyState != "open":
        logger.warning("Cannot send frame %s: channel not open", frame_id)
        return False

    total_chunks = (len(payload_bytes) + chunk_size - 1) // chunk_size
    view = memoryview(payload_bytes)
    high_threshold = WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT
    
    for chunk_index in range(total_chunks):
        if data_channel.readyState != "open":
            logger.warning("Channel closed while sending frame %s", frame_id)
            return False

        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(payload_bytes))
        chunk_data = view[start:end]

        message = create_chunked_binary_message(
            frame_id, chunk_index, total_chunks, chunk_data
        )
        
        # Check buffer before sending - wait if too full
        if data_channel.bufferedAmount > high_threshold:
            if not await wait_for_buffer_drain(
                data_channel, buffer_timeout, heartbeat_callback
            ):
                logger.warning("Buffer drain failed for frame %s", frame_id)
                return False
        
        data_channel.send(message)

        # CRITICAL: Yield to event loop every 10 chunks
        # Without this, aioice cannot send STUN Binding Indications to refresh
        # ICE consent (expires after ~30s). Event loop starvation during large
        # transfers is the root cause of "Consent to send expired" errors.
        if chunk_index % 10 == 0:
            if heartbeat_callback:
                heartbeat_callback()  # Keep server watchdog alive
            await asyncio.sleep(0)
    
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
        """Block frame production when too far ahead of client ACKs.

        Allows up to (_ack_window) frames in flight beyond the last ACK.
        Only active for non-realtime processing (video file uploads).

        Has a maximum wait time of 30 seconds to prevent infinite blocking
        if the client stops sending ACKs.
        """
        if self.realtime_processing:
            return
        if self._ack_last == 0:
            return

        wait_counter = 0
        max_wait_iterations = 150  # this is...  150 * 0.2s, 30 seconds max wait
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
                if wait_counter >= max_wait_iterations:
                    logger.warning(
                        "ACK wait timeout exceeded (30s). Disabling ACK pacing. "
                        "(next_frame_id=%s, ack_last=%s)",
                        next_frame_id,
                        self._ack_last,
                    )

                    self._ack_last = 0
                    return

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
        """Serialize workflow outputs in a thread to avoid blocking the event loop.
        
        Uses low JPEG quality (quality=10) for image compression to reduce
        frame size from ~1MB to ~50KB for efficient WebRTC transmission.
        """
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
                serialized_value = serialize_for_webrtc(output_data)
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

        logger.info("Starting data-only frame processing")

        try:
            while not self._stop_processing:
                await self._wait_for_ack_window(next_frame_id=self._received_frames + 1)
                if self._check_termination():
                    break
                if self.heartbeat_callback:
                    self.heartbeat_callback()

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
        # Keepalive frame state
        self._keepalive_pts = 0
        self._keepalive_time_base = fractions.Fraction(1, int(declared_fps))

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

    def _create_keepalive(self) -> VideoFrame:
        """Create and return a keepalive frame, incrementing the pts counter."""
        keepalive = create_keepalive_frame(
            pts=self._keepalive_pts,
            time_base=self._keepalive_time_base,
        )
        self._keepalive_pts += 1
        return keepalive

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
        # Send keepalive frames while waiting to keep the video track open
        if self.track is None:
            try:
                await asyncio.wait_for(
                    self._track_ready_event.wait(),
                    timeout=WEBRTC_KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                # Track not ready yet, send keepalive frame to keep connection alive
                logger.debug("Sending keepalive frame while waiting for track")
                return self._create_keepalive()
            
            if self.track is None:
                raise MediaStreamError("Track not available after wait")

        # Optional ACK pacing: block producing the next frame if we're too far ahead.
        # Send keepalive frames while waiting for ACKs
        ack_wait_result = await self._wait_for_ack_window_with_keepalive(
            next_frame_id=self._received_frames + 1
        )
        if ack_wait_result == "keepalive":
            return self._create_keepalive()

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

    async def _wait_for_ack_window_with_keepalive(self, next_frame_id: int) -> str:
        """Wait for ACK window with keepalive frame support.
        
        Returns:
            "ready" if ready to produce next frame
            "keepalive" if should send keepalive frame instead
        """
        if self.realtime_processing:
            return "ready"
        if self._ack_last == 0:
            return "ready"

        # Check if we're within the ACK window
        if next_frame_id <= (self._ack_last + self._ack_window):
            return "ready"

        # Need to wait for ACK - use timeout to send keepalive frames
        try:
            self._ack_event.clear()
            await asyncio.wait_for(
                self._ack_event.wait(),
                timeout=WEBRTC_KEEPALIVE_INTERVAL
            )
            # ACK received, check again
            if next_frame_id <= (self._ack_last + self._ack_window):
                return "ready"
            # Still outside window, send keepalive
            return "keepalive"
        except asyncio.TimeoutError:
            # Timeout waiting for ACK, send keepalive to keep connection alive
            logger.debug("Sending keepalive frame while waiting for ACK")
            if self.heartbeat_callback:
                self.heartbeat_callback()
            return "keepalive"


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

    # Monitor ICE connection state - consent expires after ~30s without STUN refresh
    @peer_connection.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        state = peer_connection.iceConnectionState
        logger.info("ICE connection state changed: %s", state)
        
        if state == "failed":
            logger.warning(
                "ICE connection failed - likely consent expiry. "
                "This happens when event loop is blocked and aioice cannot "
                "send STUN Binding Indications. Check for tight loops without "
                "asyncio.sleep(0) yields."
            )
            # The connectionstatechange handler will clean up
        elif state == "disconnected":
            logger.warning(
                "ICE connection disconnected - may recover. "
                "If this persists, check network connectivity."
            )

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
                        # Throttled playback - use MediaPlayer
                        player = MediaPlayer(video_path, loop=False)
                        player._throttle_playback = True
                        video_processor.set_track(
                            track=player.video, rotation_code=rotation_code
                        )
                    else:
                        # Fast processing - use ThreadedVideoTrack
                        track = ThreadedVideoTrack(video_path)
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
