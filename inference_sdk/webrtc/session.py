"""WebRTC session management."""

import asyncio
import base64
import functools
import inspect
import json
import queue
import struct
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import requests

from inference_sdk.config import (
    WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT,
    WEBRTC_INITIAL_FRAME_TIMEOUT,
    WEBRTC_VIDEO_QUEUE_MAX_SIZE,
)
from inference_sdk.utils.logging import get_logger
from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.datachannel import ChunkReassembler
from inference_sdk.webrtc.sources import StreamSource, VideoFileSource

if TYPE_CHECKING:
    from aiortc import RTCDataChannel, RTCPeerConnection


def _check_webrtc_dependencies():
    """Check if WebRTC dependencies are installed and provide helpful error message."""
    try:
        import aiortc  # noqa: F401
        import av  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "WebRTC dependencies are not installed.\n"
            "Install them with: pip install inference-sdk[webrtc]\n"
            "Or if installing from source: pip install aiortc>=1.9.0"
        ) from e


logger = get_logger("webrtc.session")


def _decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image string to BGR numpy array.

    Args:
        base64_str: Base64-encoded image data (JPEG or PNG)

    Returns:
        BGR numpy array (H, W, 3) uint8
    """
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


class SessionState(Enum):
    """WebRTC session lifecycle states."""

    NOT_STARTED = "not_started"
    STARTED = "started"
    CLOSED = "closed"


@dataclass
class VideoMetadata:
    """Metadata about a video frame received from WebRTC stream.

    This metadata is attached to each frame processed by the server
    and can be used to track frame timing, synchronization, and
    processing information.

    Attributes:
        frame_id: Unique identifier for this frame in the stream
        received_at: Timestamp when the server received the frame
        pts: Presentation timestamp from the video stream (optional)
        time_base: Time base for interpreting pts values (optional)
        declared_fps: Declared/expected frames per second (optional)
        measured_fps: Measured actual frames per second (optional)
    """

    frame_id: int
    received_at: datetime
    pts: Optional[int] = None
    time_base: Optional[float] = None
    declared_fps: Optional[float] = None
    measured_fps: Optional[float] = None


class _VideoStream:
    """Wrapper for video frame queue providing iterator interface."""

    def __init__(
        self,
        session: "WebRTCSession",
        frames: "Queue[Optional[tuple[np.ndarray, VideoMetadata]]]",
        initial_frame_timeout: float = WEBRTC_INITIAL_FRAME_TIMEOUT,
    ):
        self._session = session
        self._frames = frames
        self._initial_frame_timeout = initial_frame_timeout
        self._first_frame_received = False

    def __call__(self) -> Iterator[Tuple[np.ndarray, VideoMetadata]]:
        """Iterate over video frames with metadata.

        Automatically starts the session if not already started.
        Yields tuples of (BGR numpy array, VideoMetadata) until the stream ends (None received)
        or session is closed.
        The metadata is extracted directly from the video frame (pts, time_base, etc.).

        Raises:
            TimeoutError: If first frame not received within timeout period
        """
        self._session._ensure_started()
        while True:
            # Check if session was closed (e.g., from a handler)
            if self._session._state == SessionState.CLOSED:
                break

            # Use timeout only for first frame to detect server not sending
            timeout = (
                self._initial_frame_timeout if not self._first_frame_received else None
            )

            try:
                frame_data = self._frames.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"No video frames received within {self._initial_frame_timeout}s timeout.\n"
                    "This likely means the server is not sending video.\n"
                    "Troubleshooting:\n"
                    "  - Check that stream_output is configured in your StreamConfig\n"
                    "  - Verify the workflow outputs match your configuration\n"
                    "  - Ensure the server has WebRTC enabled and is processing frames"
                )

            if frame_data is None:
                break

            self._first_frame_received = True
            yield frame_data


class WebRTCSession:
    """WebRTC session for streaming video and receiving inference results.

    This class manages the WebRTC peer connection, video streaming,
    and data channel communication with the inference server.

    The session automatically starts on first use (e.g., calling run() or video()).
    Call close() to cleanup resources, or rely on __del__ for automatic cleanup.

    Example:
        session = client.webrtc.stream(source=source, workflow=workflow)

        @session.on_frame
        def process_frame(frame, metadata):
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                session.close()

        session.run()  # Auto-starts, auto-closes on exception
    """

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str],
        source: StreamSource,
        image_input_name: str,
        workflow_config: dict,
        stream_config: StreamConfig,
    ) -> None:
        """Initialize WebRTC session.

        Args:
            api_url: Inference server API URL
            api_key: API key for authentication
            source: Stream source instance
            image_input_name: Name of image input in workflow
            workflow_config: Workflow configuration dict
            stream_config: Stream configuration
        """

        self._state: SessionState = SessionState.NOT_STARTED
        self._state_lock: threading.Lock = threading.Lock()

        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._source = source
        self._image_input_name = image_input_name
        self._workflow_config = workflow_config
        self._config = stream_config

        # Internal state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pc: Optional["RTCPeerConnection"] = None
        self._video_queue: "Queue[Optional[tuple[np.ndarray, VideoMetadata]]]" = Queue(
            maxsize=WEBRTC_VIDEO_QUEUE_MAX_SIZE
        )
        self._video_through_datachannel = False

        # Callback handlers
        self._frame_handlers: List[Callable] = []
        self._data_field_handlers: Dict[str, List[Callable]] = {}
        self._data_global_handler: Optional[Callable] = None

        # Chunk reassembly for binary messages
        self._chunk_reassembler = ChunkReassembler()

        # Public APIs
        self.video = _VideoStream(self, self._video_queue)

    def _init_connection(self) -> None:
        """Initialize event loop, thread, and WebRTC connection."""
        # Start event loop in background thread
        self._loop = asyncio.new_event_loop()

        def _run(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run, args=(self._loop,), daemon=True
        )
        self._loop_thread.start()

        # Initialize WebRTC connection
        fut = asyncio.run_coroutine_threadsafe(self._init(), self._loop)
        try:
            fut.result()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"WebRTC endpoint not found at {self._api_url}/initialise_webrtc_worker.\n"
                    f"This API URL may not support WebRTC streaming.\n"
                    f"Troubleshooting:\n"
                    f"  - For self-hosted inference, ensure the server is started with WebRTC enabled\n"
                    f"  - For Roboflow Cloud, use a dedicated inference server URL (not serverless.roboflow.com)\n"
                    f"  - Verify the --api-url parameter points to the correct server"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to initialize WebRTC session (HTTP {e.response.status_code}).\n"
                    f"API URL: {self._api_url}\n"
                    f"Error: {e}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize WebRTC session: {e.__class__.__name__}: {e}\n"
                f"API URL: {self._api_url}"
            ) from e

    def _ensure_started(self) -> None:
        """Ensure connection is started (thread-safe, idempotent)."""
        with self._state_lock:
            if self._state == SessionState.NOT_STARTED:
                self._state = SessionState.STARTED
                self._init_connection()
            elif self._state == SessionState.CLOSED:
                raise RuntimeError("Cannot use closed WebRTCSession")

    def _parse_video_metadata(
        self, video_metadata_dict: Optional[dict]
    ) -> Optional[VideoMetadata]:
        """Parse video metadata from message dict.

        Args:
            video_metadata_dict: Dictionary containing video metadata fields

        Returns:
            VideoMetadata instance or None if parsing fails or dict is None
        """
        if not video_metadata_dict:
            return None

        try:
            return VideoMetadata(
                frame_id=video_metadata_dict["frame_id"],
                received_at=datetime.fromisoformat(video_metadata_dict["received_at"]),
                pts=video_metadata_dict.get("pts"),
                time_base=video_metadata_dict.get("time_base"),
                declared_fps=video_metadata_dict.get("declared_fps"),
                measured_fps=video_metadata_dict.get("measured_fps"),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse video_metadata: {e}")
            return None

    def close(self) -> None:
        """Close session and cleanup all resources. Idempotent - safe to call multiple times.

        This method closes the WebRTC peer connection, releases source resources
        (webcam, video files, etc.), stops the event loop, and joins the background thread.

        It's safe to call this multiple times - subsequent calls are no-ops.

        Example:
            session = client.webrtc.stream(source=source, workflow=workflow)
            session.run()  # Auto-starts and auto-closes on exception
            session.close()  # Explicit cleanup (or let __del__ handle it)
        """
        with self._state_lock:
            if self._state == SessionState.CLOSED:
                return  # Already closed, nothing to do
            self._state = SessionState.CLOSED

        # Signal video iterator to stop by putting None sentinel
        try:
            self._video_queue.put_nowait(None)
        except Exception:
            pass  # Queue might be full, but that's okay

        # Cleanup resources (nested finally ensures all cleanup steps execute)
        try:
            # Close peer connection
            if self._loop and self._pc:
                asyncio.run_coroutine_threadsafe(self._pc.close(), self._loop).result()
        finally:
            try:
                # Cleanup source (webcam, video file, etc.)
                if self._loop and self._source:
                    asyncio.run_coroutine_threadsafe(
                        self._source.cleanup(), self._loop
                    ).result()
            finally:
                # Stop event loop and join thread
                if self._loop:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread:
                    self._loop_thread.join(timeout=WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT)

    def __enter__(self) -> "WebRTCSession":
        """Enter context manager - returns self.

        Returns:
            WebRTCSession: The session instance for use in with statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - automatically closes the session.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise.
            exc_val: Exception value if an exception occurred, None otherwise.
            exc_tb: Exception traceback if an exception occurred, None otherwise.
        """
        self.close()

    def __del__(self) -> None:
        """Cleanup if user forgot to close. Not guaranteed to run immediately."""
        try:
            if self._state == SessionState.STARTED:
                logger.warning(
                    "WebRTCSession was not properly closed. "
                    "Consider calling session.close() explicitly for immediate cleanup."
                )
                self.close()
        except Exception:
            pass  # Never raise from __del__

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for session to complete.

        Blocks until the video stream ends (None received) or timeout expires.
        Automatically starts the session if not already started.

        Args:
            timeout: Maximum time to wait in seconds (None for indefinite)

        Raises:
            TimeoutError: If timeout expires before stream ends
        """
        self._ensure_started()
        try:
            while True:
                frame_data = self._video_queue.get(timeout=timeout)
                if frame_data is None:
                    break
        except queue.Empty:
            if timeout is not None:
                raise TimeoutError(
                    f"WebRTC session wait() timed out after {timeout}s.\n"
                    "The video stream did not end within the timeout period."
                )

    def on_frame(self, callback: Callable) -> Callable:
        """Decorator to register frame callback handlers.

        The registered handlers will be called for each video frame received
        when using the run() method. Handlers must accept two parameters:
        - frame: BGR numpy array (np.ndarray)
        - metadata: Video metadata (VideoMetadata) extracted from the video frame

        Args:
            callback: Callback function that accepts (frame, metadata)

        Returns:
            The callback itself

        Examples:
            @session.on_frame
            def process_frame(frame: np.ndarray, metadata: VideoMetadata):
                print(f"Frame {metadata.frame_id} - PTS: {metadata.pts}")
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    session.stop()
        """
        self._frame_handlers.append(callback)
        return callback

    def on_data(self, field_name: Optional[str] = None) -> Callable:
        """Decorator to register data channel callback handlers.

        Can be used with or without parentheses:
            @session.on_data          # without parentheses (global handler)
            @session.on_data()        # with parentheses (global handler)
            @session.on_data("field") # with field name (field-specific handler)

        Args:
            field_name: If provided, handler receives only that field's value.
                       If None, handler receives entire serialized_output_data dict.

        Returns:
            Decorator function or decorated function

        Examples:
            # Global handler without parentheses
            @session.on_data
            def handle_all(data: dict, metadata: VideoMetadata):
                print(f"All data: {data}")

            # Field-specific handler
            @session.on_data("predictions")
            def handle_predictions(data: dict, metadata: VideoMetadata):
                print(f"Frame {metadata.frame_id}: {data}")

            # Field-specific handler (no metadata)
            @session.on_data("predictions")
            def handle_predictions(data: dict):
                print(data)

            # Global handler with parentheses
            @session.on_data()
            def handle_all(data: dict, metadata: VideoMetadata):
                print(f"All data: {data}")
        """
        # Check if being used without parentheses: @session.on_data
        # In this case, field_name is actually the function being decorated
        if callable(field_name):
            fn = field_name
            self._data_global_handler = fn
            return fn

        # Being used with parentheses: @session.on_data() or @session.on_data("field")
        def decorator(fn: Callable) -> Callable:
            if field_name is None:
                self._data_global_handler = fn
            else:
                if field_name not in self._data_field_handlers:
                    self._data_field_handlers[field_name] = []
                self._data_field_handlers[field_name].append(fn)
            return fn

        return decorator

    def run(self) -> None:
        """Block and process frames until close() is called or stream ends.

        This method iterates over incoming video frames and invokes all
        registered frame handlers for each frame. Automatically starts
        the session if not already started.

        The session automatically closes when this method exits, whether
        normally or due to an exception, ensuring resources are always
        cleaned up.

        Blocks until either:
        - close() is called (e.g., from a callback)
        - The video stream ends naturally
        - An exception occurs (session auto-closes, exception re-raised)
        - KeyboardInterrupt (Ctrl+C) is received (session auto-closes)

        Data channel handlers are invoked automatically when data arrives,
        independent of this method.

        Example:
            session = client.webrtc.stream(source=source, workflow=workflow)

            @session.on_frame
            def process(frame, metadata):
                print(f"Frame {metadata.frame_id} - PTS: {metadata.pts}")
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    session.close()  # Exits run() and cleans up

            session.run()  # Auto-starts, auto-closes, blocks here
        """
        with self:
            for frame, metadata in self.video():
                # Invoke all registered frame handlers with both parameters
                for handler in self._frame_handlers:
                    try:
                        handler(frame, metadata)
                    except Exception:
                        logger.warning("Error in frame handler", exc_info=True)

    @staticmethod
    @functools.lru_cache(maxsize=100)
    def _data_handler_length(handler: Callable) -> int:
        """Get the number of parameters expected by a data handler.

        Args:
            handler: The handler callable to inspect

        Returns:
            The number of parameters expected by the handler
        """
        sig = inspect.signature(handler)
        return len(sig.parameters)

    def _invoke_data_handler(
        self, handler: Callable, value: Any, metadata: Optional[VideoMetadata]
    ) -> None:  # noqa: ANN401
        """Invoke data handler with appropriate signature (auto-detect via introspection).

        Supports two signatures:
        - handler(value, metadata) - receives both value and metadata
        - handler(value) - receives only value

        Args:
            handler: The handler callable to invoke
            value: The data value to pass
            metadata: Optional video metadata to pass
        """
        try:
            if WebRTCSession._data_handler_length(handler) >= 2:
                # Handler expects both value and metadata
                handler(value, metadata)
            else:
                # Handler expects only value
                handler(value)
        except Exception:
            logger.exception(
                f"Failed to invoke handler {handler}. The handler should have 2 parameters with signature: handler(value, metadata) or handler(value)."
            )
            raise

    async def _get_turn_config(self) -> Optional[dict]:
        """Get TURN configuration from user-provided config.

        Priority order:
        1. User-provided config via StreamConfig.turn_server (highest priority)
        2. Skip TURN for localhost connections
        3. Return None if not provided

        Returns:
            TURN configuration dict or None
        """
        # 1. Use user-provided config if available
        if self._config.turn_server:
            logger.debug("Using user-provided TURN configuration")
            return self._config.turn_server

        # 3. No TURN config provided
        logger.debug("No TURN configuration provided, proceeding without TURN server")
        return None

    def _handle_datachannel_video_frame(
        self, serialized_data: Any, metadata: Optional[VideoMetadata]
    ) -> None:
        """Handle video frame received through data channel.

        Args:
            serialized_data: The serialized output data containing base64 image
            metadata: Video metadata for the frame
        """
        for output_name in self._config.stream_output:
            if not output_name or output_name not in serialized_data:
                continue
            img_data = serialized_data[output_name]
            if isinstance(img_data, dict) and img_data.get("type") == "base64":
                try:
                    # Decode base64 image and queue it
                    frame = _decode_base64_image(img_data["value"])
                    # Backpressure: drop oldest frame if queue full
                    if self._video_queue.full():
                        try:
                            self._video_queue.get_nowait()
                        except Exception:
                            pass
                    self._video_queue.put_nowait((frame, metadata))
                except Exception:
                    logger.warning(
                        f"Failed to decode base64 image from {output_name}",
                        exc_info=True,
                    )
                break  # Only process first matching image

    async def _init(self) -> None:
        """Initialize WebRTC connection.

        Sets up peer connection, configures source, negotiates with server.
        """
        # Check dependencies and import them
        _check_webrtc_dependencies()
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
        from aiortc.contrib.media import MediaRelay
        from av import VideoFrame

        # Fetch TURN configuration (auto-fetch or user-provided)
        turn_config = await self._get_turn_config()

        # Create peer connection with TURN config if available
        configuration = None
        if turn_config:
            ice = RTCIceServer(
                urls=[turn_config.get("urls")],
                username=turn_config.get("username"),
                credential=turn_config.get("credential"),
            )
            configuration = RTCConfiguration(iceServers=[ice])

        pc = RTCPeerConnection(configuration=configuration)
        relay = MediaRelay()

        # Setup video receiver for frames from server
        @pc.on("track")
        def _on_track(track):  # noqa: ANN001
            subscribed = relay.subscribe(track)

            async def _reader():
                from aiortc.mediastreams import MediaStreamError

                while True:
                    try:
                        f: VideoFrame = await subscribed.recv()
                    except MediaStreamError:
                        # Remote stream finished normally
                        logger.info("Remote stream finished")
                        try:
                            self._video_queue.put_nowait(None)
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        # Connection closed or track ended unexpectedly
                        logger.error(
                            f"WebRTC video track ended: {e.__class__.__name__}: {e}",
                            exc_info=True,
                        )
                        try:
                            self._video_queue.put_nowait(None)
                        except Exception:
                            pass
                        break
                    img = f.to_ndarray(format="bgr24")
                    current_metadata = VideoMetadata(
                        frame_id=f.pts,
                        received_at=datetime.now(),
                        pts=f.pts,
                        time_base=f.time_base,
                        declared_fps=None,
                        measured_fps=None,
                    )
                    # Backpressure: drop oldest frame if queue full
                    if self._video_queue.full():
                        try:
                            _ = self._video_queue.get_nowait()
                        except Exception:
                            pass
                    try:
                        self._video_queue.put_nowait((img, current_metadata))
                    except Exception:
                        pass

            asyncio.ensure_future(_reader())

        # Setup data channel
        ch = pc.createDataChannel("inference")

        # Setup data channel message handler
        @ch.on("message")
        def _on_data_message(message: Any) -> None:  # noqa: ANN401
            try:
                # Handle both bytes and str messages
                if isinstance(message, bytes):
                    # Check if it's a chunked binary message
                    if len(message) >= 12:
                        try:
                            # Try to reassemble chunks
                            complete_payload, _ = self._chunk_reassembler.add_chunk(
                                message
                            )
                            if complete_payload is None:
                                # Not all chunks received yet
                                return
                            # Parse the complete JSON from reassembled payload
                            message = complete_payload.decode("utf-8")
                        except (struct.error, ValueError):
                            # Not a chunked message, try to decode as regular UTF-8
                            message = message.decode("utf-8")
                    else:
                        # Too short to be chunked, decode as regular UTF-8
                        message = message.decode("utf-8")

                parsed_message = json.loads(message)

                # Handle processing_complete signal (video file finished)
                if parsed_message.get("processing_complete"):
                    logger.info("Received processing_complete signal")
                    try:
                        self._video_queue.put_nowait(None)
                    except Exception:
                        pass
                    return

                # Extract video metadata if present (for data handlers)
                metadata = self._parse_video_metadata(
                    parsed_message.get("video_metadata")
                )

                # Get serialized output data
                serialized_data = parsed_message.get("serialized_output_data")

                # Check for base64 image in stream_output fields (for VideoFileSource)
                # This enables receiving frames via data channel instead of video track
                if serialized_data and self._video_through_datachannel:
                    self._handle_datachannel_video_frame(serialized_data, metadata)

                # Call global handler if registered
                if self._data_global_handler:
                    try:
                        # filter out video frames if video is sent through datachannel
                        filtered_data = serialized_data
                        if self._video_through_datachannel:
                            filtered_data = {
                                k: v
                                for k, v in serialized_data.items()
                                if k not in self._config.stream_output
                            }
                        self._invoke_data_handler(
                            self._data_global_handler, filtered_data, metadata
                        )
                    except Exception:
                        logger.warning(
                            "Error calling global data handler", exc_info=True
                        )

                # Route to field-specific handlers
                if isinstance(serialized_data, dict):
                    for field_name, field_value in serialized_data.items():
                        if field_name in self._data_field_handlers:
                            for handler in list(self._data_field_handlers[field_name]):
                                try:
                                    self._invoke_data_handler(
                                        handler, field_value, metadata
                                    )
                                except Exception:
                                    logger.warning(
                                        f"Error calling handler for field '{field_name}'",
                                        exc_info=True,
                                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse data channel message as JSON")

        # Let source configure the peer connection
        # (adds tracks for webcam/video/manual, or recvonly transceiver for RTSP)
        await self._source.configure_peer_connection(pc)

        # Create offer and wait for ICE gathering
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # Wait for ICE gathering to complete
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        # Build server initialization payload
        wf_conf: Dict[str, Any] = {
            "type": "WorkflowConfiguration",
            "image_input_name": self._image_input_name,
            "workflows_parameters": self._config.workflow_parameters,
        }
        wf_conf.update(self._workflow_config)

        payload = {
            "api_key": self._api_key,
            "workflow_configuration": wf_conf,
            "webrtc_offer": {
                "type": pc.localDescription.type,
                "sdp": pc.localDescription.sdp,
            },
            "webrtc_realtime_processing": self._config.realtime_processing,
            "stream_output": self._config.stream_output,
            "data_output": self._config.data_output,
        }

        # Add TURN config if available (auto-fetched or user-provided)
        if turn_config:
            payload["webrtc_turn_config"] = turn_config

        # Add FPS if provided
        if self._config.declared_fps:
            payload["declared_fps"] = self._config.declared_fps

        # Merge source-specific parameters
        # (rtsp_url for RTSP, declared_fps for webcam, stream_output/data_output overrides for VideoFile)
        payload.update(self._source.get_initialization_params(self._config))
        # Check if video is will be sent through datachannel instead of video track
        self._video_through_datachannel = bool(
            self._config.stream_output and not payload.get("stream_output")
        )

        # Call server to initialize worker
        url = f"{self._api_url}/initialise_webrtc_worker"
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        ans: Dict[str, Any] = resp.json()

        # Set remote description
        answer = RTCSessionDescription(sdp=ans["sdp"], type=ans["type"])
        await pc.setRemoteDescription(answer)

        # Start video file upload if applicable
        if isinstance(self._source, VideoFileSource):
            asyncio.ensure_future(self._source.start_upload())

        self._pc = pc
