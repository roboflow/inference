"""WebRTC session management."""

import asyncio
import inspect
import json
import logging
import queue
import sys
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from types import TracebackType
from typing import Any, Callable, Iterator, List, Optional, Type

import numpy as np
import requests
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from inference_sdk.webrtc.config import StreamConfig
from inference_sdk.webrtc.sources import StreamSource

logger = logging.getLogger(__name__)


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


# Configuration constants
DEFAULT_INITIAL_FRAME_TIMEOUT = 30.0  # seconds to wait for first video frame
VIDEO_QUEUE_MAX_SIZE = 8  # maximum number of frames to buffer
EVENT_LOOP_SHUTDOWN_TIMEOUT = 2.0  # seconds to wait for event loop thread to stop

# Configure basic logging if root logger has no handlers
# This ensures users see important errors even without explicit logging setup
if not logging.root.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class _VideoStream:
    """Wrapper for video frame queue providing iterator interface."""

    def __init__(
        self,
        frames: "Queue[Optional[np.ndarray]]",
        initial_frame_timeout: float = DEFAULT_INITIAL_FRAME_TIMEOUT,
    ):
        self._frames = frames
        self._initial_frame_timeout = initial_frame_timeout
        self._first_frame_received = False

    def __call__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames.

        Yields BGR numpy arrays until the stream ends (None received).

        Raises:
            TimeoutError: If first frame not received within timeout period
        """
        while True:
            # Use timeout only for first frame to detect server not sending
            timeout = (
                self._initial_frame_timeout if not self._first_frame_received else None
            )

            try:
                frame = self._frames.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"No video frames received within {self._initial_frame_timeout}s timeout.\n"
                    "This likely means the server is not sending video.\n"
                    "Troubleshooting:\n"
                    "  - Check that stream_output is configured in your StreamConfig\n"
                    "  - Verify the workflow outputs match your configuration\n"
                    "  - Ensure the server has WebRTC enabled and is processing frames"
                )

            if frame is None:
                break

            self._first_frame_received = True
            yield frame


class _DataChannel:
    """Data channel handler managing event-based callbacks.

    Supports two types of handlers:
    1. Global handlers: Receive entire serialized_output_data dict + metadata
    2. Field-specific handlers: Receive individual field values + metadata
    """

    def __init__(self) -> None:
        self._field_handlers: dict[str, List[Callable]] = {}  # Field-specific handlers
        self._global_handler: Optional[Callable[[Any], None]] = None

    def bind(self, channel: RTCDataChannel) -> None:
        """Bind to an RTCDataChannel and register message handler.

        Args:
            channel: The data channel to bind to
        """

        @channel.on("message")
        def _on_message(message: Any) -> None:  # noqa: ANN401
            # Parse message and route to handlers
            try:
                parsed_message = json.loads(message)

                # Extract video metadata if present
                metadata = None
                video_metadata_dict = parsed_message.get("video_metadata")
                if video_metadata_dict:
                    try:
                        metadata = VideoMetadata(
                            frame_id=video_metadata_dict["frame_id"],
                            received_at=datetime.fromisoformat(
                                video_metadata_dict["received_at"]
                            ),
                            pts=video_metadata_dict.get("pts"),
                            time_base=video_metadata_dict.get("time_base"),
                            declared_fps=video_metadata_dict.get("declared_fps"),
                            measured_fps=video_metadata_dict.get("measured_fps"),
                        )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse video_metadata: {e}")

                # Get serialized output data
                serialized_data = parsed_message.get("serialized_output_data")

                # Call global handler if registered (receives full serialized_data dict + metadata)
                if self._global_handler:
                    try:
                        self._invoke_handler(
                            self._global_handler, serialized_data, metadata
                        )
                    except Exception:
                        logger.warning(
                            "Error calling global data channel handler", exc_info=True
                        )

                # Route to field-specific handlers if data is a dict
                if isinstance(serialized_data, dict):
                    for field_name, field_value in serialized_data.items():
                        if field_name in self._field_handlers:
                            for handler in list(self._field_handlers[field_name]):
                                try:
                                    self._invoke_handler(handler, field_value, metadata)
                                except Exception:
                                    logger.warning(
                                        f"Error calling handler for field '{field_name}'",
                                        exc_info=True,
                                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse data channel message as JSON")

    def _invoke_handler(
        self, handler: Callable, value: Any, metadata: Optional[VideoMetadata]
    ) -> None:  # noqa: ANN401
        """Invoke handler with appropriate signature (auto-detect via introspection).

        Supports two signatures:
        - handler(value, metadata) - receives both value and metadata
        - handler(value) - receives only value (backward compatible)

        Args:
            handler: The handler callable to invoke
            value: The field value to pass
            metadata: Optional video metadata to pass
        """
        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())

            # Check number of parameters (excluding *args, **kwargs)
            positional_params = [
                p
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]

            if len(positional_params) >= 2:
                # Handler expects both value and metadata (even if metadata is None)
                handler(value, metadata)
            else:
                # Handler expects only value
                handler(value)
        except Exception:
            # Fallback: try calling with just the value
            try:
                handler(value)
            except Exception:
                # If that also fails, log and re-raise
                logger.exception(f"Failed to invoke handler {handler}")
                raise

    def on_data(
        self, field_name: Optional[str] = None
    ) -> Callable[[Callable], Callable]:
        """Decorator to register data handlers.

        Args:
            field_name: If provided, registers a field-specific handler that receives
                       the field value and optional metadata. If None, registers a
                       global handler that receives the entire serialized_output_data dict.

        Returns:
            Decorator function

        Examples:
            # Handle specific field with metadata
            @session.data.on_data("property_definition")
            def handle_property(value: int, metadata: VideoMetadata):
                print(f"Frame {metadata.frame_id}: property={value}")

            # Handle specific field without metadata
            @session.data.on_data("property_definition")
            def handle_property(value: int):
                print(f"Property: {value}")

            # Handle entire output dict with metadata (global handler)
            @session.data.on_data()
            def handle_all(data: dict, metadata: VideoMetadata):
                print(f"Frame {metadata.frame_id}: {data}")

            # Handle entire output dict without metadata
            @session.data.on_data()
            def handle_all(data: dict):
                print(f"Data: {data}")
        """

        def decorator(fn: Callable) -> Callable:
            if field_name is None:
                # Global handler - receives entire serialized_output_data dict
                self._global_handler = fn
            else:
                # Field-specific handler
                if field_name not in self._field_handlers:
                    self._field_handlers[field_name] = []
                self._field_handlers[field_name].append(fn)
            return fn

        return decorator


class WebRTCSession(AbstractContextManager["WebRTCSession"]):
    """WebRTC session for streaming video and receiving inference results.

    This class manages the WebRTC peer connection, video streaming,
    and data channel communication with the inference server.
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
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._source = source
        self._image_input_name = image_input_name
        self._workflow_config = workflow_config
        self._config = stream_config

        # Internal state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pc: Optional[RTCPeerConnection] = None
        self._video_queue: "Queue[Optional[np.ndarray]]" = Queue(
            maxsize=VIDEO_QUEUE_MAX_SIZE
        )

        # Public APIs
        self.video = _VideoStream(self._video_queue)
        self.data = _DataChannel()

    def __enter__(self) -> "WebRTCSession":
        """Enter context manager - start event loop and initialize connection."""
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
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit context manager - cleanup resources."""
        try:
            # Close peer connection
            if self._loop and self._pc:
                asyncio.run_coroutine_threadsafe(self._pc.close(), self._loop).result()
        finally:
            try:
                # Cleanup source
                if self._loop and self._source:
                    asyncio.run_coroutine_threadsafe(
                        self._source.cleanup(), self._loop
                    ).result()
            finally:
                # Stop event loop
                if self._loop:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread:
                    self._loop_thread.join(timeout=EVENT_LOOP_SHUTDOWN_TIMEOUT)

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for session to complete.

        Blocks until the video stream ends (None received) or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds (None for indefinite)

        Raises:
            TimeoutError: If timeout expires before stream ends
        """
        try:
            while True:
                frame = self._video_queue.get(timeout=timeout)
                if frame is None:
                    break
        except queue.Empty:
            if timeout is not None:
                raise TimeoutError(
                    f"WebRTC session wait() timed out after {timeout}s.\n"
                    "The video stream did not end within the timeout period."
                )

    async def _get_turn_config(self) -> Optional[dict]:
        """Fetch TURN configuration from server or use user-provided config.

        Priority order:
        1. User-provided config via StreamConfig.turn_server (highest priority)
        2. Auto-fetch from server endpoint /query/webrtc_turn_config
        3. Skip TURN for localhost connections
        4. Graceful fallback to None if unavailable

        Returns:
            TURN configuration dict or None
        """
        # 1. Use user-provided config if available
        if self._config.turn_server:
            logger.debug("Using user-provided TURN configuration")
            return self._config.turn_server

        # 2. Skip TURN for localhost connections
        if self._api_url.startswith(("http://localhost", "http://127.0.0.1")):
            logger.debug("Skipping TURN for localhost connection")
            return None

        # 3. Try to auto-fetch from server
        try:
            logger.debug("Attempting to fetch TURN config from server")
            response = requests.get(
                f"{self._api_url}/query/webrtc_turn_config", timeout=5
            )
            response.raise_for_status()
            data = response.json()

            turn_config = {
                "urls": data["urls"],
                "username": data["username"],
                "credential": data["credential"],
            }
            logger.info("Successfully fetched TURN configuration from server")
            return turn_config
        except Exception as e:
            # 4. Graceful fallback - proceed without TURN
            logger.info(
                f"TURN configuration not available ({e.__class__.__name__}), "
                "proceeding without TURN server",
                exc_info=True,
            )
            return None

    async def _init(self) -> None:
        """Initialize WebRTC connection.

        Sets up peer connection, configures source, negotiates with server.
        """
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
                while True:
                    try:
                        f: VideoFrame = await subscribed.recv()
                    except Exception as e:
                        # Connection closed or track ended
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
                    # Backpressure: drop oldest frame if queue full
                    if self._video_queue.full():
                        try:
                            _ = self._video_queue.get_nowait()
                        except Exception:
                            pass
                    try:
                        self._video_queue.put_nowait(img)
                    except Exception:
                        pass

            asyncio.ensure_future(_reader())

        # Setup data channel
        ch = pc.createDataChannel("inference")
        self.data.bind(ch)

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
        wf_conf: dict[str, Any] = {
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
        # (rtsp_url for RTSP, declared_fps for webcam, etc.)
        payload.update(self._source.get_initialization_params())

        # Call server to initialize worker
        url = f"{self._api_url}/initialise_webrtc_worker"
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        ans: dict[str, Any] = resp.json()

        # Set remote description
        answer = RTCSessionDescription(sdp=ans["sdp"], type=ans["type"])
        await pc.setRemoteDescription(answer)

        self._pc = pc
