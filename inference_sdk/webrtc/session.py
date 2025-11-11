"""WebRTC session management."""

import asyncio
import json
import logging
import queue
import sys
import threading
from contextlib import AbstractContextManager
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

# Configuration constants
DEFAULT_INITIAL_FRAME_TIMEOUT = 30.0  # seconds to wait for first video frame
VIDEO_QUEUE_MAX_SIZE = 8  # maximum number of frames to buffer
EVENT_LOOP_SHUTDOWN_TIMEOUT = 2.0  # seconds to wait for event loop thread to stop

# Configure basic logging if root logger has no handlers
# This ensures users see important errors even without explicit logging setup
if not logging.root.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(levelname)s [%(name)s] %(message)s")
    )
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
    """Data channel handler managing event-based callbacks."""

    def __init__(self) -> None:
        self._handlers: dict[str, List[Callable[[Any], None]]] = {}
        self._global_handler: Optional[Callable[[Any], None]] = None

    def bind(self, channel: RTCDataChannel) -> None:
        """Bind to an RTCDataChannel and register message handler.

        Args:
            channel: The data channel to bind to
        """

        @channel.on("message")
        def _on_message(message: Any) -> None:  # noqa: ANN401
            # Call global handler if registered (receives raw message)
            if self._global_handler:
                try:
                    self._global_handler(message)
                except Exception:
                    logger.warning(
                        "Error calling global data channel handler", exc_info=True
                    )

            # Parse message and route to specific handlers
            try:
                parsed_message = json.loads(message)
                output_name = parsed_message.get("output_name")
                if output_name and output_name in self._handlers:
                    serialized_data = parsed_message.get("serialized_output_data")
                    for cb in list(self._handlers[output_name]):
                        try:
                            cb(serialized_data)
                        except Exception:
                            logger.warning(
                                f"Error calling handler for output '{output_name}'",
                                exc_info=True,
                            )
            except json.JSONDecodeError:
                logger.warning("Failed to parse data channel message as JSON")

    def on_data(
        self, output_name: Optional[str] = None
    ) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        """Decorator to register data handlers.

        Args:
            output_name: If provided, handler is called only for this output.
                        If None, handler receives all raw messages.

        Returns:
            Decorator function

        Examples:
            # Handle specific output
            @session.on_data("predictions")
            def handle_predictions(data):
                print("Predictions:", data)

            # Handle all messages
            @session.on_data()
            def handle_all(raw_message):
                print("Raw message:", raw_message)
        """

        def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
            if output_name is None:
                self._global_handler = fn
            else:
                if output_name not in self._handlers:
                    self._handlers[output_name] = []
                self._handlers[output_name].append(fn)
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
                "proceeding without TURN server",exc_info=True
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
