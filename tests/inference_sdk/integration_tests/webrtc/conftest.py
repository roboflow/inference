"""Shared fixtures for WebRTC integration tests."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from av import VideoFrame


@pytest.fixture
def mock_video_capture():
    """Mock cv2.VideoCapture to return synthetic frames."""

    class MockVideoCapture:
        def __init__(self, device_id):
            self.device_id = device_id
            self._frame_count = 0
            self._is_opened = True
            self._width = 640
            self._height = 480
            self._fps = 30.0

        def isOpened(self):
            return self._is_opened

        def read(self):
            if not self._is_opened or self._frame_count >= 100:  # Limit frames
                return False, None

            # Generate synthetic BGR frame
            frame = np.random.randint(0, 255, (self._height, self._width, 3), dtype=np.uint8)
            self._frame_count += 1
            return True, frame

        def set(self, prop, value):
            if prop == 3:  # CAP_PROP_FRAME_WIDTH
                self._width = int(value)
            elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
                self._height = int(value)
            return True

        def get(self, prop):
            if prop == 3:  # CAP_PROP_FRAME_WIDTH
                return self._width
            elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
                return self._height
            elif prop == 5:  # CAP_PROP_FPS
                return self._fps
            return 0

        def release(self):
            self._is_opened = False

    with patch("cv2.VideoCapture", MockVideoCapture):
        yield MockVideoCapture


@pytest.fixture
def mock_rtc_peer_connection():
    """Mock RTCPeerConnection and related WebRTC components."""

    class MockRTCPeerConnection:
        def __init__(self, configuration=None):
            self.configuration = configuration
            self.iceGatheringState = "complete"
            self.localDescription = Mock()
            self.localDescription.type = "offer"
            self.localDescription.sdp = "mock-sdp-offer"
            self._tracks = []
            self._transceivers = []
            self._data_channels = {}
            self._track_handlers = []

        def addTrack(self, track):
            self._tracks.append(track)

        def addTransceiver(self, kind, direction=None):
            transceiver = Mock()
            transceiver.kind = kind
            transceiver.direction = direction
            self._transceivers.append(transceiver)
            return transceiver

        def createDataChannel(self, label):
            channel = MockRTCDataChannel(label)
            self._data_channels[label] = channel
            return channel

        async def createOffer(self):
            return self.localDescription

        async def setLocalDescription(self, desc):
            self.localDescription = desc

        async def setRemoteDescription(self, desc):
            pass

        async def close(self):
            pass

        def on(self, event):
            """Decorator for event handlers."""
            def decorator(func):
                if event == "track":
                    self._track_handlers.append(func)
                return func
            return decorator

        def simulate_incoming_track(self, num_frames=5):
            """Helper to simulate incoming video track for testing."""
            mock_track = MockVideoTrack(num_frames=num_frames)
            for handler in self._track_handlers:
                handler(mock_track)
            return mock_track

    class MockRTCDataChannel:
        def __init__(self, label):
            self.label = label
            self._message_handlers = []
            self.readyState = "open"

        def on(self, event):
            """Decorator for event handlers."""
            def decorator(func):
                if event == "message":
                    self._message_handlers.append(func)
                return func
            return decorator

        def send_message(self, message):
            """Helper to simulate incoming message for testing."""
            for handler in self._message_handlers:
                handler(message)

    class MockVideoTrack:
        """Mock video track that provides frames."""
        def __init__(self, num_frames=5):
            self.num_frames = num_frames
            self._frame_idx = 0

        async def recv(self):
            if self._frame_idx >= self.num_frames:
                raise Exception("Track ended")

            # Create synthetic video frame
            arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(arr, format="bgr24")
            frame.pts = self._frame_idx
            frame.time_base = "1/30"
            self._frame_idx += 1
            return frame

    with patch("inference_sdk.webrtc.session.RTCPeerConnection", MockRTCPeerConnection), \
         patch("inference_sdk.webrtc.session.MediaRelay") as mock_relay:

        # Mock MediaRelay to pass through the track
        mock_relay_instance = Mock()
        mock_relay_instance.subscribe = lambda track: track
        mock_relay.return_value = mock_relay_instance

        yield MockRTCPeerConnection


@pytest.fixture
def mock_server_endpoints():
    """Mock server HTTP endpoints for worker initialization."""

    with patch("requests.post") as mock_post:

        # Mock worker initialization response
        def post_side_effect(url, **kwargs):
            response = Mock()
            if "initialise_webrtc_worker" in url:
                response.json.return_value = {
                    "sdp": "mock-sdp-answer",
                    "type": "answer"
                }
                response.raise_for_status = Mock()
            else:
                response.raise_for_status = Mock(side_effect=Exception("Not found"))
            return response

        mock_post.side_effect = post_side_effect

        yield {"post": mock_post}


@pytest.fixture
def test_video_path():
    """Path to the test video file."""
    return Path(__file__).parent / "data" / "test_video.mp4"


@pytest.fixture
def sample_workflow_config():
    """Sample workflow configuration for testing."""
    return {
        "workspace_name": "test-workspace",
        "workflow_id": "test-workflow"
    }


@pytest.fixture
def sample_stream_config():
    """Sample stream configuration for testing."""
    from inference_sdk.webrtc import StreamConfig

    return StreamConfig(
        stream_output=["image_output"],
        data_output=["predictions"],
        realtime_processing=True
    )


@pytest.fixture
def mock_inference_client():
    """Mock InferenceHTTPClient for testing."""
    client = Mock()
    client.api_url = "http://test-server.com"
    client.api_key = "test-api-key"
    return client


@pytest.fixture
def enable_all_mocks(mock_video_capture, mock_rtc_peer_connection, mock_server_endpoints):
    """Convenience fixture to enable all common mocks at once."""
    return {
        "video_capture": mock_video_capture,
        "peer_connection": mock_rtc_peer_connection,
        "server_endpoints": mock_server_endpoints
    }
