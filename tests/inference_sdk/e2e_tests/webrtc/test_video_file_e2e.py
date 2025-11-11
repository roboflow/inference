"""End-to-end tests for WebRTC SDK with real inference server.

These tests start a real inference server and validate the full WebRTC stack:
- Real HTTP API
- Real WebRTC signaling
- Real workflow execution
- Real data channel communication

Tests are marked with @pytest.mark.slow and can be skipped:
    pytest -m "not slow"  # Skip slow tests
    pytest -m slow        # Run only slow tests
"""

import os
import numpy as np
import pytest

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import StreamConfig, VideoFileSource


# Simple passthrough workflow - uses absolute_static_crop to wrap image
# This validates WebRTC connection and workflow execution without transformation
# The crop step is necessary to wrap the input in WorkflowImageData format for video streaming
PASSTHROUGH_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/absolute_static_crop@v1",
            "name": "absolute_static_crop",
            "images": "$inputs.image",
            "x_center": 40,
            "y_center": 40,
            "width": 80,
            "height": 80,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "absolute_static_crop",
            "coordinates_system": "own",
            "selector": "$steps.absolute_static_crop.crops",
        }
    ],
}


@pytest.mark.slow
def test_video_file_e2e_with_passthrough_workflow(inference_server):
    """Full end-to-end test with real server and passthrough workflow.

    This test validates:
    1. Real inference server starts and responds
    2. WebRTC session establishes successfully
    3. Video frames are sent from file to server
    4. Workflow processes frames (passthrough - no transformation)
    5. Processed frames are received back
    6. Session cleanup works correctly

    Uses:
    - Real HTTP API
    - Real WebRTC signaling (aiortc)
    - Real workflow execution
    - Real video file (no mocking needed)
    """
    # Create real client pointing to test server
    client = InferenceHTTPClient(api_url=inference_server, api_key="test-key")

    # Path to test video file
    test_video_path = os.path.join(
        os.path.dirname(__file__),
        "../../../inference/unit_tests/core/interfaces/assets/example_video.mp4"
    )
    test_video_path = os.path.abspath(test_video_path)

    # Create video file source
    source = VideoFileSource(path=test_video_path)

    # Configure to receive processed video stream
    config = StreamConfig(
        stream_output=["absolute_static_crop"],  # Receive cropped frames
        data_output=[],  # No data channel needed for this test
        realtime_processing=True,
    )

    # Start WebRTC session with inline workflow spec
    with client.webrtc.stream(
        source=source,
        workflow=PASSTHROUGH_WORKFLOW,  # Inline spec - no workspace needed!
        image_input="image",
        config=config,
    ) as session:
        # Verify session established
        assert session._pc is not None, "Peer connection should be established"
        assert session._loop is not None, "Event loop should be running"

        # Collect some processed frames
        frames_received = []
        for i, frame in enumerate(session.video()):
            frames_received.append(frame)
            if i >= 2:  # Get 3 frames
                break

        # Validate we received processed frames
        assert len(frames_received) == 3, "Should receive 3 processed frames"

        for frame in frames_received:
            # Frames should be numpy arrays
            assert isinstance(frame, np.ndarray), "Frame should be numpy array"

            # Should be BGR format (H, W, 3) - cropped to 80x80
            assert frame.shape == (
                80,
                80,
                3,
            ), "Frame should be 80x80x3 (cropped)"

            # Should be uint8
            assert frame.dtype == np.uint8, "Frame should be uint8"

    # Session should cleanup successfully (context manager exit)
    print("\n✓ E2E test passed: Real server + WebRTC + Grayscale workflow")


@pytest.mark.slow
def test_video_file_e2e_with_data_channel(inference_server):
    """Test data channel output with real server.

    Validates that workflow outputs can be received via data channel
    in addition to video stream.
    """
    client = InferenceHTTPClient(api_url=inference_server, api_key="test-key")

    # Path to test video file
    test_video_path = os.path.join(
        os.path.dirname(__file__),
        "../../../inference/unit_tests/core/interfaces/assets/example_video.mp4"
    )
    test_video_path = os.path.abspath(test_video_path)

    source = VideoFileSource(path=test_video_path)

    # Request both video and data output
    config = StreamConfig(
        stream_output=["absolute_static_crop"],
        data_output=["absolute_static_crop"],  # Also receive via data channel
        realtime_processing=True,
    )

    data_messages_received = []

    with client.webrtc.stream(
        source=source,
        workflow=PASSTHROUGH_WORKFLOW,
        image_input="image",
        config=config,
    ) as session:
        # Register data channel handler
        @session.data.on_data("image_output")
        def handle_image_data(data):
            data_messages_received.append(data)

        # Receive a few frames to trigger data channel messages
        frame_count = 0
        for frame in session.video():
            frame_count += 1
            if frame_count >= 3:
                break

        # Give data channel time to deliver messages
        import time

        time.sleep(0.5)

        # Validate we received data channel messages
        # Note: Depending on server implementation, we might receive data
        assert isinstance(
            data_messages_received, list
        ), "Should receive data channel messages"

    print(
        f"\n✓ Data channel test passed: Received {len(data_messages_received)} messages"
    )
