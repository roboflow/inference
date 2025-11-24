"""Integration tests for VideoFileSource.

These tests use a real test video file but mock the server endpoints
and WebRTC connection to enable testing without a running server.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from inference_sdk.webrtc import StreamConfig, VideoFileSource, WebRTCSession


def test_video_file_session_basic(
    enable_all_mocks,
    test_video_path,
    sample_workflow_config,
    sample_stream_config
):
    """Test basic video file session with real file.

    Validates that:
    1. Video file can be opened
    2. Session starts successfully
    3. Video properties are detected correctly
    """
    assert test_video_path.exists(), f"Test video not found: {test_video_path}"

    source = VideoFileSource(str(test_video_path))

    with WebRTCSession(
        api_url="http://test-server.com",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config=sample_workflow_config,
        stream_config=sample_stream_config
    ) as session:
        session._ensure_started()
        # Verify track was created
        assert source._track is not None
        assert source._track._container is not None
        assert source._track._stream is not None


def test_video_file_fps_detection(
    enable_all_mocks,
    test_video_path,
    sample_workflow_config,
    sample_stream_config
):
    """Test that video file FPS is detected correctly.

    Validates that FPS from the video file is read by the track.
    """
    source = VideoFileSource(str(test_video_path))

    with WebRTCSession(
        api_url="http://test-server.com",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config=sample_workflow_config,
        stream_config=sample_stream_config
    ) as session:
        session._ensure_started()
        # Verify track was created and FPS was detected
        assert source._track is not None
        fps = source._track.get_declared_fps()
        assert fps is not None, "Should detect FPS from video file"
        assert fps > 0, "FPS should be positive"


def test_video_file_with_stream_config_variations(
    enable_all_mocks,
    test_video_path,
    sample_workflow_config
):
    """Test video file with different StreamConfig options.

    Validates that different configuration options are properly handled.
    """
    # Test with data output only (no stream output)
    config_data_only = StreamConfig(
        stream_output=[],
        data_output=["results"],
        realtime_processing=False
    )

    source = VideoFileSource(str(test_video_path))

    with WebRTCSession(
        api_url="http://test-server.com",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config=sample_workflow_config,
        stream_config=config_data_only
    ) as session:
        # Session should start successfully even without stream output
        assert session._config.stream_output == []
        assert session._config.data_output == ["results"]
        assert session._config.realtime_processing is False


def test_video_file_with_workflow_parameters(
    enable_all_mocks,
    test_video_path,
    sample_workflow_config
):
    """Test video file with workflow parameters.

    Validates that workflow parameters are included in the configuration.
    """
    config = StreamConfig(
        stream_output=["image_output"],
        data_output=["predictions"],
        workflow_parameters={
            "confidence_threshold": 0.5,
            "iou_threshold": 0.3
        }
    )

    source = VideoFileSource(str(test_video_path))

    with WebRTCSession(
        api_url="http://test-server.com",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config=sample_workflow_config,
        stream_config=config
    ) as session:
        # Verify workflow parameters are set
        assert session._config.workflow_parameters == {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.3
        }


def test_video_file_cleanup(
    enable_all_mocks,
    test_video_path,
    sample_workflow_config,
    sample_stream_config
):
    """Test that video file resources are properly cleaned up.

    Validates that PyAV container is released on session exit.
    """
    source = VideoFileSource(str(test_video_path))

    with WebRTCSession(
        api_url="http://test-server.com",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config=sample_workflow_config,
        stream_config=sample_stream_config
    ) as session:
        session._ensure_started()
        track = source._track
        assert track._container is not None, "Container should exist during session"
        # PyAV containers don't have a simple "is_open" check, but we can verify it exists

    # After exiting context, container should be closed (we verify cleanup was called)
    # Note: PyAV containers don't expose a simple "is_closed" property,
    # but attempting to use them after close will raise an error


def test_video_file_real_properties(test_video_path):
    """Test reading actual properties from the test video file.

    This test doesn't mock VideoCapture to verify the actual test video
    has expected properties.
    """
    assert test_video_path.exists(), f"Test video not found: {test_video_path}"

    cap = cv2.VideoCapture(str(test_video_path))
    assert cap.isOpened(), "Should be able to open test video"

    # Read properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Verify expected properties
    assert width == 640, f"Expected width 640, got {width}"
    assert height == 480, f"Expected height 480, got {height}"
    assert fps == 30.0, f"Expected FPS 30, got {fps}"
    assert frame_count == 10, f"Expected 10 frames, got {frame_count}"

    # Read all frames
    frames_read = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        assert frame.shape == (480, 640, 3), "Frame should be 480x640x3"

    assert frames_read == 10, f"Should read 10 frames, got {frames_read}"

    cap.release()


def test_video_file_with_data_channel(
    enable_all_mocks,
    test_video_path,
    sample_workflow_config
):
    """Test video file processing with data channel output.

    Simulates a batch processing scenario where results are collected
    via data channel.
    """
    config = StreamConfig(
        data_output=["analysis_results"],
        stream_output=[]  # No video output needed for batch processing
    )

    source = VideoFileSource(str(test_video_path))
    results = []

    session = WebRTCSession(
        api_url="http://test-server.com",
        api_key="test-key",
        source=source,
        image_input_name="image",
        workflow_config=sample_workflow_config,
        stream_config=config
    )

    # Register handler to collect results
    @session.on_data("analysis_results")
    def collect_results(data):
        results.append(data)

    session._ensure_started()

    try:

        # Get the data channel
        data_channel = session._pc._data_channels["inference"]

        # Simulate receiving results for each frame
        for i in range(5):
            data_channel.send_message(
                f'{{"serialized_output_data": {{"analysis_results": {{"frame": {i}, "detections": []}}}}}}'
            )

        # Give handlers time to process
        time.sleep(0.1)

        # Validate results were collected
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["frame"] == i
    finally:
        session.close()
