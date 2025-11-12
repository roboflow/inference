"""Unit tests for WebRTC session decorator functionality."""

import json
from datetime import datetime
from queue import Queue
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference_sdk.webrtc.session import VideoMetadata, WebRTCSession


@pytest.fixture
def mock_session():
    """Create a mock WebRTCSession instance without actually initializing WebRTC."""
    with patch("inference_sdk.webrtc.session._check_webrtc_dependencies"):
        session = WebRTCSession(
            api_url="http://localhost:9001",
            api_key="test_key",
            source=MagicMock(),
            image_input_name="image",
            workflow_config={},
            stream_config=MagicMock(),
        )
    return session


class TestOnFrameDecorator:
    """Tests for @session.on_frame decorator."""

    def test_on_frame_registration(self, mock_session):
        """Test that on_frame decorator registers handler."""
        handler_called = []

        @mock_session.on_frame
        def process_frame(frame, metadata):
            handler_called.append((frame, metadata))

        assert len(mock_session._frame_handlers) == 1
        assert mock_session._frame_handlers[0] == process_frame

    def test_on_frame_multiple_handlers(self, mock_session):
        """Test registering multiple frame handlers."""

        @mock_session.on_frame
        def handler1(frame, metadata):
            pass

        @mock_session.on_frame
        def handler2(frame, metadata):
            pass

        assert len(mock_session._frame_handlers) == 2
        assert handler1 in mock_session._frame_handlers
        assert handler2 in mock_session._frame_handlers

    def test_on_frame_returns_callback(self, mock_session):
        """Test that decorator returns the original function."""

        def my_handler(frame, metadata):
            pass

        result = mock_session.on_frame(my_handler)
        assert result is my_handler

    def test_run_calls_frame_handlers(self, mock_session):
        """Test that run() calls registered frame handlers with frame and metadata."""
        received_frames = []
        received_metadata = []

        @mock_session.on_frame
        def process_frame(frame, metadata):
            received_frames.append(frame)
            received_metadata.append(metadata)
            if len(received_frames) >= 2:
                mock_session.stop()

        # Create test frames with metadata
        test_frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata1 = VideoMetadata(
            frame_id=1,
            received_at=datetime.now(),
            pts=1000,
            time_base=0.001,
        )

        test_frame2 = np.ones((100, 100, 3), dtype=np.uint8)
        test_metadata2 = VideoMetadata(
            frame_id=2,
            received_at=datetime.now(),
            pts=2000,
            time_base=0.001,
        )

        # Put frames in queue
        mock_session._video_queue.put((test_frame1, test_metadata1))
        mock_session._video_queue.put((test_frame2, test_metadata2))

        # Run session
        mock_session.run()

        # Verify handlers were called with correct arguments
        assert len(received_frames) == 2
        assert np.array_equal(received_frames[0], test_frame1)
        assert np.array_equal(received_frames[1], test_frame2)

        assert len(received_metadata) == 2
        assert received_metadata[0].frame_id == 1
        assert received_metadata[0].pts == 1000
        assert received_metadata[1].frame_id == 2
        assert received_metadata[1].pts == 2000

    def test_run_handles_handler_exceptions(self, mock_session):
        """Test that exceptions in handlers don't crash run()."""
        handler1_calls = []
        handler2_calls = []

        @mock_session.on_frame
        def failing_handler(frame, metadata):
            handler1_calls.append(True)
            raise ValueError("Handler error")

        @mock_session.on_frame
        def working_handler(frame, metadata):
            handler2_calls.append(True)
            mock_session.stop()

        # Put a frame in queue
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
        mock_session._video_queue.put((test_frame, test_metadata))

        # Run should not raise despite first handler failing
        mock_session.run()

        # Both handlers should have been called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_run_stops_on_stop_event(self, mock_session):
        """Test that run() stops when stop() is called."""
        frame_count = []

        @mock_session.on_frame
        def count_frames(frame, metadata):
            frame_count.append(1)
            if len(frame_count) >= 2:
                mock_session.stop()

        # Put multiple frames in queue
        for i in range(10):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=i, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))

        mock_session.run()

        # Should have stopped after 2 frames
        assert len(frame_count) == 2


class TestOnDataDecorator:
    """Tests for @session.on_data decorator."""

    def test_on_data_global_handler_with_parentheses(self, mock_session):
        """Test registering global handler with @session.on_data()."""
        handler_called = []

        @mock_session.on_data()
        def handle_data(data, metadata):
            handler_called.append((data, metadata))

        assert mock_session._data_global_handler == handle_data

    def test_on_data_global_handler_without_parentheses(self, mock_session):
        """Test registering global handler with @session.on_data (no parentheses)."""
        handler_called = []

        @mock_session.on_data
        def handle_data(data, metadata):
            handler_called.append((data, metadata))

        assert mock_session._data_global_handler == handle_data

    def test_on_data_field_specific_handler(self, mock_session):
        """Test registering field-specific handler."""

        @mock_session.on_data("predictions")
        def handle_predictions(value, metadata):
            pass

        assert "predictions" in mock_session._data_field_handlers
        assert handle_predictions in mock_session._data_field_handlers["predictions"]

    def test_on_data_multiple_field_handlers(self, mock_session):
        """Test multiple handlers for same field."""

        @mock_session.on_data("predictions")
        def handler1(value, metadata):
            pass

        @mock_session.on_data("predictions")
        def handler2(value, metadata):
            pass

        assert len(mock_session._data_field_handlers["predictions"]) == 2
        assert handler1 in mock_session._data_field_handlers["predictions"]
        assert handler2 in mock_session._data_field_handlers["predictions"]

    def test_on_data_different_fields(self, mock_session):
        """Test handlers for different fields."""

        @mock_session.on_data("predictions")
        def handle_predictions(value, metadata):
            pass

        @mock_session.on_data("detections")
        def handle_detections(value, metadata):
            pass

        assert "predictions" in mock_session._data_field_handlers
        assert "detections" in mock_session._data_field_handlers
        assert len(mock_session._data_field_handlers["predictions"]) == 1
        assert len(mock_session._data_field_handlers["detections"]) == 1

    def test_on_data_returns_callback(self, mock_session):
        """Test that decorator returns the original function."""

        def my_handler(data, metadata):
            pass

        result = mock_session.on_data("test")(my_handler)
        assert result is my_handler


class TestInvokeDataHandler:
    """Tests for _invoke_data_handler method."""

    def test_invoke_with_two_parameters(self, mock_session):
        """Test invoking handler that expects (value, metadata)."""
        received_args = []

        def handler(value, metadata):
            received_args.append((value, metadata))

        metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
        mock_session._invoke_data_handler(handler, 42, metadata)

        assert len(received_args) == 1
        assert received_args[0][0] == 42
        assert received_args[0][1] == metadata

    def test_invoke_with_one_parameter(self, mock_session):
        """Test invoking handler that expects only value (backward compatible)."""
        received_values = []

        def handler(value):
            received_values.append(value)

        metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
        mock_session._invoke_data_handler(handler, 42, metadata)

        assert len(received_values) == 1
        assert received_values[0] == 42

    def test_invoke_with_none_metadata(self, mock_session):
        """Test invoking handler when metadata is None."""
        received_args = []

        def handler(value):
            received_args.append(value)

        mock_session._invoke_data_handler(handler, 42, None)

        assert len(received_args) == 1
        assert received_args[0] == 42


class TestVideoStream:
    """Tests for video stream iterator."""

    def test_video_stream_yields_tuples(self, mock_session):
        """Test that session.video() yields (frame, metadata) tuples."""
        # Put test frames in queue
        test_frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata1 = VideoMetadata(frame_id=1, received_at=datetime.now())

        test_frame2 = np.ones((100, 100, 3), dtype=np.uint8)
        test_metadata2 = VideoMetadata(frame_id=2, received_at=datetime.now())

        mock_session._video_queue.put((test_frame1, test_metadata1))
        mock_session._video_queue.put((test_frame2, test_metadata2))
        mock_session._video_queue.put(None)  # End stream

        # Iterate and collect
        frames = []
        metadatas = []
        for frame, metadata in mock_session.video():
            frames.append(frame)
            metadatas.append(metadata)

        assert len(frames) == 2
        assert np.array_equal(frames[0], test_frame1)
        assert np.array_equal(frames[1], test_frame2)

        assert len(metadatas) == 2
        assert metadatas[0].frame_id == 1
        assert metadatas[1].frame_id == 2

    def test_video_stream_ends_on_none(self, mock_session):
        """Test that video stream ends when None is received."""
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())

        mock_session._video_queue.put((test_frame, test_metadata))
        mock_session._video_queue.put(None)  # End stream

        frames = list(mock_session.video())
        assert len(frames) == 1

    def test_video_stream_timeout_on_first_frame(self, mock_session):
        """Test that timeout occurs if first frame not received."""
        # Don't put any frames in queue
        mock_session.video._initial_frame_timeout = 0.1

        with pytest.raises(TimeoutError) as exc_info:
            for _ in mock_session.video():
                pass

        assert "No video frames received" in str(exc_info.value)


class TestStopMethod:
    """Tests for stop() method."""

    def test_stop_sets_event(self, mock_session):
        """Test that stop() sets the stop event."""
        assert not mock_session._stop_event.is_set()
        mock_session.stop()
        assert mock_session._stop_event.is_set()

    def test_stop_can_be_called_from_handler(self, mock_session):
        """Test that stop() can be called from within a frame handler."""
        calls = []

        @mock_session.on_frame
        def handler(frame, metadata):
            calls.append(1)
            mock_session.stop()

        # Put frames in queue
        for i in range(5):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=i, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))

        mock_session.run()

        # Should have stopped after first frame
        assert len(calls) == 1


class TestWaitMethod:
    """Tests for wait() method."""

    def test_wait_blocks_until_stream_ends(self, mock_session):
        """Test that wait() blocks until None is received."""
        # Put frames in queue
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())

        mock_session._video_queue.put((test_frame, test_metadata))
        mock_session._video_queue.put(None)  # End stream

        # Should not raise and should consume all frames
        mock_session.wait()

    def test_wait_timeout(self, mock_session):
        """Test that wait() raises TimeoutError on timeout."""
        # Put a frame but no end signal
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
        mock_session._video_queue.put((test_frame, test_metadata))

        with pytest.raises(TimeoutError) as exc_info:
            mock_session.wait(timeout=0.1)

        assert "timed out" in str(exc_info.value)
