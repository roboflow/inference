"""Unit tests for WebRTC session lifecycle management."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np

from inference_sdk.webrtc.session import VideoMetadata, WebRTCSession, SessionState


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


class TestSessionLifecycle:
    """Tests for session lifecycle (creation, starting, closing)."""

    def test_session_starts_in_not_started_state(self, mock_session):
        """Test that session is created in not_started state."""
        assert mock_session._state == SessionState.NOT_STARTED

    def test_close_is_idempotent(self, mock_session):
        """Test that close() can be called multiple times safely."""
        mock_session._state = SessionState.STARTED  # Simulate started state
        mock_session.close()
        assert mock_session._state == SessionState.CLOSED

        # Second call should be a no-op
        mock_session.close()
        assert mock_session._state == SessionState.CLOSED

    def test_ensure_started_changes_state(self, mock_session):
        """Test that _ensure_started() transitions from not_started to started."""
        with patch.object(mock_session, "_init_connection"):
            assert mock_session._state == SessionState.NOT_STARTED
            mock_session._ensure_started()
            assert mock_session._state == SessionState.STARTED

    def test_ensure_started_is_idempotent(self, mock_session):
        """Test that _ensure_started() can be called multiple times."""
        with patch.object(mock_session, "_init_connection") as mock_init:
            mock_session._ensure_started()
            mock_session._ensure_started()
            mock_session._ensure_started()

            # _init_connection should only be called once
            assert mock_init.call_count == 1

    def test_ensure_started_raises_on_closed_session(self, mock_session):
        """Test that _ensure_started() raises error if session is closed."""
        mock_session._state = SessionState.CLOSED

        with pytest.raises(RuntimeError, match="Cannot use closed WebRTCSession"):
            mock_session._ensure_started()


class TestRunMethod:
    """Tests for run() method and exception handling."""

    def test_run_auto_starts_session(self, mock_session):
        """Test that run() automatically starts the session."""
        with patch.object(mock_session, "_ensure_started") as mock_ensure:
            # Put a frame and immediately close
            @mock_session.on_frame
            def handler(frame, metadata):
                mock_session.close()

            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._state = SessionState.STARTED

            mock_session.run()

            # Should have called _ensure_started
            mock_ensure.assert_called_once()

    def test_run_stops_when_close_called(self, mock_session):
        """Test that run() stops when close() is called from handler."""
        frame_count = []

        @mock_session.on_frame
        def count_frames(frame, metadata):
            frame_count.append(1)
            if len(frame_count) >= 2:
                mock_session.close()

        # Put multiple frames in queue (use put_nowait to avoid blocking on full queue)
        for i in range(5):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=i, received_at=datetime.now())
            mock_session._video_queue.put_nowait((test_frame, test_metadata))

        # Mock state as started
        mock_session._state = SessionState.STARTED

        mock_session.run()

        # Should have stopped after 2 frames (when close() was called)
        assert len(frame_count) == 2

    def test_run_handles_handler_exceptions_gracefully(self, mock_session):
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
            mock_session.close()

        # Put a frame in queue
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
        mock_session._video_queue.put((test_frame, test_metadata))

        mock_session._state = SessionState.STARTED

        # Run should not raise despite first handler failing
        mock_session.run()

        # Both handlers should have been called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    def test_run_closes_session_on_exception(self, mock_session):
        """Test that run() closes session if exception occurs."""
        mock_session._state = SessionState.STARTED

        # Mock video() to raise an exception
        def raise_exception():
            raise RuntimeError("Test error")
            yield  # Never reached

        with patch.object(mock_session, "video", return_value=raise_exception()):
            with patch.object(mock_session, "close") as mock_close:
                with pytest.raises(RuntimeError, match="Test error"):
                    mock_session.run()

                # Should have called close()
                mock_close.assert_called_once()

    def test_run_closes_session_on_keyboard_interrupt(self, mock_session):
        """Test that run() closes session on Ctrl+C."""
        mock_session._state = SessionState.STARTED

        # Mock video() to raise KeyboardInterrupt
        def raise_interrupt():
            raise KeyboardInterrupt()
            yield  # Never reached

        with patch.object(mock_session, "video", return_value=raise_interrupt()):
            with patch.object(mock_session, "close") as mock_close:
                with pytest.raises(KeyboardInterrupt):
                    mock_session.run()

                # Should have called close()
                mock_close.assert_called_once()


class TestDecorators:
    """Tests for decorator registration."""

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

    def test_on_data_global_handler(self, mock_session):
        """Test registering global data handler."""

        @mock_session.on_data()
        def handle_data(data, metadata):
            pass

        assert mock_session._data_global_handler == handle_data

    def test_on_data_field_specific_handler(self, mock_session):
        """Test registering field-specific data handler."""

        @mock_session.on_data("predictions")
        def handle_predictions(value, metadata):
            pass

        assert "predictions" in mock_session._data_field_handlers
        assert handle_predictions in mock_session._data_field_handlers["predictions"]


class TestVideoStream:
    """Tests for video stream iterator."""

    def test_video_auto_starts_session(self, mock_session):
        """Test that video() automatically starts the session."""
        with patch.object(mock_session, "_ensure_started") as mock_ensure:
            # Put a frame and end signal
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._video_queue.put(None)  # End stream

            # Iterate (should auto-start)
            list(mock_session.video())

            # Should have called _ensure_started
            mock_ensure.assert_called_once()

    def test_video_yields_frame_tuples(self, mock_session):
        """Test that video() yields (frame, metadata) tuples."""
        # Put test frames in queue
        test_frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        test_metadata1 = VideoMetadata(frame_id=1, received_at=datetime.now())

        test_frame2 = np.ones((100, 100, 3), dtype=np.uint8)
        test_metadata2 = VideoMetadata(frame_id=2, received_at=datetime.now())

        mock_session._video_queue.put((test_frame1, test_metadata1))
        mock_session._video_queue.put((test_frame2, test_metadata2))
        mock_session._video_queue.put(None)  # End stream

        # Mock _ensure_started
        with patch.object(mock_session, "_ensure_started"):
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


class TestWaitMethod:
    """Tests for wait() method."""

    def test_wait_auto_starts_session(self, mock_session):
        """Test that wait() automatically starts the session."""
        with patch.object(mock_session, "_ensure_started") as mock_ensure:
            # Put a frame and end signal
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())

            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._video_queue.put(None)  # End stream

            # Should not raise and should consume all frames
            mock_session.wait()

            # Should have called _ensure_started
            mock_ensure.assert_called_once()

    def test_wait_blocks_until_stream_ends(self, mock_session):
        """Test that wait() blocks until None is received."""
        with patch.object(mock_session, "_ensure_started"):
            # Put frames in queue
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())

            mock_session._video_queue.put((test_frame, test_metadata))
            mock_session._video_queue.put(None)  # End stream

            # Should not raise and should consume all frames
            mock_session.wait()

    def test_wait_timeout(self, mock_session):
        """Test that wait() raises TimeoutError on timeout."""
        with patch.object(mock_session, "_ensure_started"):
            # Put a frame but no end signal
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=1, received_at=datetime.now())
            mock_session._video_queue.put((test_frame, test_metadata))

            with pytest.raises(TimeoutError, match="timed out"):
                mock_session.wait(timeout=0.1)


class TestCloseMethod:
    """Tests for close() method."""

    def test_close_can_be_called_from_handler(self, mock_session):
        """Test that close() can be called from within a frame handler."""
        calls = []

        @mock_session.on_frame
        def handler(frame, metadata):
            calls.append(1)
            mock_session.close()

        # Put frames in queue (use put_nowait to avoid blocking on full queue)
        for i in range(5):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_metadata = VideoMetadata(frame_id=i, received_at=datetime.now())
            mock_session._video_queue.put_nowait((test_frame, test_metadata))

        mock_session._state = SessionState.STARTED
        mock_session.run()

        # Should have stopped after first frame
        assert len(calls) == 1
        assert mock_session._state == SessionState.CLOSED
