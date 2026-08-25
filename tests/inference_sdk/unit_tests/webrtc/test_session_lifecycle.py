"""Unit tests for WebRTC session lifecycle management."""

import asyncio
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference_sdk.webrtc.session import SessionState, VideoMetadata, WebRTCSession

THREAD_TIMEOUT = 5


class AsyncPeerConnectionStub:
    """Async peer connection with deterministic close controls."""

    def __init__(self, release=None, error=None):
        self.close_calls = 0
        self.close_started = threading.Event()
        self.release = release
        self.error = error

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error


class AsyncSourceStub:
    """Async stream source that records cleanup calls."""

    def __init__(self):
        self.cleanup_calls = 0
        self.cleanup_started = threading.Event()

    async def cleanup(self):
        self.cleanup_calls += 1
        self.cleanup_started.set()


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


@pytest.fixture
def running_session(mock_session):
    """Attach a real background asyncio loop to a started session."""
    loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop_started.set()
        loop.run_forever()
        loop.close()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    peer_connection = AsyncPeerConnectionStub()
    source = AsyncSourceStub()

    mock_session._loop = loop
    mock_session._loop_thread = loop_thread
    mock_session._pc = peer_connection
    mock_session._source = source
    mock_session._state = SessionState.STARTED
    loop_thread.start()
    assert loop_started.wait(timeout=THREAD_TIMEOUT)

    yield mock_session, peer_connection, source

    if loop_thread.is_alive():
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=THREAD_TIMEOUT)


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

    def test_on_error_registers_handler(self, mock_session):
        """Test registering per-frame error handler."""

        @mock_session.on_error
        def handle_err(errors, metadata):
            pass

        assert handle_err in mock_session._error_handlers

    def test_parse_video_metadata_attaches_errors(self, mock_session):
        """Errors passed alongside metadata are surfaced on VideoMetadata."""
        md = mock_session._parse_video_metadata(
            {"frame_id": 7, "received_at": datetime.now().isoformat()},
            errors=["workflow block X failed"],
        )
        assert md is not None
        assert md.frame_id == 7
        assert md.errors == ["workflow block X failed"]

    def test_video_metadata_default_errors_is_empty(self):
        """VideoMetadata constructed without errors defaults to an empty list."""
        md = VideoMetadata(frame_id=1, received_at=datetime.now())
        assert md.errors == []


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

    def test_close_from_event_loop_returns_and_cleans_up(self, running_session):
        """Test that close() does not block its own event loop."""
        session, peer_connection, source = running_session
        close_returned = threading.Event()

        def close_from_event_loop():
            session.close()
            close_returned.set()

        session._loop.call_soon_threadsafe(close_from_event_loop)

        assert close_returned.wait(timeout=THREAD_TIMEOUT)
        assert session._close_done.wait(timeout=THREAD_TIMEOUT)
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1
        assert not session._loop_thread.is_alive()

    def test_close_from_event_loop_does_not_wait_for_state_lock(self, running_session):
        """Test that callback shutdown can return while startup owns the state lock."""
        session, peer_connection, source = running_session
        close_returned = threading.Event()

        def close_from_event_loop():
            session.close()
            close_returned.set()

        session._state_lock.acquire()

        try:
            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert close_returned.wait(timeout=THREAD_TIMEOUT)
        finally:
            session._state_lock.release()

        assert session._close_done.wait(timeout=THREAD_TIMEOUT)
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_run_waits_for_event_loop_close_to_finish(self, running_session):
        """Test that context-manager exit waits for callback-initiated cleanup."""
        session, peer_connection, source = running_session
        release_peer_close = asyncio.Event()
        peer_connection.release = release_peer_close
        video_waiting = threading.Event()
        run_finished = threading.Event()
        run_errors = []
        original_get = session._video_queue.get

        def get_video_frame():
            video_waiting.set()
            return original_get()

        def run_session():
            try:
                session.run()
            except Exception as error:
                run_errors.append(error)
            finally:
                run_finished.set()

        with patch.object(session._video_queue, "get", side_effect=get_video_frame):
            run_thread = threading.Thread(target=run_session, daemon=True)
            run_thread.start()
            try:
                assert video_waiting.wait(timeout=THREAD_TIMEOUT)

                session._loop.call_soon_threadsafe(session.close)
                assert peer_connection.close_started.wait(timeout=THREAD_TIMEOUT)
                assert not run_finished.is_set()

                session._loop.call_soon_threadsafe(release_peer_close.set)
                assert run_finished.wait(timeout=THREAD_TIMEOUT)
            finally:
                if session._loop.is_running():
                    session._loop.call_soon_threadsafe(session.close)
                    session._loop.call_soon_threadsafe(release_peer_close.set)
                run_thread.join(timeout=THREAD_TIMEOUT)

        assert not run_thread.is_alive()
        assert not run_errors
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_event_loop_close_returns_during_external_cleanup(self, running_session):
        """Test that a loop-thread duplicate never waits for an external owner."""
        session, peer_connection, source = running_session
        release_peer_close = asyncio.Event()
        peer_connection.release = release_peer_close
        external_close_finished = threading.Event()
        loop_close_returned = threading.Event()

        def close_externally():
            session.close()
            external_close_finished.set()

        def close_from_event_loop():
            session.close()
            loop_close_returned.set()

        external_thread = threading.Thread(target=close_externally, daemon=True)
        external_thread.start()
        try:
            assert peer_connection.close_started.wait(timeout=THREAD_TIMEOUT)

            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert loop_close_returned.wait(timeout=THREAD_TIMEOUT)

            session._loop.call_soon_threadsafe(release_peer_close.set)
            assert external_close_finished.wait(timeout=THREAD_TIMEOUT)
        finally:
            if session._loop.is_running():
                session._loop.call_soon_threadsafe(release_peer_close.set)
            external_thread.join(timeout=THREAD_TIMEOUT)

        assert not external_thread.is_alive()
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_close_failure_still_cleans_source_and_completes(self, running_session):
        """Test that later cleanup runs when peer connection shutdown fails."""
        session, peer_connection, source = running_session
        peer_connection.error = RuntimeError("peer close failed")

        with pytest.raises(RuntimeError, match="peer close failed"):
            session.close()

        assert source.cleanup_calls == 1
        assert session._close_done.is_set()
        assert not session._loop_thread.is_alive()

    def test_background_close_failure_is_logged(self, running_session):
        """Test that callback cleanup failures are logged after the callback returns."""
        session, peer_connection, source = running_session
        peer_connection.error = RuntimeError("peer close failed")
        close_returned = threading.Event()
        error_logged = threading.Event()

        def close_from_event_loop():
            session.close()
            close_returned.set()

        with patch(
            "inference_sdk.webrtc.session.logger.exception",
            side_effect=lambda *args, **kwargs: error_logged.set(),
        ) as log_exception:
            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert close_returned.wait(timeout=THREAD_TIMEOUT)
            assert session._close_done.wait(timeout=THREAD_TIMEOUT)
            assert error_logged.wait(timeout=THREAD_TIMEOUT)

        assert source.cleanup_calls == 1
        log_exception.assert_called_once_with("Failed to close WebRTC session")

    def test_background_close_can_retry_when_thread_start_fails(self, running_session):
        """Test that a failed helper start does not strand later close callers."""
        session, peer_connection, source = running_session
        close_returned = threading.Event()
        close_errors = []

        def close_from_event_loop():
            try:
                session.close()
            except RuntimeError as error:
                close_errors.append(error)
            finally:
                close_returned.set()

        with patch(
            "inference_sdk.webrtc.session.threading.Thread.start",
            side_effect=RuntimeError("thread start failed"),
        ):
            session._loop.call_soon_threadsafe(close_from_event_loop)
            assert close_returned.wait(timeout=THREAD_TIMEOUT)

        session.close()

        assert str(close_errors[0]) == "thread start failed"
        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_close_cleans_resources_when_video_queue_is_full(self, running_session):
        """Test cleanup when the end sentinel cannot be added to a full queue."""
        session, peer_connection, source = running_session
        while not session._video_queue.full():
            session._video_queue.put_nowait(MagicMock())

        session.close()

        assert peer_connection.close_calls == 1
        assert source.cleanup_calls == 1

    def test_close_without_peer_connection_still_cleans_source(self, running_session):
        """Test source cleanup when connection setup did not assign a peer."""
        session, _, source = running_session
        session._pc = None

        session.close()

        assert source.cleanup_calls == 1
        assert session._close_done.is_set()
        assert not session._loop_thread.is_alive()
